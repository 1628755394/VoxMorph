//! WAV 文件编解码：用 `hound` 实现 [`vox_core::AudioSource`]（解码读取）
//! 与 [`vox_core::AudioSink`]（编码写入）。
//!
//! # 数据格式
//!
//! `AudioSource` / `AudioSink` 操作交错 f32 样本。`hound` 原生支持 i16/i32/f32
//! WAV，本模块统一用 f32 中间格式：读取时转换到 f32，写入时从 f32 转换。
//!
//! # 离线用途
//!
//! WAV 文件读写不在音频线程调用，允许分配。`read` 到达文件末尾时返回 `Ok(0)`
//! 表示流末尾（与 `AudioSource` 约定一致）。

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use vox_core::{AudioSink, AudioSource, VoxError};

use crate::AudioError;

/// WAV 文件读取器，实现 [`AudioSource`]。
///
/// 读取时将样本归一化到 f32 [-1.0, 1.0]，交错布局。
pub struct WavSource {
    reader: WavReader<BufReader<File>>,
    sample_rate: u32,
    channels: u16,
}

impl WavSource {
    /// 打开 WAV 文件用于读取。
    ///
    /// # Errors
    /// 文件打开或 WAV 头解析失败返回 [`AudioError`]。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let reader =
            WavReader::open(path.as_ref()).map_err(|e| AudioError::Decode(e.to_string()))?;
        let spec = reader.spec();
        if spec.sample_format != SampleFormat::Float && spec.sample_format != SampleFormat::Int {
            return Err(AudioError::Unsupported(format!(
                "wav sample format {:?} not supported",
                spec.sample_format
            )));
        }
        Ok(Self {
            reader,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }
}

impl AudioSource for WavSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize, VoxError> {
        let samples = self.reader.samples::<f32>();
        let mut count = 0;
        for (s, slot) in samples.zip(out.iter_mut()) {
            match s {
                Ok(v) => {
                    *slot = v;
                    count += 1;
                }
                Err(hound::Error::FormatError(_)) => break,
                Err(e) => return Err(VoxError::audio(e.to_string())),
            }
        }
        Ok(count)
    }
}

/// WAV 文件写入器，实现 [`AudioSink`]。
///
/// 写入时将 f32 样本编码为 32-bit float WAV。drop 时自动 finalize WAV 头。
pub struct WavSink {
    writer: WavWriter<BufWriter<File>>,
    sample_rate: u32,
    channels: u16,
}

impl WavSink {
    /// 创建 WAV 文件用于写入，32-bit float 格式。
    ///
    /// # Errors
    /// 文件创建或 WAV 头写入失败返回 [`AudioError`]。
    pub fn create(
        path: impl AsRef<Path>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, AudioError> {
        if channels == 0 {
            return Err(AudioError::Unsupported("channels must be non-zero".into()));
        }
        if sample_rate == 0 {
            return Err(AudioError::Unsupported(
                "sample rate must be non-zero".into(),
            ));
        }
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let writer = WavWriter::create(path.as_ref(), spec)
            .map_err(|e| AudioError::Encode(e.to_string()))?;
        Ok(Self {
            writer,
            sample_rate,
            channels,
        })
    }

    /// Finalize WAV 文件，写入完整头信息。消耗 `self`（`hound::WavWriter::finalize` 取所有权）。
    ///
    /// # Errors
    /// 写入失败返回 [`AudioError`]。
    pub fn finalize(self) -> Result<(), AudioError> {
        self.writer
            .finalize()
            .map_err(|e| AudioError::Encode(e.to_string()))
    }
}

impl AudioSink for WavSink {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn write(&mut self, samples: &[f32]) -> Result<(), VoxError> {
        for &s in samples {
            self.writer
                .write_sample(s)
                .map_err(|e| VoxError::audio(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrip() {
        let tmp = std::env::temp_dir().join("voxmorph_wav_roundtrip.wav");
        // 1 秒 440Hz 正弦波，单声道，16000 Hz
        let sr = 16000u32;
        let n = sr as usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        // 写入
        {
            let mut sink = WavSink::create(&tmp, sr, 1).unwrap();
            sink.write(&input).unwrap();
            sink.finalize().unwrap();
        }

        // 读取
        let mut source = WavSource::open(&tmp).unwrap();
        assert_eq!(source.sample_rate(), sr);
        assert_eq!(source.channels(), 1);

        let mut buf = vec![0.0f32; n];
        let read = source.read(&mut buf).unwrap();
        assert_eq!(read, n);

        // 验证数据近似（f32 WAV 精度无损）
        let max_err = input
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "roundtrip max error {max_err}");

        // 再读应返回 0（EOF）
        let mut eof_buf = [0.0f32; 16];
        let eof_read = source.read(&mut eof_buf).unwrap();
        assert_eq!(eof_read, 0, "expected EOF");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn rejects_zero_channels_on_create() {
        let tmp = std::env::temp_dir().join("voxmorph_wav_bad.wav");
        assert!(WavSink::create(&tmp, 16000, 0).is_err());
    }

    #[test]
    fn rejects_zero_sample_rate_on_create() {
        let tmp = std::env::temp_dir().join("voxmorph_wav_bad2.wav");
        assert!(WavSink::create(&tmp, 0, 1).is_err());
    }

    #[test]
    fn stereo_roundtrip() {
        let tmp = std::env::temp_dir().join("voxmorph_wav_stereo.wav");
        let sr = 8000u32;
        let n_frames = 100usize;
        // 交错立体声：左声道 = 0.5，右声道 = -0.5
        let input: Vec<f32> = (0..n_frames).flat_map(|_| [0.5f32, -0.5f32]).collect();

        {
            let mut sink = WavSink::create(&tmp, sr, 2).unwrap();
            sink.write(&input).unwrap();
            sink.finalize().unwrap();
        }

        let mut source = WavSource::open(&tmp).unwrap();
        assert_eq!(source.channels(), 2);
        let mut buf = vec![0.0f32; n_frames * 2];
        let read = source.read(&mut buf).unwrap();
        assert_eq!(read, n_frames * 2);

        let max_err = input
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "stereo roundtrip max error {max_err}");

        let _ = std::fs::remove_file(&tmp);
    }
}
