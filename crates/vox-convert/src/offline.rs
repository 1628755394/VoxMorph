//! 离线变声编排：文件解码 → DSP 处理 → 文件编码。
//!
//! M2 实现：WAV 读取 → pitch shift → WAV 写入。无重采样（pitch shift 保持
//! 采样率与时长）。后续里程碑加入特征提取 + 推理 + vocoder。

use std::path::Path;

use tracing::info;
use vox_core::{AudioSink, AudioSource};
use vox_dsp::pitch::PitchShifter;
use vox_io::wav::{WavSink, WavSource};

use crate::ConvertError;

/// 离线变声参数。
pub struct OfflineConvertParams {
    /// Pitch shift 半音数，正数升调、负数降调。
    pub semitones: f64,
}

impl Default for OfflineConvertParams {
    fn default() -> Self {
        Self { semitones: 0.0 }
    }
}

/// 离线变声器：读取 WAV → pitch shift → 写入 WAV。
///
/// M2 仅做 pitch shift。输入/输出采样率必须一致（pitch shift 不改变采样率）。
pub struct OfflineConverter;

impl OfflineConverter {
    /// 执行离线变声：`input_path` → `output_path`。
    ///
    /// 整段读入内存，pitch shift 后整段写出。适合离线 demo，不适合大文件
    /// （大文件需分块流式处理，留待后续里程碑）。
    ///
    /// # Errors
    /// 文件读写或 DSP 处理失败返回 [`ConvertError`]。
    pub fn convert(
        input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        params: &OfflineConvertParams,
    ) -> Result<(), ConvertError> {
        // 1. 解码 WAV。
        let mut source = WavSource::open(input_path.as_ref())?;
        let sr = source.sample_rate();
        let ch = source.channels();

        // 整段读入。
        let mut all_samples = Vec::new();
        let mut buf = vec![0.0f32; 4096];
        loop {
            let n = source.read(&mut buf)?;
            if n == 0 {
                break;
            }
            all_samples.extend_from_slice(&buf[..n]);
        }
        info!(
            input_samples = all_samples.len(),
            sample_rate = sr,
            channels = ch,
            "decoded input wav"
        );

        // 2. Pitch shift（跳过 0 semitones 的无操作情况）。
        let processed = if params.semitones == 0.0 {
            all_samples
        } else {
            let mut shifter = PitchShifter::new(sr, ch, params.semitones)?;
            let result = shifter.shift_buffer(&all_samples)?;
            info!(output_samples = result.len(), "pitch shift completed");
            result
        };

        // 3. 编码 WAV。
        let mut sink = WavSink::create(output_path.as_ref(), sr, ch)?;
        sink.write(&processed)?;
        sink.finalize()?;
        info!(output_samples = processed.len(), "encoded output wav");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_applies_pitch_shift() {
        let tmp_in = std::env::temp_dir().join("voxmorph_offline_in.wav");
        let tmp_out = std::env::temp_dir().join("voxmorph_offline_out.wav");

        // 生成 1 秒 440Hz 正弦波 at 16000 Hz，单声道。
        let sr = 16000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        // 写入测试输入 WAV。
        {
            let mut sink = WavSink::create(&tmp_in, sr, 1).unwrap();
            sink.write(&samples).unwrap();
            sink.finalize().unwrap();
        }

        // 执行 +5 semitones pitch shift。
        let params = OfflineConvertParams { semitones: 5.0 };
        OfflineConverter::convert(&tmp_in, &tmp_out, &params).unwrap();

        // 验证输出 WAV。
        let mut source = WavSource::open(&tmp_out).unwrap();
        assert_eq!(source.sample_rate(), sr);
        assert_eq!(source.channels(), 1);
        let mut out_buf = vec![0.0f32; n];
        let read = source.read(&mut out_buf).unwrap();
        assert!(
            (read as isize - n as isize).abs() <= 1024,
            "output length should approximate input, got {read} vs {n}"
        );

        let _ = std::fs::remove_file(&tmp_in);
        let _ = std::fs::remove_file(&tmp_out);
    }

    #[test]
    fn convert_zero_semitones_preserves_signal() {
        let tmp_in = std::env::temp_dir().join("voxmorph_offline_zero_in.wav");
        let tmp_out = std::env::temp_dir().join("voxmorph_offline_zero_out.wav");

        let sr = 8000u32;
        let n = 4096usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin())
            .collect();

        {
            let mut sink = WavSink::create(&tmp_in, sr, 1).unwrap();
            sink.write(&samples).unwrap();
            sink.finalize().unwrap();
        }

        let params = OfflineConvertParams { semitones: 0.0 };
        OfflineConverter::convert(&tmp_in, &tmp_out, &params).unwrap();

        let mut source = WavSource::open(&tmp_out).unwrap();
        let mut out_buf = vec![0.0f32; n];
        let read = source.read(&mut out_buf).unwrap();
        assert_eq!(read, n, "0 semitones should preserve length");

        // 0 semitones 走 passthrough 路径，数据应完全一致。
        let max_err = samples
            .iter()
            .zip(out_buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-6,
            "0 semitones passthrough max error {max_err}"
        );

        let _ = std::fs::remove_file(&tmp_in);
        let _ = std::fs::remove_file(&tmp_out);
    }
}
