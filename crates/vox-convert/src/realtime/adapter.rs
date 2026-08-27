//! FrameAdapter：在 `AudioSource`/`AudioSink` 与 `Frame` 之间转换。
//!
//! # 职责
//!
//! - **capture**: 从 `AudioSource` 读取固定大小样本块，打包成 `Frame`
//! - **output**: 从 `Frame` 提取样本，写入 `AudioSink`
//!
//! 采样率/声道数由 source/sink 决定，adapter 只做打包/拆包，不重采样。
//! 重采样由管线前置阶段或 cpal 配置处理（规范：不依赖 cpal 自动转换）。

use vox_core::VoxError;
use vox_core::{AudioSink, AudioSource, Frame};

/// FrameAdapter 配置。
#[derive(Debug, Clone)]
pub struct FrameAdapterConfig {
    /// 每帧样本数（每声道）。
    pub frame_size: usize,
    /// 声道数。
    pub channels: u16,
    /// 采样率。
    pub sample_rate: u32,
}

impl FrameAdapterConfig {
    /// 从 source 的参数构造。
    pub fn from_source(source: &dyn AudioSource) -> Self {
        Self {
            frame_size: 512, // 默认 32ms @ 16kHz
            channels: source.channels(),
            sample_rate: source.sample_rate(),
        }
    }

    /// 从 sink 的参数构造。
    pub fn from_sink(sink: &dyn AudioSink) -> Self {
        Self {
            frame_size: 512,
            channels: sink.channels(),
            sample_rate: sink.sample_rate(),
        }
    }
}

/// 帧适配器：在 AudioSource/AudioSink 与 Frame 之间转换。
///
/// capture 侧维护一个累积缓冲，从 source 读取的样本累积到 frame_size 后打包成 Frame。
/// output 侧将 Frame 的样本直接写入 sink。
pub struct FrameAdapter {
    config: FrameAdapterConfig,
    /// 累积缓冲（capture 侧）。
    accum: Vec<f32>,
    /// 当前时间戳（样本数）。
    timestamp: u64,
}

impl FrameAdapter {
    /// 构造适配器。
    pub fn new(config: FrameAdapterConfig) -> Self {
        let frame_samples = config.frame_size * config.channels as usize;
        Self {
            config,
            accum: Vec::with_capacity(frame_samples * 2),
            timestamp: 0,
        }
    }

    /// 从 source 读取数据，返回完整的 Frame（可能多个）。
    ///
    /// 非阻塞：source 无数据时返回空 Vec。
    ///
    /// # Errors
    /// source 读取失败返回 [`VoxError`]。
    pub fn capture(&mut self, source: &mut dyn AudioSource) -> Result<Vec<Frame>, VoxError> {
        let frame_samples = self.config.frame_size * self.config.channels as usize;
        let mut temp = vec![0.0_f32; frame_samples];
        let mut frames = Vec::new();

        loop {
            let n = source.read(&mut temp)?;
            if n == 0 {
                break;
            }
            self.accum.extend_from_slice(&temp[..n]);

            // 当累积足够一帧时，打包输出。
            while self.accum.len() >= frame_samples {
                let frame_data: Vec<f32> = self.accum.drain(..frame_samples).collect();
                frames.push(Frame {
                    samples: frame_data,
                    sample_rate: self.config.sample_rate,
                    channels: self.config.channels,
                    timestamp: self.timestamp,
                });
                self.timestamp += self.config.frame_size as u64;
            }
        }

        Ok(frames)
    }

    /// 将一个 Frame 的样本写入 sink。
    ///
    /// # Errors
    /// sink 写入失败返回 [`VoxError`]（如 `Dropped` 背压）。
    pub fn output(&mut self, frame: &Frame, sink: &mut dyn AudioSink) -> Result<(), VoxError> {
        sink.write(&frame.samples)
    }

    /// 刷新累积缓冲中不足一帧的残余样本（补零到完整帧）。
    ///
    /// 用于流结束时输出最后的不完整帧。
    pub fn flush_remaining(&mut self) -> Option<Frame> {
        let frame_samples = self.config.frame_size * self.config.channels as usize;
        if self.accum.is_empty() {
            return None;
        }
        let mut samples = std::mem::take(&mut self.accum);
        samples.resize(frame_samples, 0.0);
        let frame = Frame {
            samples,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            timestamp: self.timestamp,
        };
        self.timestamp += self.config.frame_size as u64;
        Some(frame)
    }

    /// 重置适配器状态（清空累积缓冲、重置时间戳）。
    pub fn reset(&mut self) {
        self.accum.clear();
        self.timestamp = 0;
    }

    /// 当前累积的样本数。
    pub fn accumulated_samples(&self) -> usize {
        self.accum.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock AudioSource：从预置样本队列读取。
    struct MockSource {
        samples: Mutex<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
    }

    impl MockSource {
        fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
            Self {
                samples: Mutex::new(samples),
                sample_rate,
                channels,
            }
        }
    }

    impl AudioSource for MockSource {
        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }
        fn channels(&self) -> u16 {
            self.channels
        }
        fn read(&mut self, out: &mut [f32]) -> Result<usize, VoxError> {
            let mut samples = self.samples.lock().unwrap();
            let n = out.len().min(samples.len());
            out[..n].copy_from_slice(&samples[..n]);
            samples.drain(..n);
            Ok(n)
        }
    }

    /// Mock AudioSink：收集写入的样本。
    struct MockSink {
        collected: Mutex<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
    }

    impl MockSink {
        fn new(sample_rate: u32, channels: u16) -> Self {
            Self {
                collected: Mutex::new(vec![]),
                sample_rate,
                channels,
            }
        }

        fn collected(&self) -> Vec<f32> {
            self.collected.lock().unwrap().clone()
        }
    }

    impl AudioSink for MockSink {
        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }
        fn channels(&self) -> u16 {
            self.channels
        }
        fn write(&mut self, samples: &[f32]) -> Result<(), VoxError> {
            self.collected.lock().unwrap().extend_from_slice(samples);
            Ok(())
        }
    }

    #[test]
    fn capture_packs_into_frames() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        let mut source = MockSource::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 16000, 1);

        let frames = adapter.capture(&mut source).unwrap();
        assert_eq!(frames.len(), 2, "should produce 2 frames of 4 samples each");
        assert_eq!(frames[0].samples, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(frames[1].samples, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(frames[0].timestamp, 0);
        assert_eq!(frames[1].timestamp, 4);
    }

    #[test]
    fn capture_accumulates_partial_frames() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        let mut source = MockSource::new(vec![1.0, 2.0, 3.0], 16000, 1); // 不足一帧

        let frames = adapter.capture(&mut source).unwrap();
        assert!(frames.is_empty(), "no complete frame yet");
        assert_eq!(adapter.accumulated_samples(), 3);

        // flush 残余。
        let remaining = adapter.flush_remaining().unwrap();
        assert_eq!(remaining.samples, vec![1.0, 2.0, 3.0, 0.0]); // 补零
    }

    #[test]
    fn capture_stereo_frames() {
        let config = FrameAdapterConfig {
            frame_size: 2,
            channels: 2,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        // 立体声交错：[L0, R0, L1, R1, L2, R2, L3, R3]
        let mut source = MockSource::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 16000, 2);

        let frames = adapter.capture(&mut source).unwrap();
        assert_eq!(frames.len(), 2, "2 frames of 2 samples each (stereo)");
        assert_eq!(frames[0].samples, vec![1.0, 2.0, 3.0, 4.0]); // 2 samples * 2 ch
        assert_eq!(frames[0].channels, 2);
    }

    #[test]
    fn output_writes_to_sink() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        let mut sink = MockSink::new(16000, 1);

        let frame = Frame {
            samples: vec![0.5, 0.6, 0.7, 0.8],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        adapter.output(&frame, &mut sink).unwrap();
        assert_eq!(sink.collected(), vec![0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn capture_empty_source_returns_no_frames() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        let mut source = MockSource::new(vec![], 16000, 1);

        let frames = adapter.capture(&mut source).unwrap();
        assert!(frames.is_empty());
    }

    #[test]
    fn reset_clears_accumulation() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        let mut source = MockSource::new(vec![1.0, 2.0], 16000, 1);
        let _ = adapter.capture(&mut source).unwrap();
        assert_eq!(adapter.accumulated_samples(), 2);

        adapter.reset();
        assert_eq!(adapter.accumulated_samples(), 0);
        assert_eq!(adapter.timestamp, 0);
    }

    #[test]
    fn flush_empty_returns_none() {
        let config = FrameAdapterConfig {
            frame_size: 4,
            channels: 1,
            sample_rate: 16000,
        };
        let mut adapter = FrameAdapter::new(config);
        assert!(adapter.flush_remaining().is_none());
    }

    #[test]
    fn from_source_extracts_params() {
        let source = MockSource::new(vec![], 48000, 2);
        let config = FrameAdapterConfig::from_source(&source);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.frame_size, 512);
    }
}
