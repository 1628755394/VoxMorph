//! Feature Stage：audio Frame → content features。
//!
//! # 流程
//!
//! 1. 接收 `Frame`（16kHz 单声道音频样本）
//! 2. 调用 `FeatureExtractor` 执行 HuBERT 推理
//! 3. 输出 `Frame`（samples 存放展平的 content features，供下游 ConvertStage 消费）
//!
//! # 设计说明
//!
//! - 输入必须是 16kHz 单声道（由前置预处理阶段保证）
//! - 输出 `Frame.samples` 是展平的 content features（`[T, feat_dim]` → 1D）
//! - 推理错误降级为静音（输出零向量），不 panic
//! - 复用 FeatureExtractor 的 session（HuBERT 模型）

use vox_core::{Frame, VoxError};
use vox_feature::FeatureExtractor;

use crate::Stage;

/// Feature Stage：audio Frame → content features（HuBERT 推理）。
///
/// 包装 `FeatureExtractor` 使其实现 `Stage` trait。
pub struct FeatureStage<S: vox_core::InferenceSession> {
    extractor: FeatureExtractor<S>,
    /// 预分配的输出 buffer（复用）。
    output_buffer: Vec<f32>,
}

impl<S: vox_core::InferenceSession> FeatureStage<S> {
    /// 构造 FeatureStage。
    pub fn new(session: S) -> Self {
        Self {
            extractor: FeatureExtractor::new(session),
            output_buffer: Vec::new(),
        }
    }

    /// 获取目标采样率（HuBERT 要求 16kHz）。
    pub fn target_sample_rate(&self) -> u32 {
        self.extractor.target_sample_rate()
    }
}

impl<S: vox_core::InferenceSession> Stage for FeatureStage<S> {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        // 调用 FeatureExtractor 提取 content features。
        let features = match self.extractor.extract(input) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "feature extraction failed, outputting silence");
                // 降级为静音（零特征向量）。
                output.samples = vec![0.0; input.samples.len()];
                output.sample_rate = input.sample_rate;
                output.channels = input.channels;
                output.timestamp = input.timestamp;
                return Ok(());
            }
        };

        // 复用 output buffer。
        self.output_buffer.clear();
        self.output_buffer.extend_from_slice(&features.data);

        // 输出 Frame：samples = 展平的 content features。
        output.samples = std::mem::take(&mut self.output_buffer);
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;

        Ok(())
    }

    fn reset(&mut self) {
        self.output_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_infer::MockSession;

    fn make_frame(samples: Vec<f32>) -> Frame {
        Frame {
            samples,
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        }
    }

    #[test]
    fn feature_stage_runs_extraction() {
        // Mock: identity 返回输入的 shape 和 data。
        let mut stage = FeatureStage::new(MockSession::identity());
        let input = make_frame(vec![0.1, 0.2, 0.3, 0.4]);
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        // identity mock 返回 [1, 4] → 展平为 4 个 f32。
        assert_eq!(output.samples, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(output.sample_rate, 16000);
    }

    #[test]
    fn feature_stage_with_zeros_mock() {
        let mut stage = FeatureStage::new(MockSession::zeros(vec![1, 2, 768]));
        let input = make_frame(vec![0.5; 320]);
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert_eq!(output.samples.len(), 2 * 768);
        assert!(output.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn wrong_sample_rate_outputs_silence() {
        let mut stage = FeatureStage::new(MockSession::identity());
        let input = Frame {
            samples: vec![0.1, 0.2],
            sample_rate: 48000, // 错误采样率
            channels: 1,
            timestamp: 0,
        };
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        // 采样率不匹配 → 降级为静音。
        assert!(output.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn preserves_timestamp() {
        let mut stage = FeatureStage::new(MockSession::identity());
        let input = Frame {
            samples: vec![0.1, 0.2],
            sample_rate: 16000,
            channels: 1,
            timestamp: 999,
        };
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert_eq!(output.timestamp, 999);
    }

    #[test]
    fn target_sample_rate_is_16k() {
        let stage = FeatureStage::new(MockSession::identity());
        assert_eq!(stage.target_sample_rate(), 16000);
    }

    #[test]
    fn reset_clears_buffer() {
        let mut stage = FeatureStage::new(MockSession::identity());
        let input = make_frame(vec![0.1, 0.2]);
        let mut output = Frame::zero(16000, 1, 0);
        stage.process(&input, &mut output).unwrap();
        assert!(!output.samples.is_empty());

        stage.reset();
        // reset 后仍可正常处理。
        stage.process(&input, &mut output).unwrap();
        assert!(!output.samples.is_empty());
    }
}
