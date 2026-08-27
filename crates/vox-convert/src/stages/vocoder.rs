//! Vocoder Stage：converted features → audio waveform。
//!
//! # 流程
//!
//! 1. 接收 `Frame`（samples 存放展平的 converted features）
//! 2. 重塑为模型期望的输入张量形状
//! 3. 调用 Vocoder ONNX 模型推理
//! 4. 输出音频波形 `Frame`（samples = 波形样本）
//!
//! # 设计说明
//!
//! - 输入 `Frame.samples` 是展平的 converted features（来自 ConvertStage）
//! - 输出 `Frame.samples` 是音频波形（供下游 output 线程播放）
//! - 推理错误降级为静音，不 panic
//! - 输出采样率由 Vocoder 模型决定（通常 16kHz 或 24kHz），需与播放设备匹配

use vox_core::{Frame, InferenceSession, Tensor, VoxError};

use crate::Stage;

/// Vocoder 模型输入布局。
#[derive(Debug, Clone)]
pub struct VocoderInputLayout {
    /// converted features 的形状（不含 batch 维），如 `[T, feat_dim]`。
    pub feature_shape: Vec<usize>,
    /// Vocoder 输出的采样率。
    pub output_sample_rate: u32,
}

impl VocoderInputLayout {
    /// 默认布局：features `[T, 256]`，输出 16kHz。
    pub fn default_16k() -> Self {
        Self {
            feature_shape: vec![/* T */ 0, 256],
            output_sample_rate: 16000,
        }
    }
}

/// Vocoder Stage：converted features → audio waveform。
///
/// 持有 Vocoder 模型的 `InferenceSession`。
/// 输入 Frame.samples 应为展平的 converted features。
pub struct VocoderStage<S: InferenceSession> {
    session: S,
    layout: VocoderInputLayout,
    /// 预分配的输出 buffer（复用）。
    output_buffer: Vec<f32>,
}

impl<S: InferenceSession> VocoderStage<S> {
    /// 构造 VocoderStage。
    pub fn new(session: S, layout: VocoderInputLayout) -> Self {
        Self {
            session,
            layout,
            output_buffer: Vec::new(),
        }
    }

    /// 获取输出采样率。
    pub fn output_sample_rate(&self) -> u32 {
        self.layout.output_sample_rate
    }

    /// 构造 Vocoder 模型的输入张量。
    fn build_input(&self, features: &[f32]) -> Tensor {
        let feat_dim = self.layout.feature_shape.get(1).copied().unwrap_or(256);
        let t_len = if features.len() % feat_dim == 0 {
            features.len() / feat_dim
        } else {
            features.len()
        };

        Tensor::f32(features.to_vec(), vec![1, t_len, feat_dim])
    }

    /// 静音输出（错误降级）。
    fn silent_output(&self, timestamp: u64, frame_samples: usize) -> Frame {
        Frame {
            samples: vec![0.0; frame_samples],
            sample_rate: self.layout.output_sample_rate,
            channels: 1,
            timestamp,
        }
    }
}

impl<S: InferenceSession> Stage for VocoderStage<S> {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        // 构造输入张量。
        let input_tensor = self.build_input(&input.samples);

        // 推理。
        let outputs = match self.session.run(std::slice::from_ref(&input_tensor)) {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "vocoder inference failed, outputting silence");
                *output = self.silent_output(input.timestamp, input.samples.len());
                return Ok(());
            }
        };

        // 取第一个输出张量作为音频波形。
        let waveform = match outputs.into_iter().next() {
            Some(t) => t,
            None => {
                tracing::error!("vocoder session returned no output");
                *output = self.silent_output(input.timestamp, input.samples.len());
                return Ok(());
            }
        };

        // 复用 output buffer。
        self.output_buffer.clear();
        let waveform_data = waveform
            .as_f32()
            .ok_or_else(|| vox_core::VoxError::infer("vocoder output is not f32".to_string()))?;
        self.output_buffer.extend_from_slice(waveform_data);

        // 输出 Frame：samples = 音频波形。
        output.samples = std::mem::take(&mut self.output_buffer);
        output.sample_rate = self.layout.output_sample_rate;
        output.channels = 1;
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
    use vox_infer::{MockSession, MockStrategy};

    fn make_frame(features: Vec<f32>) -> Frame {
        Frame {
            samples: features,
            sample_rate: 16000, // 特征帧率标识
            channels: 1,
            timestamp: 0,
        }
    }

    #[test]
    fn vocoder_runs_inference() {
        // Mock: 返回固定波形。
        let mut stage = VocoderStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 1024],
                value: 0.3,
            }),
            VocoderInputLayout::default_16k(),
        );

        let input = make_frame(vec![0.5; 256]); // 1 frame * 256 features
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert_eq!(
            output.samples.len(),
            1024,
            "should be 1024 waveform samples"
        );
        assert!(output.samples.iter().all(|&s| (s - 0.3).abs() < 1e-6));
        assert_eq!(output.sample_rate, 16000);
        assert_eq!(output.channels, 1);
    }

    #[test]
    fn no_output_outputs_silence() {
        // Mock: 返回空输出模拟"无输出"场景。
        let mut stage = VocoderStage::new(
            MockSession::new(MockStrategy::Custom(|_| vec![])),
            VocoderInputLayout::default_16k(),
        );

        let input = make_frame(vec![0.5; 256]);
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert!(output.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn output_sample_rate_matches_layout() {
        let stage = VocoderStage::new(
            MockSession::identity(),
            VocoderInputLayout {
                feature_shape: vec![0, 128],
                output_sample_rate: 24000,
            },
        );
        assert_eq!(stage.output_sample_rate(), 24000);
    }

    #[test]
    fn reset_clears_buffer() {
        let mut stage = VocoderStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 512],
                value: 0.5,
            }),
            VocoderInputLayout::default_16k(),
        );

        let input = make_frame(vec![0.5; 256]);
        let mut output = Frame::zero(16000, 1, 0);
        stage.process(&input, &mut output).unwrap();
        assert!(!output.samples.is_empty());

        stage.reset();
        // reset 后仍可正常处理。
        stage.process(&input, &mut output).unwrap();
        assert!(!output.samples.is_empty());
    }

    #[test]
    fn build_input_shapes() {
        let stage = VocoderStage::new(MockSession::identity(), VocoderInputLayout::default_16k());
        let tensor = stage.build_input(&[0.5; 512]); // 2 frames * 256
        assert_eq!(tensor.shape, vec![1, 2, 256]);
    }

    #[test]
    fn preserves_timestamp() {
        let mut stage = VocoderStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 100],
                value: 0.0,
            }),
            VocoderInputLayout::default_16k(),
        );

        let input = Frame {
            samples: vec![0.5; 256],
            sample_rate: 16000,
            channels: 1,
            timestamp: 12345,
        };
        let mut output = Frame::zero(16000, 1, 0);
        stage.process(&input, &mut output).unwrap();
        assert_eq!(output.timestamp, 12345);
    }
}
