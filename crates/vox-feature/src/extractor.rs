//! 特征提取编排：Frame → HuBERT ONNX → content features。
//!
//! # 流程
//!
//! 1. 预处理：多声道混音为单声道，重采样到 16kHz（HuBERT 要求）
//! 2. 构造输入张量：`[1, T]`（batch=1, time=T）
//! 3. 调用 [`InferenceSession::run`] 执行 HuBERT 推理
//! 4. 返回 content features 张量（shape 由模型决定，通常 `[1, T', 768]`）
//!
//! # 测试
//!
//! 用 [`vox_infer::MockSession`] 测试编排逻辑，不跑真实 ONNX 模型
//!（`test-mock-traits`）。

use vox_core::{Frame, InferenceSession, Tensor};
use vox_infer::InferError;

use crate::FeatureError;

/// HuBERT 模型要求的采样率。
const HUBERT_SAMPLE_RATE: u32 = 16000;

/// 特征提取器：封装 HuBERT 推理编排。
///
/// 持有 [`InferenceSession`] 的所有权（或 `Arc<Mutex<_>>`），复用 session
/// 而非每次新建（规范要求）。
pub struct FeatureExtractor<S: InferenceSession> {
    session: S,
    /// 目标采样率（固定 16kHz，HuBERT 要求）。
    target_sr: u32,
}

impl<S: InferenceSession> FeatureExtractor<S> {
    /// 构造特征提取器，绑定推理 session。
    pub fn new(session: S) -> Self {
        Self {
            session,
            target_sr: HUBERT_SAMPLE_RATE,
        }
    }

    /// 从音频帧提取 content features。
    ///
    /// # 预处理
    /// - 多声道 → 单声道（取均值）
    /// - 采样率不匹配时返回错误（M3 不做内嵌重采样，由管线前置阶段处理）
    ///
    /// # Errors
    /// - 帧采样率不匹配返回 [`FeatureError::InvalidInput`]
    /// - 推理失败返回 [`FeatureError::Infer`]
    pub fn extract(&mut self, frame: &Frame) -> Result<Tensor, FeatureError> {
        // 采样率校验（M3 不做内嵌重采样）。
        if frame.sample_rate != self.target_sr {
            return Err(FeatureError::InvalidInput(format!(
                "sample rate mismatch: expected {} got {}",
                self.target_sr, frame.sample_rate
            )));
        }

        // 多声道 → 单声道。
        let mono = to_mono(&frame.samples, frame.channels);
        let t_len = mono.len();

        // 构造输入张量 [1, T]。
        let input = Tensor::f32(mono, vec![1, t_len]);

        // 推理。
        let outputs = self
            .session
            .run(std::slice::from_ref(&input))
            .map_err(|e| FeatureError::Infer(InferError::Runtime(e.to_string())))?;

        // HuBERT 输出第一个张量即 content features。
        outputs.into_iter().next().ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime("no output from session".into()))
        })
    }

    /// 获取目标采样率。
    pub fn target_sample_rate(&self) -> u32 {
        self.target_sr
    }
}

/// 多声道交错样本 → 单声道（取均值）。
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let n_frames = samples.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let frame = &samples[i * ch..(i + 1) * ch];
        let avg = frame.iter().sum::<f32>() / ch as f32;
        mono.push(avg);
    }
    mono
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_infer::MockSession;

    #[test]
    fn extract_with_mock_identity_returns_input() {
        let mut extractor = FeatureExtractor::new(MockSession::identity());
        let frame = Frame {
            samples: vec![0.1, 0.2, 0.3, 0.4],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        let output = extractor.extract(&frame).unwrap();
        assert_eq!(output.shape, vec![1, 4]);
        assert_eq!(output.as_f32().unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn extract_rejects_wrong_sample_rate() {
        let mut extractor = FeatureExtractor::new(MockSession::identity());
        let frame = Frame {
            samples: vec![0.1, 0.2],
            sample_rate: 48000,
            channels: 1,
            timestamp: 0,
        };
        let result = extractor.extract(&frame);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, FeatureError::InvalidInput(_)));
    }

    #[test]
    fn extract_stereo_downmixes_to_mono() {
        // Mock 策略：返回输入的 shape 和 data（identity）。
        let mut extractor = FeatureExtractor::new(MockSession::identity());
        // 立体声：左 0.2，右 0.4 → 单声道 0.3
        let frame = Frame {
            samples: vec![0.2, 0.4, 0.2, 0.4],
            sample_rate: 16000,
            channels: 2,
            timestamp: 0,
        };
        let output = extractor.extract(&frame).unwrap();
        assert_eq!(output.shape, vec![1, 2]);
        let data = output.as_f32().unwrap();
        assert!((data[0] - 0.3).abs() < 1e-6);
        assert!((data[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn extract_with_mock_zeros_returns_zeros() {
        let mut extractor = FeatureExtractor::new(MockSession::zeros(vec![1, 2, 768]));
        let frame = Frame {
            samples: vec![0.5; 320],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        let output = extractor.extract(&frame).unwrap();
        assert_eq!(output.shape, vec![1, 2, 768]);
        assert!(output.as_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn extract_with_mock_constant_returns_constant() {
        let mut extractor = FeatureExtractor::new(MockSession::constant(vec![1, 3, 4], 0.7));
        let frame = Frame {
            samples: vec![0.1; 480],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        let output = extractor.extract(&frame).unwrap();
        assert_eq!(output.shape, vec![1, 3, 4]);
        assert!(output
            .as_f32()
            .unwrap()
            .iter()
            .all(|&v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn target_sample_rate_is_16k() {
        let extractor = FeatureExtractor::new(MockSession::identity());
        assert_eq!(extractor.target_sample_rate(), 16000);
    }

    #[test]
    fn to_mono_single_channel_passthrough() {
        let input = vec![1.0, 2.0, 3.0];
        let output = to_mono(&input, 1);
        assert_eq!(output, input);
    }

    #[test]
    fn to_mono_stereo_averages() {
        let input = vec![0.0, 1.0, 2.0, 3.0]; // 2 frames: [0,1], [2,3]
        let output = to_mono(&input, 2);
        assert_eq!(output, vec![0.5, 2.5]);
    }

    #[test]
    fn to_mono_empty_input() {
        let output = to_mono(&[], 2);
        assert!(output.is_empty());
    }
}
