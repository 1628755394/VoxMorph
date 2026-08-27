//! RVC 模型 session 封装：ContentVec + RMVPE + RVC 三个 ONNX session。
//!
//! 借鉴 vc-rs 的 `sessions.rs`，适配 VoxMorph 的 `InferenceSession` trait。
//! 每个 session 封装特定模型的推理编排，复用缓冲避免每帧 alloc。
//!
//! # 模型角色
//!
//! - **ContentVec**（embedder）：音频 → content features `[1, frames, 256/768]`
//! - **RMVPE**（F0 估计）：音频 → F0 曲线 `Vec<f32>`
//! - **RVC**（生成模型）：content features + pitch + pitchf → 音频波形

use vox_core::{InferenceSession, Tensor};
use vox_infer::InferError;

use crate::rvc::FeatureTensor;
use crate::FeatureError;

/// ContentVec（HuBERT）embedder session。
///
/// 输入：`[1, T]` 16kHz 单声道音频
/// 输出：`[1, frames, channels]` content features
pub struct ContentVecSession<S: InferenceSession> {
    session: S,
    /// 预期输出通道数（ContentVec 通常 256 或 768）。
    expected_channels: i64,
}

impl<S: InferenceSession> ContentVecSession<S> {
    /// 构造 ContentVec session。
    pub fn new(session: S, expected_channels: i64) -> Self {
        Self {
            session,
            expected_channels,
        }
    }

    /// 提取 content features，写入复用的 `FeatureTensor` 缓冲。
    ///
    /// `audio_16k` 为 16kHz 单声道样本。输出 shape = `[1, frames, channels]`。
    pub fn extract_into(
        &mut self,
        audio_16k: &[f32],
        out: &mut FeatureTensor,
    ) -> Result<(), FeatureError> {
        let input = Tensor::f32(audio_16k.to_vec(), vec![1, audio_16k.len()]);

        let outputs = self
            .session
            .run(std::slice::from_ref(&input))
            .map_err(|e| FeatureError::Infer(InferError::Runtime(e.to_string())))?;

        let result = outputs.into_iter().next().ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime(
                "contentvec: no output from session".into(),
            ))
        })?;

        // 提取 f32 数据到 FeatureTensor。
        let data = result.as_f32().ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime("contentvec output is not f32".into()))
        })?;
        out.data.clear();
        out.data.extend_from_slice(data);
        out.shape = result.shape.into_iter().map(|d| d as i64).collect();

        // 验证 shape。
        if out.shape.len() != 3 {
            return Err(FeatureError::Infer(InferError::ShapeMismatch(format!(
                "contentvec output must be rank-3, got rank {}",
                out.shape.len()
            ))));
        }
        if out.shape[2] != self.expected_channels {
            tracing::warn!(
                expected = self.expected_channels,
                actual = out.shape[2],
                "contentvec output channel count mismatch"
            );
        }

        Ok(())
    }
}

/// RMVPE F0 估计 session。
///
/// 输入：`[1, T]` 或 `[T]` 16kHz 单声道音频
/// 输出：F0 曲线 `Vec<f32>`（Hz，无声段为 0.0）
pub struct RmvpeSession<S: InferenceSession> {
    session: S,
}

impl<S: InferenceSession> RmvpeSession<S> {
    /// 构造 RMVPE session。
    pub fn new(session: S) -> Self {
        Self { session }
    }

    /// 估计 F0 曲线。
    ///
    /// `audio_16k` 为 16kHz 单声道样本。返回 F0（Hz）列表，
    /// 长度由模型决定（通常 = feature_len_for_samples(T, 16000)）。
    pub fn estimate_f0(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>, FeatureError> {
        let input = Tensor::f32(audio_16k.to_vec(), vec![1, audio_16k.len()]);

        let outputs = self
            .session
            .run(std::slice::from_ref(&input))
            .map_err(|e| FeatureError::Infer(InferError::Runtime(e.to_string())))?;

        // RMVPE 通常输出 `[1, frames]` 或 `[frames]` 的 F0。
        let result = outputs.into_iter().next().ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime("rmvpe: no output from session".into()))
        })?;

        result.as_f32().map(|d| d.to_vec()).ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime("rmvpe output is not f32".into()))
        })
    }
}

/// RVC 生成模型 session。
///
/// 输入（RVC 标准格式）：
/// - `features`: content features `[1, frames, 256]`（已 2x 重复）
/// - `pitch`: coarse pitch bins `[1, frames]`（i64 量化）
/// - `pitchf`: 连续 F0 `[1, frames]`（f32 Hz）
/// - `sid`: speaker ID `[1]`（i64）
///
/// 输出：音频波形 `[1, T]` 或 `[T]`
pub struct RvcModelSession<S: InferenceSession> {
    session: S,
    /// RVC 模型输出采样率（通常 48kHz，可由元数据覆盖）。
    rvc_sample_rate: u32,
}

impl<S: InferenceSession> RvcModelSession<S> {
    /// 构造 RVC model session。
    pub fn new(session: S, rvc_sample_rate: u32) -> Self {
        Self {
            session,
            rvc_sample_rate,
        }
    }

    /// 获取输出采样率。
    pub fn sample_rate(&self) -> u32 {
        self.rvc_sample_rate
    }

    /// 执行 RVC 推理。
    ///
    /// # 输入张量顺序
    /// 1. content features `[1, frames, 256]`（f32）
    /// 2. coarse pitch `[1, frames]`（i64）
    /// 3. continuous F0 `[1, frames]`（f32）
    /// 4. speaker ID `[1]`（i64）
    pub fn convert(
        &mut self,
        features: &FeatureTensor,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
    ) -> Result<Vec<f32>, FeatureError> {
        let frames = features.shape.get(1).copied().unwrap_or(0) as usize;
        let channels = features.shape.get(2).copied().unwrap_or(0) as usize;

        // 构造输入张量。
        // RVC 模型输入顺序：features, pitch, pitchf, sid
        let feat_tensor = Tensor::f32(features.data.clone(), vec![1, frames, channels]);
        let pitch_tensor = Tensor::i64(pitch.to_vec(), vec![1, frames]);
        let pitchf_tensor = Tensor::f32(pitchf.to_vec(), vec![1, frames]);
        let sid_tensor = Tensor::i64(vec![speaker_id], vec![1]);

        let inputs = [feat_tensor, pitch_tensor, pitchf_tensor, sid_tensor];
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| FeatureError::Infer(InferError::Runtime(e.to_string())))?;

        let result = outputs.into_iter().next().ok_or_else(|| {
            FeatureError::Infer(InferError::Runtime(
                "rvc model: no output from session".into(),
            ))
        })?;

        result
            .as_f32()
            .map(|d| d.to_vec())
            .ok_or_else(|| FeatureError::Infer(InferError::Runtime("rvc output is not f32".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_infer::MockSession;

    #[test]
    fn contentvec_extract_with_mock() {
        let mut session = ContentVecSession::new(MockSession::constant(vec![1, 5, 256], 0.5), 256);
        let audio = vec![0.1; 1600]; // 100ms @ 16kHz
        let mut feat = FeatureTensor::default();
        session.extract_into(&audio, &mut feat).unwrap();
        assert_eq!(feat.shape, vec![1, 5, 256]);
        assert!(feat.data.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn contentvec_extract_identity() {
        // identity mock 返回输入 shape [1, 4] → rank-2，应报错
        let mut session = ContentVecSession::new(MockSession::identity(), 4);
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let mut feat = FeatureTensor::default();
        let result = session.extract_into(&audio, &mut feat);
        assert!(result.is_err(), "rank-2 identity output should error");
    }

    #[test]
    fn rmvpe_estimate_f0_with_mock() {
        let mut session = RmvpeSession::new(MockSession::constant(vec![1, 10], 200.0));
        let audio = vec![0.5; 1600];
        let f0 = session.estimate_f0(&audio).unwrap();
        assert_eq!(f0.len(), 10);
        assert!(f0.iter().all(|&v| (v - 200.0).abs() < 1e-6));
    }

    #[test]
    fn rvc_model_convert_with_mock() {
        let mut session = RvcModelSession::new(MockSession::constant(vec![1, 4800], 0.3), 48000);
        let features = FeatureTensor {
            data: vec![0.5; 500], // 5 frames * 100 channels (doesn't matter for mock)
            shape: vec![1, 5, 100],
        };
        let pitch = vec![100i64; 5];
        let pitchf = vec![200.0f32; 5];
        let output = session.convert(&features, &pitch, &pitchf, 0).unwrap();
        assert_eq!(output.len(), 4800);
        assert!(output.iter().all(|&v| (v - 0.3).abs() < 1e-6));
    }

    #[test]
    fn rvc_model_sample_rate() {
        let session = RvcModelSession::new(MockSession::identity(), 48000);
        assert_eq!(session.sample_rate(), 48000);
    }

    #[test]
    fn contentvec_wrong_rank_errors() {
        // Mock 返回 rank-2 输出。
        let mut session = ContentVecSession::new(MockSession::identity(), 4);
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let mut feat = FeatureTensor::default();
        let result = session.extract_into(&audio, &mut feat);
        assert!(result.is_err(), "rank-2 output should error");
    }
}
