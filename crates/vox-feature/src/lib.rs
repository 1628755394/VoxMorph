//! VoxMorph 特征提取：HuBERT / ContentVec 内容特征 + PYIN 基频。
//!
//! 本 crate 把原始音频帧转换为推理所需的"内容 token"与 F0 曲线，
//! 是 AI 变声路线（方案 B）的入口。
//!
//! # 流程
//!
//! ```text
//! Frame(f32 pcm) -> [HuBERT ONNX] -> content features (Tensor)
//!                 -> [PYIN]        -> f0 curve (Vec<f32>)
//! ```
//!
//! # 实现里程碑
//!
//! - M3: 接入 HuBERT ONNX，输出 content features
//! - M3: PYIN 基频提取（可放在 `vox-dsp`，本 crate 仅做编排）
//!
//! # 约束
//!
//! 特征提取在专用 OS 线程运行，预算 <20ms/帧。通过 [`vox_infer::InferenceSession`]
//! 调用 ONNX，不直接依赖 `ort`（解耦推理后端）。

pub mod extractor;
pub mod rvc;
pub mod sessions;
pub mod stream;

pub use extractor::FeatureExtractor;
pub use rvc::{
    align_up, feature_len_for_samples, keep_tail_in_place, ms_to_samples,
    onnx_silence_front_feature_frames, rmvpe_model_input_samples_16k, samples_between_rates,
    FeatureTensor, Rounding, CONTENTVEC_CONTEXT_ALIGN_SAMPLES, EMBEDDER_SAMPLE_RATE,
    RMVPE_BUCKET_FRAMES, RMVPE_FRAME_SAMPLES_16K, RMVPE_GUARD_FRAMES, RVC_SAMPLE_RATE,
};
pub use sessions::{ContentVecSession, RmvpeSession, RvcModelSession};

use thiserror::Error;
use vox_core::VoxError;

/// 特征提取错误。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FeatureError {
    /// 推理后端错误。
    #[error(transparent)]
    Infer(#[from] vox_infer::InferError),
    /// 输入帧无效。
    #[error("invalid feature input: {0}")]
    InvalidInput(String),
    /// 来自核心层的错误。
    #[error(transparent)]
    Core(#[from] VoxError),
}
