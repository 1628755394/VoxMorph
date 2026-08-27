//! VoxMorph 顶层错误类型。
//!
//! 下游 crate 各自定义专用错误（如 `vox_io::AudioError`、`vox_infer::InferError`），
//! 并通过 `#[from]` 把 [`VoxError`] 链入，使 `?` 可在跨 crate 边界传播。

use thiserror::Error;

/// 核心 trait 共用的错误类型。
///
/// `#[non_exhaustive]` 保证未来新增变体不破坏下游匹配（`api-non-exhaustive`）。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VoxError {
    /// 音频源/汇读写失败。
    #[error("audio io failure: {0}")]
    Audio(String),
    /// 推理后端调用失败。
    #[error("inference failure: {0}")]
    Infer(String),
    /// 张量形状或 dtype 不匹配。
    #[error("tensor shape mismatch: {0}")]
    ShapeMismatch(String),
    /// 模型或音色未找到 / 加载失败。
    #[error("model not available: {0}")]
    Model(String),
    /// 实时管线背压丢帧（非致命，用于上报）。
    #[error("frame dropped due to backpressure")]
    Dropped,
    /// 输入参数无效（采样率、通道数、帧长等）。
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl VoxError {
    /// 构造一个 [`VoxError::Audio`]，避免调用方手写 `format!`（`mem-avoid-format`）。
    #[inline]
    pub fn audio(msg: impl Into<String>) -> Self {
        Self::Audio(msg.into())
    }

    /// 构造一个 [`VoxError::Infer`]。
    #[inline]
    pub fn infer(msg: impl Into<String>) -> Self {
        Self::Infer(msg.into())
    }

    /// 构造一个 [`VoxError::InvalidInput`]。
    #[inline]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}
