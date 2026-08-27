//! VoxMorph ONNX 推理封装。
//!
//! 本 crate 是 `ort` 的唯一接触点：把 [`ort::Session`] 包装成
//! [`vox_core::InferenceSession`]，避免 `ort` 类型泄露到 `vox-convert`。
//!
//! # ExecutionProvider 优先级
//!
//! CUDA > DirectML > CoreML > CPU。每平台只启用可用 EP，
//! 未启用 EP 用 `tracing::warn!` 记录而非 panic。
//!
//! # Session 复用
//!
//! `ort::Session` 用 `Arc<Session>` 共享，**禁止**每次推理新建 session。
//! 输入/输出 `ort::Value` 用预分配缓冲复用，避免每帧 alloc。
//!
//! # 量化
//!
//! INT8 量化在离线（Python 脚本）完成，Rust 只加载已量化模型，
//! 与 FP32 模型共用同一 [`vox_core::InferenceSession`] trait。
//!
//! # 实现里程碑
//!
//! - M3: `OrtSession` 实现 [`vox_core::InferenceSession`]
//! - M3: EP 自动选择 + `tracing` 上报
//! - M6: 输入/输出缓冲复用、INT8 模型加载

use thiserror::Error;
use vox_core::VoxError;

/// 推理错误。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InferError {
    /// 模型加载失败（文件不存在、格式错误）。
    #[error("model load failure: {0}")]
    Load(String),
    /// ExecutionProvider 不可用或初始化失败。
    #[error("execution provider unavailable: {0}")]
    ExecutionProvider(String),
    /// 张量形状或 dtype 不匹配。
    #[error("tensor shape mismatch: {0}")]
    ShapeMismatch(String),
    /// 后端运行时错误。
    #[error("runtime error: {0}")]
    Runtime(String),
    /// 来自核心层的错误。
    #[error(transparent)]
    Core(#[from] VoxError),
}

// 占位：实现阶段移除。
#[allow(dead_code)]
fn _ensure_link() -> VoxError {
    VoxError::infer("link")
}
