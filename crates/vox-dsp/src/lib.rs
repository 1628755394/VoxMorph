//! VoxMorph DSP 模块：重采样、VAD、AGC、PYIN、窗函数。
//!
//! 所有处理节点实现 [`vox_core::VoiceProcessor`]，可串联进实时管线。
//!
//! # 性能约束（CRITICAL）
//!
//! - 内层循环用 `#[inline]` 小函数（`opt-inline-small`）
//! - 禁止 `as` 缩窄转换，用 `try_from`（`num-cast-try-from`）
//! - 浮点比较用容差或 `total_cmp`，不用 `==`（`num-float-compare`）
//! - 整数算术用 `saturating_*` / `checked_*`（`num-overflow-explicit`）
//! - 音频线程内禁止 `Vec::new()` / `format!`，复用预分配缓冲
//!
//! # 实现里程碑
//!
//! - M2: 重采样（`rubato`）、窗函数、AGC
//! - M2: 传统 pitch shift（PSOLA / phase vocoder）作为离线 demo
//! - M3: PYIN 基频提取（供 AI 路线保留原始 F0 曲线）

pub mod pitch;
pub mod resample;

use thiserror::Error;
use vox_core::VoxError;

/// DSP 处理错误。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DspError {
    /// 输入参数无效（采样率、帧长、通道数）。
    #[error("invalid dsp input: {0}")]
    InvalidInput(String),
    /// 算法内部错误（如 FFT 规划失败）。
    #[error("dsp computation failure: {0}")]
    Compute(String),
    /// 来自核心层的错误。
    #[error(transparent)]
    Core(#[from] VoxError),
}
