//! VoxMorph 音频 I/O：cpal 设备抽象与 symphonia/hound 文件编解码。
//!
//! 实现目标（按里程碑）：
//! - M1: `cpal` 输入/输出实现 [`vox_core::AudioSource`] / [`vox_core::AudioSink`]
//! - M1: `cpal` 设备枚举实现 [`vox_core::DeviceEnumerator`]
//! - M2: `hound` WAV 文件解码器/编码器实现 [`vox_core::AudioSource`] / [`vox_core::AudioSink`]
//! - M3: `symphonia` 多格式解码
//!
//! # 错误
//!
//! 本 crate 定义 [`AudioError`]，用 `#[from]` 链入 [`vox_core::VoxError`]，
//! 同时供 `vox-convert` 通过 `#[from]` 进一步传播。

pub mod cpal;

use thiserror::Error;
use vox_core::VoxError;

/// 音频 I/O 错误。
///
/// `#[non_exhaustive]` 保证未来新增变体不破坏下游匹配。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioError {
    /// 底层 cpal 设备错误。
    #[error("cpal device error: {0}")]
    Device(String),
    /// 设备不可用或已断开。
    #[error("device unavailable: {0}")]
    Unavailable(String),
    /// 文件解码失败。
    #[error("decode error: {0}")]
    Decode(String),
    /// 文件编码失败。
    #[error("encode error: {0}")]
    Encode(String),
    /// 不支持的格式。
    #[error("unsupported format: {0}")]
    Unsupported(String),
    /// 来自核心层的错误（如无效参数）。
    #[error(transparent)]
    Core(#[from] VoxError),
}
