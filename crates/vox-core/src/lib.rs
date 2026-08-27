//! VoxMorph 核心 trait、类型与错误定义。
//!
//! 本 crate 不含任何 I/O 或推理实现，仅定义跨 crate 共享的契约：
//! - 音频源/汇：[`audio::AudioSource`] / [`audio::AudioSink`]
//! - 音频设备抽象：[`device::AudioDevice`] / [`device::DeviceEnumerator`]
//! - 变声处理节点：[`frame::VoiceProcessor`] / [`frame::Frame`]
//! - 推理后端：[`infer::InferenceSession`] / [`infer::Tensor`]
//! - 音色库：[`timbre::Timbre`] / [`timbre::TimbreId`]
//!
//! 所有 trait 的错误返回 [`error::VoxError`]，下游 crate（如 `vox-io`）
//! 用 `#[from]` 把它链入各自的专用错误类型。

pub mod audio;
pub mod device;
pub mod error;
pub mod frame;
pub mod infer;
pub mod timbre;

pub use audio::{AudioSink, AudioSource};
pub use device::{AudioDevice, DeviceEnumerator};
pub use error::VoxError;
pub use frame::{Frame, VoiceProcessor};
pub use infer::{InferenceSession, Tensor};
pub use timbre::{Timbre, TimbreId};
