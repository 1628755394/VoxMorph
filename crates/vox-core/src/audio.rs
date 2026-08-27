//! 音频源/汇 trait。
//!
//! 实现者：`vox-io` 中的 cpal 输入/输出、文件解码器与编码器。
//! 用 trait 便于测试 mock（`test-mock-traits`）。
//!
//! # 缓冲约定
//!
//! `read` / `write` 操作的是 **交错 (interleaved)** 的 `f32` 样本，
//! 即 `samples[ch + frame * channels]`。`read` 返回实际填入的样本数，
//! 调用方应据此判断是否到达流末尾。

use crate::VoxError;

/// 音频输入源：麦克风、文件解码器、环形缓冲消费者等。
///
/// 实现者必须保证 `read` 在音频线程内不分配堆内存（`mem-reuse-collections`），
/// 不持锁超过 1ms，失败时返回 [`VoxError`] 而非 panic（`err-result-over-panic`）。
pub trait AudioSource: Send {
    /// 采样率 (Hz)。
    fn sample_rate(&self) -> u32;

    /// 声道数。
    fn channels(&self) -> u16;

    /// 向 `out` 填充交错样本，返回实际填入的样本数。
    ///
    /// 返回 `0` 表示流末尾；返回值小于 `out.len()` 表示欠载。
    ///
    /// # Errors
    /// 设备断开、缓冲欠载不可恢复时返回 [`VoxError::Audio`]。
    fn read(&mut self, out: &mut [f32]) -> Result<usize, VoxError>;
}

/// 音频输出汇：扬声器、文件编码器、环形缓冲生产者等。
pub trait AudioSink: Send {
    /// 采样率 (Hz)。
    fn sample_rate(&self) -> u32;

    /// 声道数。
    fn channels(&self) -> u16;

    /// 写入交错样本。
    ///
    /// 实现者应尽量消费全部 `samples`；若内部缓冲满，可返回 [`VoxError::Dropped`]
    /// 以触发背压统计，**不应**阻塞音频线程。
    ///
    /// # Errors
    /// 设备断开或缓冲溢出时返回 [`VoxError`]。
    fn write(&mut self, samples: &[f32]) -> Result<(), VoxError>;
}
