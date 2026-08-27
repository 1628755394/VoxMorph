//! 变声处理节点与音频帧类型。
//!
//! 实时管线中的每个阶段（预处理、特征提取、转换、vocoder）都实现
//! [`VoiceProcessor`]，以统一的 `process` 接口串联（`api-sealed-trait` 思路）。

use crate::VoxError;

/// 一个音频帧：交错样本 + 元数据。
///
/// 帧长动态时用 `Vec<f32>`；若某节点固定帧长，可在内部改用 `Box<[f32]>`
/// 以省一次容量字段（`mem-boxed-slice`）。
#[derive(Debug, Clone)]
pub struct Frame {
    /// 交错样本：`samples[ch + frame * channels]`。
    pub samples: Vec<f32>,
    /// 采样率 (Hz)。
    pub sample_rate: u32,
    /// 声道数。
    pub channels: u16,
    /// 帧时间戳（自管线启动以来的样本数），用于同步与丢帧统计。
    pub timestamp: u64,
}

impl Frame {
    /// 构造一个零填充的帧。
    #[inline]
    pub fn zero(sample_rate: u32, channels: u16, frames: usize) -> Self {
        Self {
            samples: vec![0.0; frames.saturating_mul(channels.max(1) as usize)],
            sample_rate,
            channels,
            timestamp: 0,
        }
    }

    /// 每帧样本数（= `samples.len() / channels`）。
    #[inline]
    pub fn frame_count(&self) -> usize {
        let ch = self.channels.max(1) as usize;
        self.samples.len() / ch
    }

    /// 帧时长（毫秒，浮点）。
    #[inline]
    pub fn duration_ms(&self) -> f64 {
        let frames = self.frame_count() as f64;
        if self.sample_rate == 0 {
            return 0.0;
        }
        frames * 1000.0 / f64::from(self.sample_rate)
    }
}

/// 变声处理节点。
///
/// 实时节点必须是无状态的或可 `reset` 的，避免跨帧状态泄漏导致爆音。
/// `process` 在音频线程内调用，禁止分配堆内存、禁止持锁、禁止 panic。
///
/// # Errors
/// 处理失败返回 [`VoxError`]；调用方在实时路径应降级为静音并上报，
/// 而非向上传播导致线程退出。
pub trait VoiceProcessor: Send {
    /// 处理一帧：从 `input` 生成 `output`。
    ///
    /// `output` 的 `samples` 应已被调用方预分配到合适容量，实现者应原地写入
    /// 而非重新分配（`mem-reuse-collections`）。
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError>;

    /// 重置内部状态（如滤波器历史、相位累加器）。
    ///
    /// 默认空实现，有状态的节点覆写之。
    fn reset(&mut self) {}
}
