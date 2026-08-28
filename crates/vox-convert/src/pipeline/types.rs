//! 实时管线核心类型：配置、帧消息、指标、阶段 trait。
//!
//! # 线程模型
//!
//! ```text
//! [Capture] -> ringbuf -> [Preprocess] -> channel -> [Feature] -> channel
//!          -> [Convert] -> channel -> [Vocoder] -> ringbuf -> [Output]
//! ```
//!
//! 每阶段一个专用 OS 线程，阶段间用 `crossbeam-channel::bounded`（容量 2~4 帧）。
//! 背压触发丢帧而非无限增长。音频线程内禁止 alloc / panic，错误降级为静音。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use vox_core::Frame;

/// 管线配置。
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// 声道数。
    pub channels: u16,
    /// 帧长（样本数）。20~40ms @ 16kHz → 320~640 样本。
    pub frame_size: usize,
    /// 阶段间 channel 容量（帧数，2~4）。
    pub channel_capacity: usize,
}

impl PipelineConfig {
    /// 构造默认配置：16kHz 单声道，32ms 帧（512 样本），channel 容量 3。
    pub fn default_16k_mono() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            frame_size: 512,
            channel_capacity: 3,
        }
    }

    /// 帧长（毫秒）。
    pub fn frame_ms(&self) -> f32 {
        self.frame_size as f32 * 1000.0 / self.sample_rate as f32
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::default_16k_mono()
    }
}

/// 阶间传递的帧消息。
///
/// 用 `Frame` 的 owned 版本，因跨线程通过 channel 传递需所有权。
/// `None` 表示流末尾（EOS），用于优雅关闭。
pub type FrameMessage = Option<Frame>;

/// 管线运行时指标（原子计数器，线程安全读取）。
///
/// GUI 仪表盘通过 `tauri::emit` 读取这些值，不走日志通道。
#[derive(Debug, Default)]
pub struct PipelineMetrics {
    /// 累计输入帧数。
    pub input_frames: AtomicU64,
    /// 累计输出帧数。
    pub output_frames: AtomicU64,
    /// 累计丢帧数（背压或处理错误）。
    pub dropped_frames: AtomicU64,
    /// 累计处理错误数（已降级为静音）。
    pub error_count: AtomicU64,
    /// 上一帧推理耗时（微秒，0 表示尚未处理过帧）。
    pub last_infer_us: AtomicU64,
}

impl PipelineMetrics {
    /// 构造零值指标。
    pub fn new() -> Self {
        Self::default()
    }

    /// 包装为 `Arc` 供多线程共享。
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// 输入帧 +1。
    #[inline]
    pub fn inc_input(&self) {
        self.input_frames.fetch_add(1, Ordering::Relaxed);
    }

    /// 输出帧 +1。
    #[inline]
    pub fn inc_output(&self) {
        self.output_frames.fetch_add(1, Ordering::Relaxed);
    }

    /// 丢帧 +1。
    #[inline]
    pub fn inc_dropped(&self) {
        self.dropped_frames.fetch_add(1, Ordering::Relaxed);
    }

    /// 错误 +1。
    #[inline]
    pub fn inc_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 记录上一帧推理耗时（微秒）。
    #[inline]
    pub fn set_last_infer_us(&self, us: u64) {
        self.last_infer_us.store(us, Ordering::Relaxed);
    }

    /// 快照当前指标值。
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            input_frames: self.input_frames.load(Ordering::Relaxed),
            output_frames: self.output_frames.load(Ordering::Relaxed),
            dropped_frames: self.dropped_frames.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            last_infer_us: self.last_infer_us.load(Ordering::Relaxed),
        }
    }
}

/// 指标快照（可序列化，供 GUI 展示）。
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub struct MetricsSnapshot {
    pub input_frames: u64,
    pub output_frames: u64,
    pub dropped_frames: u64,
    pub error_count: u64,
    pub last_infer_us: u64,
}

/// 管线阶段 trait：处理一帧，输出一帧。
///
/// 实现者必须是 `Send`（跨线程）且可 `reset()`（避免跨帧泄漏导致爆音）。
/// 处理错误不应 panic，由调用方降级为静音。
pub trait Stage: Send {
    /// 处理一帧。
    ///
    /// `input` 为输入帧，`output` 为输出帧（调用方预分配）。
    /// 实现者应填充 `output.samples`，保持 `sample_rate` / `channels` 一致。
    ///
    /// # Errors
    /// 处理失败返回 [`vox_core::VoxError`]，调用方降级为静音帧。
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), vox_core::VoxError>;

    /// 重置内部状态（避免跨帧泄漏）。
    fn reset(&mut self) {}
}

/// 为 `VoiceProcessor` 自动实现 `Stage`（适配已有 DSP 节点）。
impl<P: vox_core::VoiceProcessor + Send> Stage for P {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), vox_core::VoxError> {
        vox_core::VoiceProcessor::process(self, input, output)
    }

    fn reset(&mut self) {
        vox_core::VoiceProcessor::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_16k_mono_32ms() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.frame_size, 512);
        assert_eq!(cfg.channel_capacity, 3);
        let ms = cfg.frame_ms();
        assert!((ms - 32.0).abs() < 0.1, "frame_ms should be ~32, got {ms}");
    }

    #[test]
    fn metrics_increment() {
        let m = PipelineMetrics::new();
        m.inc_input();
        m.inc_input();
        m.inc_output();
        m.inc_dropped();
        m.inc_error();
        m.inc_error();
        let snap = m.snapshot();
        assert_eq!(snap.input_frames, 2);
        assert_eq!(snap.output_frames, 1);
        assert_eq!(snap.dropped_frames, 1);
        assert_eq!(snap.error_count, 2);
    }

    #[test]
    fn metrics_shared_is_arc() {
        let m = PipelineMetrics::shared();
        m.inc_input();
        assert_eq!(m.snapshot().input_frames, 1);
    }

    #[test]
    fn stage_trait_object_compatible_with_voice_processor() {
        // PitchShifter 实现 VoiceProcessor，应自动实现 Stage。
        use vox_dsp::pitch::PitchShifter;
        let mut shifter: Box<dyn Stage> =
            Box::new(PitchShifter::new(16000, 1, 5.0).expect("pitch shifter creation failed"));
        let input = Frame {
            samples: vec![0.0; 512],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        let mut output = Frame::zero(16000, 1, 0);
        // process 不应 panic
        let _ = shifter.process(&input, &mut output);
    }
}
