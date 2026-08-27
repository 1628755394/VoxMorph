//! VoxMorph 变声主流程编排：实时管线 + 离线批处理。
//!
//! # 实时管线（CRITICAL）
//!
//! ```text
//! [Capture] -> ringbuf -> [Preprocess] -> channel -> [Feature] -> channel
//!          -> [Convert] -> channel -> [Vocoder] -> ringbuf -> [Output]
//! ```
//!
//! - 每阶段一个 **专用 OS 线程**（`std::thread`），**禁用 tokio**（避免调度抖动）
//! - 阶段间用 `crossbeam-channel::bounded` 或 `ringbuf`，容量 2~4 帧
//! - 背压触发丢帧而非无限增长；丢帧数通过 `tracing` 上报 GUI
//! - 音频线程内禁止 alloc / 持锁 >1ms / panic；错误降级为静音 + `tracing::error!`
//!
//! # 离线管线
//!
//! ```text
//! file decode -> 全量特征提取 -> 批量推理 -> vocoder -> file encode
//! ```
//! 分块处理以控制内存，支持进度回调。
//!
//! # 延迟预算
//!
//! 总单向 <200ms (GPU) / <300ms (CPU INT8)。超预算优先降帧长 → 启用 GPU EP →
//! 模型量化，**不要**先减缓冲（会爆音）。
//!
//! # 实现里程碑
//!
//! - M2: 离线 DSP 变声（pitch/formant）跑通文件处理
//! - M4: 实时管线流水线 + ringbuf + 低延迟输出

pub mod offline;
pub mod pipeline;
pub mod realtime;
pub mod stages;

pub use offline::{OfflineConvertParams, OfflineConverter};
pub use pipeline::{
    FrameMessage, MetricsSnapshot, Pipeline, PipelineConfig, PipelineHandle, PipelineMetrics,
    Stage, StageWorker, WorkerHandle,
};
pub use realtime::{
    FrameAdapter, FrameAdapterConfig, RealtimeEngine, RealtimeEngineConfig, RealtimeEngineHandle,
};
pub use stages::{
    ConvertInputLayout, ConvertStage, FeatureStage, VocoderInputLayout, VocoderStage,
};

use thiserror::Error;
use vox_core::VoxError;

/// 变声编排错误。
///
/// 用 `#[from]` 链接各下游 crate 错误，使 `?` 可在编排层自由传播。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConvertError {
    /// 音频 I/O 错误。
    #[error(transparent)]
    Io(#[from] vox_io::AudioError),
    /// DSP 错误。
    #[error(transparent)]
    Dsp(#[from] vox_dsp::DspError),
    /// 特征提取错误。
    #[error(transparent)]
    Feature(#[from] vox_feature::FeatureError),
    /// 推理错误。
    #[error(transparent)]
    Infer(#[from] vox_infer::InferError),
    /// 音色库错误。
    #[error(transparent)]
    Model(#[from] vox_models::ModelError),
    /// 管线状态错误（如未加载模型即启动）。
    #[error("pipeline state error: {0}")]
    State(String),
    /// 来自核心层的错误。
    #[error(transparent)]
    Core(#[from] VoxError),
}
