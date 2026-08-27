//! 实时变声管线：多线程流水线编排。
//!
//! # 架构
//!
//! ```text
//! [Capture] -> ringbuf -> [Preprocess] -> channel -> [Feature] -> channel
//!          -> [Convert] -> channel -> [Vocoder] -> ringbuf -> [Output]
//! ```
//!
//! 每阶段一个专用 OS 线程（`std::thread`），阶段间用 `crossbeam-channel::bounded`。
//! 背压触发丢帧，错误降级为静音 + `tracing::error!`。

pub mod types;
pub mod worker;

pub use types::{FrameMessage, MetricsSnapshot, PipelineConfig, PipelineMetrics, Stage};
pub use worker::{StageWorker, WorkerHandle};
