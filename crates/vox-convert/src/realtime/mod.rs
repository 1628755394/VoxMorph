//! 实时引擎：cpal 输入 → Pipeline → cpal 输出 端到端编排。
//!
//! # 架构
//!
//! ```text
//! [CpalSource] → capture_thread → [Pipeline] → output_thread → [CpalSink]
//!    (ringbuf)    (FrameAdapter)   (channels)   (FrameAdapter)   (ringbuf)
//! ```
//!
//! - **capture_thread**: 从 `AudioSource` 读取样本，打包成 `Frame`，喂入 `Pipeline`
//! - **output_thread**: 从 `Pipeline` 读取 `Frame`，写入 `AudioSink`
//! - 两个线程都是专用 OS 线程（`std::thread`），不用 tokio
//! - 错误降级为静音 + `tracing::error!`，不 panic 不退出
//!
//! # 生命周期
//!
//! `RealtimeEngine::start()` → 运行 → `stop()` 优雅关闭（EOS 传播）。

pub mod adapter;
pub mod engine;

pub use adapter::{FrameAdapter, FrameAdapterConfig};
pub use engine::{RealtimeEngine, RealtimeEngineConfig, RealtimeEngineHandle};
