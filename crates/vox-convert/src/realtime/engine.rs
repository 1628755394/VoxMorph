//! RealtimeEngine：cpal 输入 → Pipeline → cpal 输出 端到端编排。
//!
//! # 线程模型
//!
//! ```text
//! [AudioSource] → capture_thread → [Pipeline] → output_thread → [AudioSink]
//! ```
//!
//! - **capture_thread**: 从 `AudioSource` 读取 → `FrameAdapter::capture` → `Pipeline::feed`
//! - **output_thread**: `Pipeline::recv` → `FrameAdapter::output` → `AudioSink::write`
//! - 两个线程都是专用 OS 线程，不用 tokio
//! - stop 时设置 flag → capture 发送 EOS → pipeline 传播 → output 退出 → join
//!
//! # 错误处理
//!
//! - capture 错误：`tracing::error!` + 继续（不退出线程）
//! - output 错误：`tracing::error!` + 继续（背压 `Dropped` 是正常的）
//! - 管线错误已由 `StageWorker` 降级为静音

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tracing::{error, info, warn};

use vox_core::{AudioSink, AudioSource, VoxError};
use vox_io::cpal::{CpalEnumerator, CpalSink, CpalSource};

use crate::pipeline::{Pipeline, PipelineHandle, PipelineMetrics};
use crate::realtime::adapter::{FrameAdapter, FrameAdapterConfig};

/// 实时引擎配置。
#[derive(Debug, Clone)]
pub struct RealtimeEngineConfig {
    /// 每帧样本数（每声道）。
    pub frame_size: usize,
    /// 管线 channel 容量。
    pub channel_capacity: usize,
    /// capture 线程轮询间隔（当 source 无数据时休眠）。
    pub poll_interval: Duration,
}

impl Default for RealtimeEngineConfig {
    fn default() -> Self {
        Self {
            frame_size: 512, // 32ms @ 16kHz
            channel_capacity: 4,
            poll_interval: Duration::from_millis(5),
        }
    }
}

/// 实时引擎：编排 AudioSource → Pipeline → AudioSink。
///
/// 提供两种启动方式：
/// - [`RealtimeEngine::start_with_default_devices`]: 使用 cpal 默认设备
/// - [`RealtimeEngine::start_with_source_sink`]: 使用任意 AudioSource/AudioSink（供测试）
pub struct RealtimeEngine;

impl RealtimeEngine {
    /// 使用默认 cpal 设备 + 预构建 Pipeline 启动实时变声。
    ///
    /// # Errors
    /// 设备打开失败返回 [`VoxError`]。
    pub fn start_with_default_devices(
        pipeline: Pipeline,
        config: RealtimeEngineConfig,
    ) -> Result<RealtimeEngineHandle, VoxError> {
        let enumerator = CpalEnumerator::new();
        let input_device = enumerator
            .default_input_cpal()
            .ok_or_else(|| VoxError::audio("no default input device"))?;
        let output_device = enumerator
            .default_output_cpal()
            .ok_or_else(|| VoxError::audio("no default output device"))?;

        let (source, input_stream) =
            CpalSource::new(input_device).map_err(|e| VoxError::audio(e.to_string()))?;
        let (sink, output_stream) =
            CpalSink::new(output_device).map_err(|e| VoxError::audio(e.to_string()))?;

        let handle = Self::start_with_source_sink(pipeline, config, source, sink)?;
        Ok(RealtimeEngineHandle {
            inner: handle,
            _input_stream: Some(input_stream),
            _output_stream: Some(output_stream),
        })
    }

    /// 使用任意 AudioSource/AudioSink 启动实时变声（供测试）。
    ///
    /// # Errors
    /// Pipeline 启动失败或线程 spawn 失败返回 [`VoxError`]。
    pub fn start_with_source_sink(
        pipeline: Pipeline,
        config: RealtimeEngineConfig,
        source: impl AudioSource + 'static,
        sink: impl AudioSink + 'static,
    ) -> Result<RealtimeEngineHandleInner, VoxError> {
        let metrics = pipeline.metrics();
        let mut pipeline_handle = pipeline
            .start()
            .map_err(|e| VoxError::audio(format!("pipeline start failed: {e}")))?;

        // 拆分 feed_tx 和 output_rx 给各自的线程。
        let feed_tx = std::mem::replace(
            &mut pipeline_handle.feed_tx,
            crossbeam_channel::bounded(1).0,
        );
        let output_rx = std::mem::replace(
            &mut pipeline_handle.output_rx,
            crossbeam_channel::bounded(1).1,
        );

        let stop_flag = Arc::new(AtomicBool::new(false));

        // capture 线程。
        let capture_stop = Arc::clone(&stop_flag);
        let capture_metrics = Arc::clone(&metrics);
        let capture_config = config.clone();
        let capture_thread = thread::Builder::new()
            .name("voxmorph-capture".to_string())
            .spawn(move || {
                run_capture_loop(
                    source,
                    feed_tx,
                    capture_config,
                    capture_stop,
                    capture_metrics,
                );
            })
            .map_err(|e| VoxError::audio(format!("capture thread spawn failed: {e}")))?;

        // output 线程。
        let output_stop = Arc::clone(&stop_flag);
        let output_metrics = Arc::clone(&metrics);
        let output_config = config.clone();
        let output_thread = thread::Builder::new()
            .name("voxmorph-output".to_string())
            .spawn(move || {
                run_output_loop(sink, output_rx, output_config, output_stop, output_metrics);
            })
            .map_err(|e| VoxError::audio(format!("output thread spawn failed: {e}")))?;

        info!("realtime engine started");

        Ok(RealtimeEngineHandleInner {
            pipeline_handle,
            stop_flag,
            capture_thread: Some(capture_thread),
            output_thread: Some(output_thread),
        })
    }
}

/// capture 线程主循环。
fn run_capture_loop(
    mut source: impl AudioSource,
    feed_tx: crossbeam_channel::Sender<crate::pipeline::FrameMessage>,
    config: RealtimeEngineConfig,
    stop_flag: Arc<AtomicBool>,
    metrics: Arc<PipelineMetrics>,
) {
    let adapter_config = FrameAdapterConfig {
        frame_size: config.frame_size,
        channels: source.channels(),
        sample_rate: source.sample_rate(),
    };
    let mut adapter = FrameAdapter::new(adapter_config);

    while !stop_flag.load(Ordering::Relaxed) {
        match adapter.capture(&mut source) {
            Ok(frames) => {
                for frame in frames {
                    metrics.inc_input();
                    if feed_tx.try_send(Some(frame)).is_err() {
                        metrics.inc_dropped();
                        warn!("pipeline input full, dropping frame in capture");
                    }
                }
                if adapter.accumulated_samples() == 0 {
                    thread::sleep(config.poll_interval);
                }
            }
            Err(e) => {
                error!(error = %e, "capture read failed");
                thread::sleep(config.poll_interval);
            }
        }
    }

    let _ = feed_tx.send(None);
    info!("capture thread exiting");
}

/// output 线程主循环。
fn run_output_loop(
    mut sink: impl AudioSink,
    output_rx: crossbeam_channel::Receiver<crate::pipeline::FrameMessage>,
    config: RealtimeEngineConfig,
    stop_flag: Arc<AtomicBool>,
    metrics: Arc<PipelineMetrics>,
) {
    let adapter_config = FrameAdapterConfig {
        frame_size: config.frame_size,
        channels: sink.channels(),
        sample_rate: sink.sample_rate(),
    };
    let mut adapter = FrameAdapter::new(adapter_config);

    while !stop_flag.load(Ordering::Relaxed) {
        match output_rx.recv_timeout(config.poll_interval * 4) {
            Ok(Some(frame)) => {
                metrics.inc_output();
                if let Err(e) = adapter.output(&frame, &mut sink) {
                    if !matches!(e, VoxError::Dropped) {
                        error!(error = %e, "output write failed");
                    } else {
                        warn!("output sink full, frame dropped");
                    }
                }
            }
            Ok(None) => {
                info!("output thread received eos, exiting");
                break;
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // 超时无数据，继续等待。
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                info!("output channel disconnected, exiting");
                break;
            }
        }
    }

    info!("output thread exiting");
}

/// 内部引擎句柄（不含 cpal 流句柄）。
pub struct RealtimeEngineHandleInner {
    pipeline_handle: PipelineHandle,
    stop_flag: Arc<AtomicBool>,
    capture_thread: Option<thread::JoinHandle<()>>,
    output_thread: Option<thread::JoinHandle<()>>,
}

impl RealtimeEngineHandleInner {
    /// 获取共享指标。
    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        self.pipeline_handle.metrics()
    }

    /// 停止引擎：设置 stop flag → 等待线程退出 → 停止 pipeline。
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(t) = self.capture_thread.take() {
            let _ = t.join();
        }
        if let Some(t) = self.output_thread.take() {
            let _ = t.join();
        }
        self.pipeline_handle.stop();
        info!("realtime engine stopped");
    }
}

/// 实时引擎运行句柄（含 cpal 流句柄保活）。
pub struct RealtimeEngineHandle {
    inner: RealtimeEngineHandleInner,
    /// 保持 cpal 流句柄存活（drop 即停止采集/播放）。
    _input_stream: Option<cpal::Stream>,
    _output_stream: Option<cpal::Stream>,
}

impl RealtimeEngineHandle {
    /// 获取共享指标。
    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        self.inner.metrics()
    }

    /// 停止引擎。
    pub fn stop(self) {
        self.inner.stop();
        // cpal streams drop here, stopping capture/playback.
    }
}
