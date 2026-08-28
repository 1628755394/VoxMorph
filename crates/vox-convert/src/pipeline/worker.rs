//! StageWorker：管线阶段线程封装。
//!
//! # 职责
//!
//! - 从输入 channel 读取 `FrameMessage`
//! - 调用 `Stage::process` 处理帧
//! - 写入输出 channel（背压触发丢帧 + `tracing::warn!`）
//! - 处理错误降级为静音帧 + `tracing::error!` + `metrics.inc_error()`
//! - 收到 `None`（EOS）时优雅退出
//!
//! # 线程安全
//!
//! Worker 在专用 OS 线程内运行，`Stage` 实现 `Send`。
//! 不持锁、不 alloc（除启动期）、不 panic。

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{bounded, Receiver, Sender};
use tracing::{error, warn};

use vox_core::Frame;

use super::{FrameMessage, PipelineMetrics, Stage};

/// 阶段工作线程句柄。
pub struct WorkerHandle {
    /// 用于向该 worker 发送帧的 sender。
    pub input: Sender<FrameMessage>,
    /// worker 线程 join handle。
    pub thread: Option<thread::JoinHandle<()>>,
}

/// 阶段工作线程构造器。
pub struct StageWorker {
    name: String,
    stage: Box<dyn Stage>,
    config: super::PipelineConfig,
}

impl StageWorker {
    /// 构造一个阶段 worker。
    pub fn new(
        name: impl Into<String>,
        stage: Box<dyn Stage>,
        config: super::PipelineConfig,
    ) -> Self {
        Self {
            name: name.into(),
            stage,
            config,
        }
    }

    /// 启动 worker 线程，返回输入 sender（供上游喂帧）和 join handle。
    ///
    /// `input_rx` 为该 worker 的输入 channel receiver。
    /// `output_tx` 为下游 channel 的 sender。
    /// `metrics` 为共享指标。
    /// `input_tx` 为输入 channel 的 sender 克隆（返回给调用方供喂帧）。
    pub fn spawn(
        self,
        input_rx: Receiver<FrameMessage>,
        input_tx: Sender<FrameMessage>,
        output_tx: Sender<FrameMessage>,
        metrics: Arc<PipelineMetrics>,
    ) -> WorkerHandle {
        let name = self.name;
        let thread = thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                run_stage_loop(self.stage, input_rx, output_tx, metrics, self.config, &name);
            })
            .expect("failed to spawn stage worker thread");

        WorkerHandle {
            input: input_tx,
            thread: Some(thread),
        }
    }
}

/// 阶段处理主循环。
///
/// 从 `input_rx` 读取帧，处理后写入 `output_tx`。
/// 背压（channel 满）时丢帧 + `tracing::warn!`。
/// 处理错误时降级为静音帧 + `tracing::error!`。
fn run_stage_loop(
    mut stage: Box<dyn Stage>,
    input_rx: Receiver<FrameMessage>,
    output_tx: Sender<FrameMessage>,
    metrics: Arc<PipelineMetrics>,
    config: super::PipelineConfig,
    name: &str,
) {
    let frame_size = config.frame_size;
    let sr = config.sample_rate;
    let ch = config.channels;

    for msg in &input_rx {
        let input_frame = match msg {
            Some(frame) => frame,
            None => {
                // EOS：传递给下游并退出。
                let _ = output_tx.send(None);
                tracing::info!(stage = name, "stage received eos, exiting");
                break;
            }
        };

        metrics.inc_input();

        // 预分配输出帧（复用避免 alloc — 但 Vec 需要在循环外预分配）。
        // TODO: M6 优化为预分配 buffer 复用（mem-reuse-collections）。
        let frame_len = frame_size * ch as usize;
        let mut output_frame = Frame::zero(sr, ch, frame_len);
        output_frame.timestamp = input_frame.timestamp;

        // 处理帧，错误降级为静音。
        let started = Instant::now();
        match stage.process(&input_frame, &mut output_frame) {
            Ok(()) => {
                metrics.set_last_infer_us(started.elapsed().as_micros() as u64);
            }
            Err(e) => {
                metrics.set_last_infer_us(started.elapsed().as_micros() as u64);
                metrics.inc_error();
                error!(stage = name, error = %e, "stage processing failed, outputting silence");
                // output_frame 已是 zero（静音），继续发送。
            }
        }

        // 发送到下游，背压时丢帧。
        match output_tx.try_send(Some(output_frame)) {
            Ok(()) => {}
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                metrics.inc_dropped();
                warn!(stage = name, "output channel full, dropping frame");
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                tracing::info!(stage = name, "output channel disconnected, exiting");
                break;
            }
        }
    }

    // 确保帧大小用于日志（避免 unused 警告）。
    let _ = frame_size;
}

/// 创建一对 bounded channel，容量来自 config。
pub fn create_channel(
    config: &super::PipelineConfig,
) -> (Sender<FrameMessage>, Receiver<FrameMessage>) {
    bounded(config.channel_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::types::PipelineConfig;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use vox_core::VoxError;

    /// Passthrough stage：原样输出输入帧。
    struct Passthrough;

    impl Stage for Passthrough {
        fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
            output.samples = input.samples.clone();
            output.sample_rate = input.sample_rate;
            output.channels = input.channels;
            output.timestamp = input.timestamp;
            Ok(())
        }
    }

    /// Always-error stage：总是返回错误，测试降级为静音。
    struct AlwaysError;

    impl Stage for AlwaysError {
        fn process(&mut self, _input: &Frame, _output: &mut Frame) -> Result<(), VoxError> {
            Err(VoxError::invalid_input("test error"))
        }
    }

    #[test]
    fn passthrough_worker_processes_frames() {
        let config = PipelineConfig::default_16k_mono();
        let metrics = PipelineMetrics::shared();
        let (in_tx, in_rx) = create_channel(&config);
        let (out_tx, out_rx) = create_channel(&config);

        let worker = StageWorker::new("test-passthrough", Box::new(Passthrough), config.clone());
        let in_tx_clone = in_tx.clone();
        let _handle = worker.spawn(in_rx, in_tx_clone, out_tx, metrics.clone());

        // 发送 3 帧 + EOS。
        for i in 0..3 {
            let frame = Frame {
                samples: vec![i as f32; 512],
                sample_rate: 16000,
                channels: 1,
                timestamp: i,
            };
            in_tx.send(Some(frame)).unwrap();
        }
        in_tx.send(None).unwrap();

        // 等待 worker 处理完。
        thread::sleep(Duration::from_millis(100));

        // 读取输出。
        let mut count = 0;
        while let Ok(msg) = out_rx.recv_timeout(Duration::from_millis(200)) {
            if msg.is_none() {
                break;
            }
            count += 1;
        }
        assert_eq!(count, 3, "should receive 3 processed frames");
        assert_eq!(metrics.output_frames.load(Ordering::Relaxed), 0); // output_frames 由最终消费者计数
    }

    #[test]
    fn error_stage_produces_silence() {
        let config = PipelineConfig::default_16k_mono();
        let metrics = PipelineMetrics::shared();
        let (in_tx, in_rx) = create_channel(&config);
        let (out_tx, out_rx) = create_channel(&config);

        let worker = StageWorker::new("test-error", Box::new(AlwaysError), config);
        let in_tx_clone = in_tx.clone();
        let _handle = worker.spawn(in_rx, in_tx_clone, out_tx, metrics.clone());

        let frame = Frame {
            samples: vec![0.5; 512],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        in_tx.send(Some(frame)).unwrap();
        in_tx.send(None).unwrap();

        thread::sleep(Duration::from_millis(100));

        let msg = out_rx
            .recv_timeout(Duration::from_millis(200))
            .unwrap()
            .unwrap();
        assert!(
            msg.samples.iter().all(|&s| s == 0.0),
            "error stage should output silence"
        );
        assert!(metrics.error_count.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn eos_propagates_through_worker() {
        let config = PipelineConfig::default_16k_mono();
        let metrics = PipelineMetrics::shared();
        let (in_tx, in_rx) = create_channel(&config);
        let (out_tx, out_rx) = create_channel(&config);

        let worker = StageWorker::new("test-eos", Box::new(Passthrough), config);
        let in_tx_clone = in_tx.clone();
        let _handle = worker.spawn(in_rx, in_tx_clone, out_tx, metrics);

        in_tx.send(None).unwrap();

        let msg = out_rx.recv_timeout(Duration::from_millis(200)).unwrap();
        assert!(msg.is_none(), "EOS should propagate");
    }
}
