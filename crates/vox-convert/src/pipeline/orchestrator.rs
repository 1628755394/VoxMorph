//! Pipeline 编排器：串联多个 StageWorker，提供 start/stop/feed/receive API。
//!
//! # 用法
//!
//! ```no_run
//! use vox_convert::pipeline::*;
//! use vox_core::{Frame, VoxError};
//!
//! struct Passthrough;
//! impl Stage for Passthrough {
//!     fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
//!         output.samples = input.samples.clone();
//!         output.sample_rate = input.sample_rate;
//!         output.channels = input.channels;
//!         output.timestamp = input.timestamp;
//!         Ok(())
//!     }
//! }
//!
//! let config = PipelineConfig::default_16k_mono();
//! let mut pipeline = Pipeline::new(config);
//! pipeline.add_stage("passthrough", Box::new(Passthrough));
//! let handle = pipeline.start().unwrap();
//!
//! let frame = Frame { samples: vec![0.0; 512], sample_rate: 16000, channels: 1, timestamp: 0 };
//! handle.feed(frame);
//! while let Some(out) = handle.recv() { let _ = out; }
//! handle.stop();
//! ```

use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use tracing::info;

use super::types::{FrameMessage, PipelineConfig, PipelineMetrics};
use super::worker::{create_channel, StageWorker, WorkerHandle};
use vox_core::Frame;

/// 管线构建器：添加阶段，启动后返回运行句柄。
pub struct Pipeline {
    config: PipelineConfig,
    stages: Vec<(String, Box<dyn super::Stage>)>,
    metrics: Arc<PipelineMetrics>,
}

impl Pipeline {
    /// 构造空管线。
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            metrics: PipelineMetrics::shared(),
        }
    }

    /// 添加一个处理阶段（按添加顺序串联）。
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        stage: Box<dyn super::Stage>,
    ) -> &mut Self {
        self.stages.push((name.into(), stage));
        self
    }

    /// 获取共享指标引用。
    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// 启动管线，返回运行句柄。
    ///
    /// 创建 N+1 个 channel（N 个阶段间 + 输入 + 输出），每个阶段一个线程。
    ///
    /// # Errors
    /// 无阶段时返回错误。
    pub fn start(self) -> Result<PipelineHandle, String> {
        if self.stages.is_empty() {
            return Err("pipeline has no stages".to_string());
        }

        let n_stages = self.stages.len();
        let config = self.config;

        // 创建 N+1 个 channel：input → stage0 → stage1 → ... → output。
        let channels: Vec<(Sender<FrameMessage>, Receiver<FrameMessage>)> =
            (0..=n_stages).map(|_| create_channel(&config)).collect();

        let mut handles: Vec<WorkerHandle> = Vec::with_capacity(n_stages);
        let mut stage_names: Vec<String> = Vec::with_capacity(n_stages);

        for (i, (name, stage)) in self.stages.into_iter().enumerate() {
            let (input_tx, input_rx) = channels[i].clone();
            let (output_tx, _output_rx) = channels[i + 1].clone();
            let worker = StageWorker::new(name.clone(), stage, config.clone());
            let handle = worker.spawn(input_rx, input_tx, output_tx, Arc::clone(&self.metrics));
            handles.push(handle);
            stage_names.push(name);
        }

        // 输入 channel 的 sender（供外部喂帧）。
        let feed_tx = channels[0].0.clone();
        // 输出 channel 的 receiver（供外部取帧）。
        let output_rx = channels[n_stages].1.clone();

        info!(
            stages = stage_names.len(),
            stage_names = ?stage_names,
            "pipeline started"
        );

        Ok(PipelineHandle {
            handles,
            feed_tx,
            output_rx,
            metrics: self.metrics,
            config,
        })
    }
}

/// 管线运行句柄：喂帧、取输出、停止。
pub struct PipelineHandle {
    handles: Vec<WorkerHandle>,
    feed_tx: Sender<FrameMessage>,
    output_rx: Receiver<FrameMessage>,
    metrics: Arc<PipelineMetrics>,
    #[allow(dead_code)]
    config: PipelineConfig,
}

impl PipelineHandle {
    /// 喂入一帧到管线入口。
    ///
    /// 如果管线入口 channel 满（背压），返回 `false` 表示应丢帧。
    pub fn feed(&self, frame: Frame) -> bool {
        match self.feed_tx.try_send(Some(frame)) {
            Ok(()) => true,
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.metrics.inc_dropped();
                tracing::warn!("pipeline input full, dropping frame");
                false
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => false,
        }
    }

    /// 发送 EOS 信号，通知所有阶段优雅关闭。
    pub fn send_eos(&self) {
        let _ = self.feed_tx.send(None);
    }

    /// 从管线出口读取一帧（阻塞，超时由调用方控制）。
    ///
    /// 返回 `None` 表示管线已关闭（所有阶段退出）。
    pub fn recv(&self) -> Option<Frame> {
        match self.output_rx.recv() {
            Ok(Some(frame)) => {
                self.metrics.inc_output();
                Some(frame)
            }
            Ok(None) => None,
            Err(_) => None,
        }
    }

    /// 非阻塞尝试读取一帧。
    pub fn try_recv(&self) -> Option<Frame> {
        match self.output_rx.try_recv() {
            Ok(Some(frame)) => {
                self.metrics.inc_output();
                Some(frame)
            }
            Ok(None) => None,
            Err(_) => None,
        }
    }

    /// 获取共享指标。
    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// 停止管线：发送 EOS，等待所有线程退出。
    pub fn stop(mut self) {
        self.send_eos();
        for handle in self.handles.drain(..) {
            if let Some(thread) = handle.thread {
                let _ = thread.join();
            }
        }
        info!("pipeline stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::VoxError;

    /// Passthrough stage。
    struct Passthrough;

    impl super::super::Stage for Passthrough {
        fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
            output.samples = input.samples.clone();
            output.sample_rate = input.sample_rate;
            output.channels = input.channels;
            output.timestamp = input.timestamp;
            Ok(())
        }
    }

    /// 增益 stage：每帧样本乘以 2。
    struct Gain(f32);

    impl super::super::Stage for Gain {
        fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
            output.samples = input.samples.iter().map(|&s| s * self.0).collect();
            output.sample_rate = input.sample_rate;
            output.channels = input.channels;
            output.timestamp = input.timestamp;
            Ok(())
        }
    }

    #[test]
    fn empty_pipeline_start_fails() {
        let config = PipelineConfig::default_16k_mono();
        let pipeline = Pipeline::new(config);
        assert!(pipeline.start().is_err());
    }

    #[test]
    fn single_stage_passthrough() {
        let config = PipelineConfig::default_16k_mono();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage("passthrough", Box::new(Passthrough));
        let handle = pipeline.start().unwrap();

        let frame = Frame {
            samples: vec![0.5; 512],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        assert!(handle.feed(frame));
        handle.send_eos();

        let out = handle.recv();
        assert!(out.is_some());
        let out = out.unwrap();
        assert_eq!(out.samples, vec![0.5; 512]);

        // EOS 后 recv 返回 None。
        assert!(handle.recv().is_none());

        handle.stop();
    }

    #[test]
    fn multi_stage_chain() {
        let config = PipelineConfig::default_16k_mono();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage("gain2x", Box::new(Gain(2.0)));
        pipeline.add_stage("gain3x", Box::new(Gain(3.0)));

        let handle = pipeline.start().unwrap();

        let frame = Frame {
            samples: vec![1.0; 4],
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        };
        handle.feed(frame);
        handle.send_eos();

        let out = handle.recv().expect("should receive frame");
        // 1.0 * 2.0 * 3.0 = 6.0
        assert!(
            out.samples.iter().all(|&s| (s - 6.0).abs() < 1e-5),
            "expected 6.0, got {:?}",
            out.samples
        );

        handle.stop();
    }

    #[test]
    fn metrics_track_frames() {
        // 用大 channel 容量避免背压丢帧（本测试验证计数，非背压）。
        let config = PipelineConfig {
            channel_capacity: 16,
            ..PipelineConfig::default_16k_mono()
        };
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage("passthrough", Box::new(Passthrough));
        let metrics = pipeline.metrics();
        let handle = pipeline.start().unwrap();

        for i in 0..5 {
            let frame = Frame {
                samples: vec![i as f32; 4],
                sample_rate: 16000,
                channels: 1,
                timestamp: i,
            };
            handle.feed(frame);
        }
        handle.send_eos();

        // 接收所有输出。
        let mut count = 0;
        while handle.recv().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);
        assert_eq!(metrics.snapshot().output_frames, 5);

        handle.stop();
    }

    #[test]
    fn stop_joins_all_threads() {
        let config = PipelineConfig::default_16k_mono();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage("a", Box::new(Passthrough));
        pipeline.add_stage("b", Box::new(Passthrough));
        pipeline.add_stage("c", Box::new(Passthrough));

        let handle = pipeline.start().unwrap();
        handle.stop(); // 应在合理时间内返回（所有线程 join 成功）
    }
}
