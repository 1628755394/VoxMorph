//! 实时管线集成测试：端到端验证多阶段流水线。
//!
//! 模拟真实场景：
//! - Stage 1: Gain（增益）
//! - Stage 2: Limiter（限幅到 [-1, 1]）
//! - Stage 3: Passthrough
//!
//! 验证：帧顺序保持、多阶段串联正确、EOS 优雅关闭、线程全部 join。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use vox_convert::pipeline::{Pipeline, PipelineConfig, Stage};
use vox_core::{Frame, VoxError};

/// 增益 stage。
struct Gain(f32);

impl Stage for Gain {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        output.samples = input.samples.iter().map(|&s| s * self.0).collect();
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;
        Ok(())
    }
}

/// 限幅器 stage：限制样本到 [-1, 1]。
struct Limiter;

impl Stage for Limiter {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        output.samples = input.samples.iter().map(|&s| s.clamp(-1.0, 1.0)).collect();
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;
        Ok(())
    }
}

/// 帧计数 stage：统计处理过的帧数。
struct FrameCounter {
    count: Arc<AtomicUsize>,
}

impl Stage for FrameCounter {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        self.count.fetch_add(1, Ordering::Relaxed);
        output.samples = input.samples.clone();
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;
        Ok(())
    }
}

/// Passthrough stage：原样输出。
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

#[test]
fn end_to_end_gain_limiter_passthrough() {
    let config = PipelineConfig {
        channel_capacity: 8,
        ..PipelineConfig::default_16k_mono()
    };
    let mut pipeline = Pipeline::new(config);
    pipeline.add_stage("gain3x", Box::new(Gain(3.0)));
    pipeline.add_stage("limiter", Box::new(Limiter));
    pipeline.add_stage("passthrough", Box::new(Passthrough));

    let handle = pipeline.start().expect("pipeline start failed");

    // 输入 0.5 → gain 1.5 → limiter 1.0
    // 输入 -0.5 → gain -1.5 → limiter -1.0
    // 输入 0.2 → gain 0.6 → limiter 0.6
    let inputs: Vec<f32> = vec![0.5, -0.5, 0.2, 0.4, -0.4];
    for (i, &val) in inputs.iter().enumerate() {
        let frame = Frame {
            samples: vec![val; 4],
            sample_rate: 16000,
            channels: 1,
            timestamp: i as u64,
        };
        assert!(handle.feed(frame), "frame {i} should be accepted");
    }
    handle.send_eos();

    // 接收并验证。
    let expected: Vec<f32> = vec![1.0, -1.0, 0.6, 1.0, -1.0];
    for (i, expected_val) in expected.iter().enumerate() {
        let out = handle
            .recv()
            .unwrap_or_else(|| panic!("expected output frame {i}"));
        assert!(
            out.samples.iter().all(|&s| (s - expected_val).abs() < 1e-5),
            "frame {i}: expected {expected_val}, got {:?}",
            out.samples
        );
    }

    // EOS 后 recv 返回 None。
    assert!(handle.recv().is_none());

    handle.stop();
}

#[test]
fn frame_counter_tracks_all_frames() {
    let counter = Arc::new(AtomicUsize::new(0));
    let config = PipelineConfig {
        channel_capacity: 32,
        ..PipelineConfig::default_16k_mono()
    };
    let mut pipeline = Pipeline::new(config);
    pipeline.add_stage(
        "counter",
        Box::new(FrameCounter {
            count: Arc::clone(&counter),
        }),
    );

    let handle = pipeline.start().expect("pipeline start failed");

    for i in 0..10 {
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
    assert_eq!(count, 10);
    assert_eq!(
        counter.load(Ordering::Relaxed),
        10,
        "counter stage should see all 10 frames"
    );

    handle.stop();
}

#[test]
fn pipeline_preserves_frame_order() {
    let config = PipelineConfig {
        channel_capacity: 32,
        ..PipelineConfig::default_16k_mono()
    };
    let mut pipeline = Pipeline::new(config);
    pipeline.add_stage("passthrough", Box::new(Passthrough));

    let handle = pipeline.start().expect("pipeline start failed");

    // 用 timestamp 标记帧顺序。
    for i in 0..20u64 {
        let frame = Frame {
            samples: vec![i as f32],
            sample_rate: 16000,
            channels: 1,
            timestamp: i,
        };
        handle.feed(frame);
    }
    handle.send_eos();

    // 验证输出顺序与输入一致。
    for expected_ts in 0..20u64 {
        let out = handle
            .recv()
            .unwrap_or_else(|| panic!("expected frame with ts={expected_ts}"));
        assert_eq!(
            out.timestamp, expected_ts,
            "frame order should be preserved"
        );
    }

    handle.stop();
}

#[test]
fn error_in_one_stage_produces_silence_but_continues() {
    use std::sync::Mutex;

    struct FailOnce {
        failed: Mutex<bool>,
    }

    impl Stage for FailOnce {
        fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
            let mut failed = self.failed.lock().unwrap();
            if !*failed {
                *failed = true;
                return Err(VoxError::invalid_input("first frame fails"));
            }
            output.samples = input.samples.clone();
            output.sample_rate = input.sample_rate;
            output.channels = input.channels;
            output.timestamp = input.timestamp;
            Ok(())
        }
    }

    let config = PipelineConfig {
        channel_capacity: 8,
        ..PipelineConfig::default_16k_mono()
    };
    let mut pipeline = Pipeline::new(config);
    pipeline.add_stage(
        "fail_once",
        Box::new(FailOnce {
            failed: Mutex::new(false),
        }),
    );

    let metrics = pipeline.metrics();
    let handle = pipeline.start().expect("pipeline start failed");

    // 第一帧会失败 → 静音。
    handle.feed(Frame {
        samples: vec![0.9; 4],
        sample_rate: 16000,
        channels: 1,
        timestamp: 0,
    });
    // 第二帧正常。
    handle.feed(Frame {
        samples: vec![0.5; 4],
        sample_rate: 16000,
        channels: 1,
        timestamp: 1,
    });
    handle.send_eos();

    let out0 = handle.recv().expect("first frame (silence)");
    assert!(
        out0.samples.iter().all(|&s| s == 0.0),
        "failed frame should be silence, got {:?}",
        out0.samples
    );

    let out1 = handle.recv().expect("second frame");
    assert!(
        out1.samples.iter().all(|&s| (s - 0.5).abs() < 1e-5),
        "second frame should pass through, got {:?}",
        out1.samples
    );

    assert!(handle.recv().is_none(), "EOS should end stream");
    assert!(
        metrics.snapshot().error_count >= 1,
        "should have recorded at least 1 error"
    );

    handle.stop();
}
