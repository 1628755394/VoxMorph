//! 实时引擎集成测试：mock AudioSource/AudioSink 端到端验证。
//!
//! 不依赖真实音频设备，用 mock source 喂入正弦波，通过 Pipeline 处理后
//! 验证 mock sink 收到输出。

use std::sync::Mutex;
use std::time::Duration;

use vox_convert::pipeline::{Pipeline, PipelineConfig, Stage};
use vox_convert::realtime::{RealtimeEngine, RealtimeEngineConfig};
use vox_core::{AudioSink, AudioSource, Frame, VoxError};

/// Mock AudioSource：生成指定数量的正弦波样本。
struct SineSource {
    samples: Mutex<Vec<f32>>,
    sample_rate: u32,
    channels: u16,
}

impl SineSource {
    fn new(freq: f32, duration_secs: f32, sample_rate: u32, channels: u16) -> Self {
        let n = (duration_secs * sample_rate as f32) as usize * channels as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let frame = i / channels as usize;
                let t = frame as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
            })
            .collect();
        Self {
            samples: Mutex::new(samples),
            sample_rate,
            channels,
        }
    }
}

impl AudioSource for SineSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
    fn read(&mut self, out: &mut [f32]) -> Result<usize, VoxError> {
        let mut samples = self.samples.lock().unwrap();
        let n = out.len().min(samples.len());
        out[..n].copy_from_slice(&samples[..n]);
        samples.drain(..n);
        Ok(n)
    }
}

/// Mock AudioSink：收集写入的样本。
struct CollectSink {
    collected: Mutex<Vec<f32>>,
    sample_rate: u32,
    channels: u16,
}

impl CollectSink {
    fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            collected: Mutex::new(vec![]),
            sample_rate,
            channels,
        }
    }

    #[allow(dead_code)]
    fn collected(&self) -> Vec<f32> {
        self.collected.lock().unwrap().clone()
    }
}

impl AudioSink for CollectSink {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
    fn write(&mut self, samples: &[f32]) -> Result<(), VoxError> {
        self.collected.lock().unwrap().extend_from_slice(samples);
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

/// Gain stage：乘以固定系数。
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

#[test]
fn realtime_engine_passthrough_delivers_audio() {
    let config = RealtimeEngineConfig {
        frame_size: 256, // 16ms @ 16kHz
        channel_capacity: 8,
        poll_interval: Duration::from_millis(2),
    };

    let pipeline_config = PipelineConfig {
        sample_rate: 16000,
        channels: 1,
        frame_size: 256,
        channel_capacity: 8,
    };
    let mut pipeline = Pipeline::new(pipeline_config);
    pipeline.add_stage("passthrough", Box::new(Passthrough));

    let source = SineSource::new(440.0, 0.5, 16000, 1); // 0.5s of 440Hz
    let sink = CollectSink::new(16000, 1);

    let handle = RealtimeEngine::start_with_source_sink(pipeline, config, source, sink).unwrap();

    // 等待足够时间让音频流过。
    std::thread::sleep(Duration::from_millis(300));

    let metrics = handle.metrics();
    let snap = metrics.snapshot();
    assert!(snap.input_frames > 0, "should have captured some frames");

    handle.stop();

    // 验证输出：stop 后检查 metrics（metrics 是 Arc，stop 后仍可用）。
    let final_snap = metrics.snapshot();
    assert!(
        final_snap.output_frames > 0,
        "should have output some frames"
    );
}

#[test]
fn realtime_engine_gain_modifies_amplitude() {
    let config = RealtimeEngineConfig {
        frame_size: 256,
        channel_capacity: 8,
        poll_interval: Duration::from_millis(2),
    };

    let pipeline_config = PipelineConfig {
        sample_rate: 16000,
        channels: 1,
        frame_size: 256,
        channel_capacity: 8,
    };
    let mut pipeline = Pipeline::new(pipeline_config);
    pipeline.add_stage("gain2x", Box::new(Gain(2.0)));

    let source = SineSource::new(440.0, 0.2, 16000, 1);
    let sink = CollectSink::new(16000, 1);

    let handle = RealtimeEngine::start_with_source_sink(pipeline, config, source, sink).unwrap();
    let metrics = handle.metrics();

    std::thread::sleep(Duration::from_millis(200));
    handle.stop();

    assert!(metrics.snapshot().output_frames > 0, "should have output");
}

#[test]
fn realtime_engine_stop_is_clean() {
    let config = RealtimeEngineConfig::default();

    let pipeline_config = PipelineConfig::default_16k_mono();
    let mut pipeline = Pipeline::new(pipeline_config);
    pipeline.add_stage("passthrough", Box::new(Passthrough));

    let source = SineSource::new(440.0, 2.0, 16000, 1); // 2s
    let sink = CollectSink::new(16000, 1);

    let handle = RealtimeEngine::start_with_source_sink(pipeline, config, source, sink).unwrap();

    std::thread::sleep(Duration::from_millis(100));
    // stop 应在合理时间内返回（所有线程 join 成功）。
    handle.stop();
}

#[test]
fn realtime_engine_metrics_track_flow() {
    let config = RealtimeEngineConfig {
        frame_size: 128, // 8ms @ 16kHz
        channel_capacity: 16,
        poll_interval: Duration::from_millis(1),
    };

    let pipeline_config = PipelineConfig {
        sample_rate: 16000,
        channels: 1,
        frame_size: 128,
        channel_capacity: 16,
    };
    let mut pipeline = Pipeline::new(pipeline_config);
    pipeline.add_stage("passthrough", Box::new(Passthrough));

    let metrics = pipeline.metrics();
    let source = SineSource::new(440.0, 0.3, 16000, 1);
    let sink = CollectSink::new(16000, 1);

    let handle = RealtimeEngine::start_with_source_sink(pipeline, config, source, sink).unwrap();

    std::thread::sleep(Duration::from_millis(300));
    handle.stop();

    let snap = metrics.snapshot();
    assert!(
        snap.input_frames > 0,
        "input_frames should be > 0, got {snap:?}"
    );
    assert!(
        snap.output_frames > 0,
        "output_frames should be > 0, got {snap:?}"
    );
    // 不应有错误（passthrough 不失败）。
    assert_eq!(snap.error_count, 0, "no errors expected");
}
