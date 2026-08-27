//! 麦克风回声测试：默认输入设备 → 默认输出设备。
//!
//! 运行：`cargo run -p vox-demo --example echo`
//!
//! M1 不做重采样：输入/输出采样率不一致时报错退出（重采样留 M2 的 vox-dsp）。
//! 默认运行 10 秒后退出，可用 `--duration <秒>` 覆盖。
//!
//! 两个 cpal `Stream`（输入/输出）都在主线程创建并保活，echo 循环也在主线程
//! 轮询（read 非阻塞，空时 sleep 1ms 避免忙等）。`Stream` 为 `!Send`，不能
//! 跨线程移动，故全部在主线程完成。

use std::time::{Duration, Instant};

use tracing::{info, warn};
use vox_core::{AudioDevice, AudioSink, AudioSource, VoxError};
use vox_io::cpal::{CpalEnumerator, CpalSink, CpalSource};
use vox_io::AudioError;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("vox_demo=info,vox_io=warn")
            }),
        )
        .init();

    match run_echo(Duration::from_secs(10)) {
        Ok(()) => info!("echo test completed"),
        Err(e) => {
            eprintln!("echo test failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_echo(duration: Duration) -> Result<(), AudioError> {
    let en = CpalEnumerator::new();

    let input_dev = en
        .default_input_cpal()
        .ok_or_else(|| AudioError::Unavailable("no default input device".into()))?;
    let output_dev = en
        .default_output_cpal()
        .ok_or_else(|| AudioError::Unavailable("no default output device".into()))?;

    info!(
        input = input_dev.name(),
        input_sr = input_dev.sample_rate(),
        input_ch = input_dev.channels(),
        output = output_dev.name(),
        output_sr = output_dev.sample_rate(),
        output_ch = output_dev.channels(),
        "devices selected"
    );

    if input_dev.sample_rate() != output_dev.sample_rate() {
        return Err(AudioError::Unsupported(format!(
            "sample rate mismatch: input {} Hz vs output {} Hz (resampling not implemented in M1)",
            input_dev.sample_rate(),
            output_dev.sample_rate()
        )));
    }
    if input_dev.channels() != output_dev.channels() {
        warn!(
            input_ch = input_dev.channels(),
            output_ch = output_dev.channels(),
            "channel count mismatch, echo may sound incorrect"
        );
    }

    let (mut source, in_stream) = CpalSource::new(input_dev)?;
    let (mut sink, out_stream) = CpalSink::new(output_dev)?;

    info!(duration_secs = duration.as_secs(), "starting echo loop");

    let start = Instant::now();
    let mut buf = vec![0.0f32; 1024];
    let mut total_in: usize = 0;
    let mut total_out: usize = 0;
    let mut dropped: usize = 0;

    while start.elapsed() < duration {
        let n = source.read(&mut buf).map_err(AudioError::from)?;
        if n == 0 {
            // 欠载：短暂 sleep 避免忙等（非音频线程，可安全 sleep）。
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        total_in += n;
        match sink.write(&buf[..n]) {
            Ok(()) => total_out += n,
            Err(VoxError::Dropped) => dropped += n,
            Err(e) => return Err(AudioError::from(e)),
        }
    }

    info!(
        elapsed_ms = start.elapsed().as_millis(),
        total_in_samples = total_in,
        total_out_samples = total_out,
        dropped_samples = dropped,
        "echo loop finished"
    );

    // 显式 drop 流以停止采集/播放，避免 out_stream/in_stream 顺序不确定。
    drop(out_stream);
    drop(in_stream);
    Ok(())
}
