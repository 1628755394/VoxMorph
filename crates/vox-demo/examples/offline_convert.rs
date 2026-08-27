//! 离线文件变声 demo：生成测试音频 → pitch shift → 输出 WAV。
//!
//! 运行：`cargo run -p vox-demo --example offline_convert`
//!
//! 生成 2 秒 220Hz 正弦波 at 16000 Hz，执行 +7 semitones（升五度）pitch
//! shift，输出到 `target/voxmorph_demo_output.wav`。无需外部输入文件，
//! 零配置运行。

use std::path::PathBuf;

use tracing::info;
use vox_convert::{OfflineConvertParams, OfflineConverter};
use vox_core::{AudioSink, AudioSource};
use vox_io::wav::{WavSink, WavSource};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(
                    "vox_demo=info,vox_convert=info,vox_io=warn,vox_dsp=warn",
                )
            }),
        )
        .init();

    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target");
    let input_path = output_dir.join("voxmorph_demo_input.wav");
    let output_path = output_dir.join("voxmorph_demo_output.wav");

    if let Err(e) = run_demo(&input_path, &output_path) {
        eprintln!("offline convert demo failed: {e}");
        std::process::exit(1);
    }
}

fn run_demo(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. 生成测试音频：2 秒 220Hz 正弦波 at 16000 Hz，单声道。
    let sr = 16000u32;
    let duration_secs = 2u32;
    let n = (sr * duration_secs) as usize;
    let freq = 220.0f32;
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    info!(path = %input_path.display(), samples = samples.len(), "generating test input wav");
    {
        let mut sink = WavSink::create(input_path, sr, 1)?;
        sink.write(&samples)?;
        sink.finalize()?;
    }

    // 2. 执行 pitch shift：+7 semitones（升五度）。
    let params = OfflineConvertParams { semitones: 7.0 };
    info!(semitones = params.semitones, "starting offline conversion");
    OfflineConverter::convert(input_path, output_path, &params)?;
    info!(path = %output_path.display(), "conversion completed");

    // 3. 验证输出：读取并报告基本信息。
    let mut source = WavSource::open(output_path)?;
    let out_sr = source.sample_rate();
    let out_ch = source.channels();
    let mut out_buf = vec![0.0f32; 4096];
    let mut total = 0usize;
    loop {
        let read = source.read(&mut out_buf)?;
        if read == 0 {
            break;
        }
        total += read;
    }

    info!(
        output_samples = total,
        output_sample_rate = out_sr,
        output_channels = out_ch,
        duration_secs = total as f64 / (out_sr as f64 * out_ch as f64),
        "output wav verified"
    );

    // 清理输入文件，保留输出供用户试听。
    let _ = std::fs::remove_file(input_path);

    println!(
        "demo complete: {} ({} samples, {:.1}s, +7 semitones)",
        output_path.display(),
        total,
        total as f64 / (out_sr as f64 * out_ch as f64)
    );

    Ok(())
}
