//! RVC 端到端测试：使用真实 ONNX 模型验证完整管线。
//!
//! # 前置条件
//!
//! 先运行 `python scripts/generate_test_models.py` 生成测试模型到 `models/test/`。
//!
//! # 运行
//!
//! ```sh
//! cargo test --features test-real-models -p vox-convert --test rvc_e2e -- --nocapture
//! ```
//!
//! # CI
//!
//! 此测试在 `test-real-models` feature 下，CI 默认不运行（无模型文件）。

#![cfg(feature = "test-real-models")]

use std::path::PathBuf;

use vox_convert::{RvcStage, RvcStageConfig, Stage};
use vox_core::Frame;
use vox_infer::OrtSession;

/// 测试模型目录。
fn test_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("test")
}

/// 检查测试模型是否存在，不存在则跳过。
fn models_exist() -> bool {
    let dir = test_models_dir();
    dir.join("contentvec_test.onnx").exists()
        && dir.join("rmvpe_test.onnx").exists()
        && dir.join("rvc_test.onnx").exists()
}

/// 加载三个测试 ONNX 模型并构造 RvcStage。
fn load_test_rvc_stage() -> RvcStage<OrtSession, OrtSession, OrtSession> {
    let dir = test_models_dir();
    let embedder = OrtSession::load(dir.join("contentvec_test.onnx"))
        .expect("failed to load contentvec_test.onnx");
    let f0 = OrtSession::load(dir.join("rmvpe_test.onnx")).expect("failed to load rmvpe_test.onnx");
    let rvc = OrtSession::load(dir.join("rvc_test.onnx")).expect("failed to load rvc_test.onnx");

    RvcStage::new(embedder, 256, f0, rvc, 48000, RvcStageConfig::default())
}

#[test]
fn rvc_stage_processes_real_onnx_models() {
    if !models_exist() {
        eprintln!("跳过: 测试模型不存在，请先运行 python scripts/generate_test_models.py");
        return;
    }

    let mut stage = load_test_rvc_stage();

    // 构造 100ms @ 48kHz 的输入音频。
    let input = Frame {
        samples: vec![0.1; 4800],
        sample_rate: 48000,
        channels: 1,
        timestamp: 0,
    };
    let mut output = Frame::zero(48000, 1, 0);

    // 处理一帧（可能成功或降级静音，但不应 panic）。
    let result = stage.process(&input, &mut output);
    eprintln!("process result: {:?}", result.is_ok());
    eprintln!(
        "output: {} samples @ {}Hz",
        output.samples.len(),
        output.sample_rate
    );

    // 验证输出基本属性。
    assert_eq!(output.sample_rate, 48000);
    assert_eq!(output.channels, 1);
    assert!(!output.samples.is_empty(), "output should not be empty");
}

#[test]
fn rvc_stage_reset_with_real_models() {
    if !models_exist() {
        eprintln!("跳过: 测试模型不存在，请先运行 python scripts/generate_test_models.py");
        return;
    }

    let mut stage = load_test_rvc_stage();

    let input = Frame {
        samples: vec![0.1; 4800],
        sample_rate: 48000,
        channels: 1,
        timestamp: 0,
    };
    let mut output = Frame::zero(48000, 1, 0);
    let _ = stage.process(&input, &mut output);

    // reset 不应 panic。
    stage.reset();

    // reset 后再处理一帧。
    let mut output2 = Frame::zero(48000, 1, 0);
    let _ = stage.process(&input, &mut output2);
    assert_eq!(output2.sample_rate, 48000);
}

#[test]
fn rvc_stage_multiple_chunks() {
    if !models_exist() {
        eprintln!("跳过: 测试模型不存在，请先运行 python scripts/generate_test_models.py");
        return;
    }

    let mut stage = load_test_rvc_stage();

    // 连续处理 5 个 chunk，验证 SOLA 平滑不会崩溃。
    for i in 0..5 {
        let input = Frame {
            samples: vec![0.1; 4800],
            sample_rate: 48000,
            channels: 1,
            timestamp: i as u64 * 4800,
        };
        let mut output = Frame::zero(48000, 1, 0);
        let result = stage.process(&input, &mut output);
        eprintln!(
            "chunk {}: ok={}, {} samples",
            i,
            result.is_ok(),
            output.samples.len()
        );
        assert_eq!(output.sample_rate, 48000);
    }
}

#[test]
fn rvc_stage_silence_input() {
    if !models_exist() {
        eprintln!("跳过: 测试模型不存在，请先运行 python scripts/generate_test_models.py");
        return;
    }

    let mut stage = load_test_rvc_stage();

    // 全零输入（静音）→ 应跳过推理，输出静音。
    let input = Frame {
        samples: vec![0.0; 4800],
        sample_rate: 48000,
        channels: 1,
        timestamp: 0,
    };
    let mut output = Frame::zero(48000, 1, 0);
    let _ = stage.process(&input, &mut output);

    // 静音输入应产生静音输出。
    assert!(
        output.samples.iter().all(|&s| s.abs() < 1e-6),
        "silence input should produce silence output"
    );
}
