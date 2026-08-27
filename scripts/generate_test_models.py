#!/usr/bin/env python3
"""生成用于测试的微型 ONNX 模型。

生成三个最小化 ONNX 模型，用于 VoxMorph RvcStage 端到端测试：
  1. contentvec_test.onnx — 输入 [1, T] f32 → 输出 [1, T//160, 256] f32
  2. rmvpe_test.onnx — 输入 [1, T] f32 → 输出 [1, T//160] f32
  3. rvc_test.onnx — 输入 features/pitch/pitchf/sid → 输出 [1, 4800] f32

这些模型不产生有意义的输出，仅用于验证管线编排正确性。

用法:
  python scripts/generate_test_models.py [--output-dir models/test/]

依赖:
  pip install onnx numpy
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto
    import numpy as np
except ImportError:
    print("错误: 需要安装 onnx 和 numpy")
    print("  pip install onnx numpy")
    sys.exit(1)


def make_contentvec_model() -> onnx.ModelProto:
    """ContentVec 测试模型：[1, T] f32 → [1, T//160, 256] f32。

    使用 Slice + ConstantFill：取输入长度，除以 160，填充 256 维零向量。
    为简化，用 Reshape + Constant 实现：输出固定 shape [1, 10, 256] 的零张量。
    """
    # 输入: audio [1, T] f32 (动态 T)
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"])

    # 输出: features [1, 10, 256] f32 (固定 shape 简化测试)
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 10, 256])

    # 用 Constant + Reshape: 创建 [2560] 的零向量，reshape 为 [1, 10, 256]
    zero_const = helper.make_tensor("zero_data", TensorProto.FLOAT, [2560], [0.0] * 2560)
    shape_const = helper.make_tensor("target_shape", TensorProto.INT64, [3], [1, 10, 256])

    nodes = [
        helper.make_node("Constant", [], ["zero_data"], value=zero_const),
        helper.make_node("Constant", [], ["target_shape"], value=shape_const),
        helper.make_node("Reshape", ["zero_data", "target_shape"], ["features"]),
    ]

    graph = helper.make_graph(nodes, "contentvec_test", [audio], [features])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def make_rmvpe_model() -> onnx.ModelProto:
    """RMVPE 测试模型：[1, T] f32 → [1, 10] f32。

    输出固定 200Hz 的 F0 曲线（10 帧）。
    """
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"])
    f0 = helper.make_tensor_value_info("f0", TensorProto.FLOAT, [1, 10])

    # Constant 200Hz × 10 帧
    f0_const = helper.make_tensor("f0_data", TensorProto.FLOAT, [1, 10], [200.0] * 10)

    nodes = [
        helper.make_node("Constant", [], ["f0_data"], value=f0_const),
        # Identity 传递（确保输出名匹配）
        helper.make_node("Identity", ["f0_data"], ["f0"]),
    ]

    graph = helper.make_graph(nodes, "rmvpe_test", [audio], [f0])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def make_rvc_model() -> onnx.ModelProto:
    """RVC 测试模型：features + pitch + pitchf + sid → [1, 4800] f32。

    忽略所有输入，输出固定 [1, 4800] 的 0.5 常量音频。
    """
    # 输入
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, "F", 256])
    pitch = helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "F"])
    pitchf = helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "F"])
    sid = helper.make_tensor_value_info("sid", TensorProto.INT64, [1])

    # 输出: audio [1, 4800] f32
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 4800])

    # Constant 0.5 × 4800 样本
    audio_const = helper.make_tensor("audio_data", TensorProto.FLOAT, [1, 4800], [0.5] * 4800)

    nodes = [
        helper.make_node("Constant", [], ["audio_data"], value=audio_const),
        helper.make_node("Identity", ["audio_data"], ["audio"]),
    ]

    graph = helper.make_graph(
        nodes,
        "rvc_test",
        [features, pitch, pitchf, sid],
        [audio],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 VoxMorph 测试用 ONNX 模型")
    parser.add_argument(
        "--output-dir",
        default="models/test",
        help="模型输出目录（默认: models/test/）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("contentvec_test.onnx", make_contentvec_model),
        ("rmvpe_test.onnx", make_rmvpe_model),
        ("rvc_test.onnx", make_rvc_model),
    ]

    print("=" * 60)
    print("VoxMorph 测试模型生成")
    print("=" * 60)

    for filename, make_fn in models:
        path = output_dir / filename
        print(f"生成 {filename}...")
        model = make_fn()
        onnx.save(model, str(path))
        size = path.stat().st_size
        print(f"  完成: {size} 字节")

    print()
    print(f"测试模型保存在: {output_dir.resolve()}")
    print("运行测试: cargo test --features test-real-models -p vox-convert --test rvc_e2e")
    return 0


if __name__ == "__main__":
    sys.exit(main())
