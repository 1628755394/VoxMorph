#!/usr/bin/env python3
"""VoxMorph 模型下载脚本。

下载 RVC 变声所需的三个 ONNX 模型：
  1. ContentVec (content_vec_500.onnx) — 内容特征提取
  2. RMVPE (rmvpe.onnx) — F0 基频估计
  3. RVC 模型 (.onnx) — 由用户提供，本脚本不下载

用法:
  python scripts/download_models.py [--output-dir models/]

模型来源:
  - ContentVec: https://huggingface.co/therealvinter/ContentVec/resolve/main/content_vec_500.onnx
    (原模型来自 RVC 项目，GPL-3.0 许可)
  - RMVPE: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx
    (原模型来自 RVC 项目，GPL-3.0 许可)

注意: 这些模型是第三方分发，许可为 GPL-3.0，与 VoxMorph 的 MIT 许可无关。
      使用、改变、再分发时请遵循原始许可。
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# 模型下载 URL（第三方分发，GPL-3.0 许可）。
MODEL_URLS = {
    "content_vec_500.onnx": "https://huggingface.co/therealvinter/ContentVec/resolve/main/content_vec_500.onnx",
    "rmvpe.onnx": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
}

# 已知文件大小（字节），用于下载后校验。
EXPECTED_SIZES = {
    "content_vec_500.onnx": 379_000_000,  # ~379MB（近似值，允许 10% 误差）
    "rmvpe.onnx": 50_000_000,  # ~50MB（近似值，允许 10% 误差）
}

CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def download_file(url: str, output_path: Path) -> bool:
    """下载文件到指定路径，显示进度条。

    Returns:
        True 如果下载成功，False 如果失败。
    """
    print(f"  URL: {url}")
    print(f"  目标: {output_path}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VoxMorph/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "=" * (pct // 2) + " " * (50 - pct // 2)
                        sys.stdout.write(f"\r  [{bar}] {pct}% ({downloaded // 1024 // 1024}MB/{total // 1024 // 1024}MB)")
                        sys.stdout.flush()

            print()  # 换行
            return True

    except Exception as e:
        print(f"\n  下载失败: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def verify_size(path: Path, expected: int) -> bool:
    """校验文件大小是否在预期范围内（允许 10% 误差）。"""
    if expected == 0:
        return True
    actual = path.stat().st_size
    lower = int(expected * 0.9)
    upper = int(expected * 1.1)
    if not (lower <= actual <= upper):
        print(f"  警告: 文件大小 {actual} 字节，预期 ~{expected} 字节（允许 10% 误差）")
        return False
    print(f"  大小校验通过: {actual // 1024 // 1024}MB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxMorph RVC 模型下载")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="模型输出目录（默认: models/）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使文件已存在）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxMorph RVC 模型下载")
    print("=" * 60)
    print()

    all_ok = True

    for filename, url in MODEL_URLS.items():
        output_path = output_dir / filename
        print(f"[{filename}]")

        if output_path.exists() and not args.force:
            print(f"  已存在，跳过（使用 --force 强制重新下载）")
            print()
            continue

        success = download_file(url, output_path)
        if not success:
            all_ok = False
            print()
            continue

        # 大小校验。
        expected = EXPECTED_SIZES.get(filename, 0)
        if not verify_size(output_path, expected):
            all_ok = False

        print()

    # RVC 模型提示。
    print("=" * 60)
    print("RVC 变声模型 (.onnx)")
    print("=" * 60)
    print()
    print("RVC 变声模型需要您自行准备：")
    print("  1. 从 RVC 项目或 VCClient 等工具导出 .onnx 格式的 RVC 模型")
    print("  2. 将 .onnx 文件放入 models/ 目录")
    print("  3. 在 VoxMorph GUI 中选择该模型文件")
    print()
    print("注意: .pth 格式的 RVC 模型不能直接使用，需先转换为 .onnx。")
    print()

    if all_ok:
        print("模型下载完成！")
        print(f"模型保存在: {output_dir.resolve()}")
        return 0
    else:
        print("部分模型下载失败，请检查网络连接后重试。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
