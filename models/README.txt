# 模型与音色文件目录
#
# *.onnx  — 推理模型（HuBERT / Converter / Vocoder）
# *.bin   — 音色 embedding
# *.toml  — 音色元数据
#
# 大文件不入库（见 .gitignore）。运行时通过配置注入路径，禁止硬编码绝对路径。
