# VoxMorph

Rust + AI 实时/离线变声器。基于 Tauri + ONNX Runtime (ort) + cpal 音频管线。

## 项目规范

编写或审查任何代码前，必须先调用 `voxmorph` skill（见 `.devin/skills/voxmorph/SKILL.md`）。
该 skill 优先级高于通用 Rust 规范，冲突时以本 skill 为准。

## 构建与验证命令

| 操作 | 命令 |
|------|------|
| 格式化检查 | `cargo fmt --all --check` |
| 格式化 | `cargo fmt --all` |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| 构建 | `cargo build --workspace` |
| 测试 | `cargo test --workspace` |
| Release 构建 | `cargo build --workspace --release` |
| 基准 | `cargo bench --workspace` |

### 提交前自检（必须全过）

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Workspace 结构

```
crates/
├── vox-core/      核心 trait、类型、错误定义
├── vox-io/        cpal 输入输出、symphonia 文件编解码
├── vox-dsp/       重采样、VAD、AGC、PYIN、窗函数
├── vox-feature/   HuBERT/ContentVec 特征提取封装
├── vox-infer/     ort 推理封装、模型加载、EP 选择
├── vox-convert/   变声主流程编排（实时+离线）
├── vox-models/    模型管理、音色库、量化元数据
└── vox-ui/        tauri 前端 + #[tauri::command]
```

- 模块按 feature 划分，不按 type 划分
- 依赖版本统一在根 `Cargo.toml` 的 `[workspace.dependencies]` 声明，各 crate 用 `dep.workspace = true` 继承
- 内部 API 用 `pub(crate)` / `pub(super)`，仅 `vox-ui` 对外暴露 Tauri 命令

## 技术栈（锁定，禁止擅自替换）

GUI: tauri · 音频 I/O: cpal · DSP: dasp/rubato/realfft · 推理: ort ·
编解码: symphonia/hound · 线程通信: crossbeam-channel + ringbuf ·
序列化: serde + toml · 日志: tracing · 错误: thiserror(库) / anyhow(应用)

实时管线禁用 tokio，使用专用 OS 线程 + crossbeam/ringbuf。

## 当前状态

骨架阶段：8 个 crate 已创建，仅 `vox-core` 含核心 trait 定义，其余为最小 `lib.rs`。
重型依赖（cpal/ort/symphonia/tauri）已在 workspace 声明，待对应 crate 实现时引入。
