---
name: voxmorph
description: >
  VoxMorph 项目专属开发规范。Rust + AI 实时/离线变声器，基于 Tauri + ONNX Runtime
  (ort) + cpal 音频管线。涵盖 crate 划分、音频线程模型、ONNX 推理封装、实时延迟
  控制、跨平台音频设备抽象、音色库管理等约束。在编写、审查或重构 VoxMorph 任何
  代码前必须调用本 skill。
license: MIT
metadata:
  project: VoxMorph
  stack: rust, tauri, ort, cpal, dasp, symphonia
  edition: "2021"
  msrv: "1.75"
---

# VoxMorph 开发规范

本 skill 是 VoxMorph（Rust + AI 变声器）的项目级约束，优先级高于通用 Rust 规范。
当本 skill 与通用 rust-skills 冲突时，以本 skill 为准。

## 何时应用

在以下任一情形必须先调用本 skill：

- 新增/修改 `crates/` 下任何 crate 的代码
- 涉及音频 I/O、DSP、ONNX 推理、实时管线、GUI 命令
- 新增依赖、调整 Cargo.toml、修改 workspace 结构
- 处理延迟、缓冲、线程模型相关问题
- 审查 PR、提交 commit 前的最终自检

## 技术栈锁定

| 用途 | 选型 | 禁止替换 |
|------|------|---------|
| GUI | `tauri` + svelte/vue | 不用 egui/iced/slint |
| 音频 I/O | `cpal` | 不用 portaudio/rustaudio |
| DSP 基础 | `dasp`、`rubato`、`realfft` | |
| 推理后端 | `ort` (ONNX Runtime) | 不用 tract/tract-core；`candle` 仅作备选 EP |
| 文件编解码 | `symphonia`、`hound` | 不用 ffmpeg-sys |
| 线程通信 | `crossbeam-channel` + `ringbuf` | 实时管线不用 tokio mpsc |
| 序列化 | `serde` + `toml` | |
| 日志 | `tracing` + `tracing-subscriber` | 不用 log/env_logger |
| 错误 | 库 crate 用 `thiserror`，应用层用 `anyhow` | |

新增依赖前必须确认：未在表中、且无等价已有依赖时，提请用户确认。

## Workspace 结构（必须遵守）

```
voxmorph/
├── crates/
│   ├── vox-core/        # 核心 trait、类型、错误定义
│   ├── vox-io/          # cpal 输入输出、symphonia 文件编解码
│   ├── vox-dsp/         # 重采样、VAD、AGC、PYIN、窗函数
│   ├── vox-feature/     # HuBERT/ContentVec 特征提取封装
│   ├── vox-infer/       # ort 推理封装、模型加载、EP 选择
│   ├── vox-convert/     # 变声主流程编排（实时+离线）
│   ├── vox-models/      # 模型管理、音色库、量化元数据
│   └── vox-ui/          # tauri 前端 + #[tauri::command]
├── models/              # *.onnx 模型与音色 embedding
├── assets/              # 图标、默认配置
├── .devin/skills/voxmorph/SKILL.md
├── Cargo.toml           # [workspace]
└── AGENTS.md
```

- 模块按 **feature** 划分，不按 type 划分（`proj-mod-by-feature`）
- workspace 依赖统一在根 `Cargo.toml` 用 `[workspace.dependencies]` 声明版本
- 每个 crate 的 `Cargo.toml` 用 `dep.workspace = true` 继承（`proj-workspace-deps`）
- `main.rs` / `lib.rs` 保持极简，逻辑下沉到各 crate（`proj-lib-main-split`）
- 内部 API 用 `pub(crate)` 或 `pub(super)`，仅 `vox-ui` 对外暴露 Tauri 命令（`proj-pub-crate-internal`）

## 核心 Trait 约定

### 音频源/汇

```rust
pub trait AudioSource {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
    fn read(&mut self, out: &mut [f32]) -> Result<usize, AudioError>;
}

pub trait AudioSink {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
    fn write(&mut self, samples: &[f32]) -> Result<(), AudioError>;
}
```

- 实现者：`cpal` 输入/输出、文件解码器、文件编码器
- 用 trait 便于测试 mock（`test-mock-traits`）
- `read`/`write` 用 `&[f32]` / `&mut [f32]`，**不要** `&Vec<f32>`（`own-slice-over-vec`）

### 变声处理节点

```rust
pub trait VoiceProcessor {
    fn process(&mut self, frame: &Frame, out: &mut Frame) -> Result<(), ConvertError>;
}

pub struct Frame {
    pub samples: Vec<f32>,        // interleaved
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp: u64,
}
```

- 实时节点必须是无状态的或可 `reset()` 的，避免跨帧泄漏导致爆音
- `Frame` 用 `Vec<f32>` 因为帧长动态；如固定帧长改用 `Box<[f32]>`（`mem-boxed-slice`）

### 推理后端

```rust
pub trait InferenceSession {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferError>;
}
```

- `ort::Session` 实现此 trait，便于测试用假后端
- 不要把 `ort::Session` 直接暴露到 `vox-convert`，必须经 `vox-infer` 封装

## 实时管线约束（CRITICAL）

### 线程模型

```
[Capture] -> ringbuf -> [Preprocess] -> channel -> [Feature] -> channel
         -> [Convert] -> channel -> [Vocoder] -> ringbuf -> [Output]
```

- 每个阶段一个 **专用 OS 线程**（`std::thread`），**不用 tokio**（避免调度抖动）
- 阶段间用 `crossbeam-channel::bounded` 或 `ringbuf::HeapRingBuf`
- **有界 channel** 容量按 2~4 帧设计，背压触发丢帧而非无限增长（`async-bounded-channel` 思想）
- GUI 与引擎之间用 `tauri::Manager::emit` 事件，不直接共享内存

### 延迟预算

| 阶段 | 预算 |
|------|------|
| 帧长 | 20~40 ms |
| 重叠 | 50% |
| 特征提取 | <20 ms |
| 转换推理 | <80 ms (GPU) / <150 ms (CPU INT8) |
| Vocoder | <20 ms |
| 总单向 | <200 ms (GPU) / <300 ms (CPU) |

- 超预算时优先：降帧长 → 启用 GPU EP → 模型量化，**不要**先减缓冲（会爆音）
- 缓冲水位、推理耗时、丢帧数必须通过 `tracing` 上报 GUI 仪表盘

### 禁止项

- **禁止** 在音频线程内 `alloc`（除启动期）：复用预分配 buffer（`mem-reuse-collections`）
- **禁止** 在音频线程内持锁超过 1ms：用 `ringbuf` 无锁队列
- **禁止** 在音频线程内 `unwrap()`/`panic()`：错误降级为静音并上报（`err-result-over-panic`）
- **禁止** 跨 `.await` 持锁（音频线程本就不该有 async）

## ONNX 推理封装

### 模型加载

- 模型文件放 `models/`，路径通过配置注入，**不硬编码绝对路径**
- 加载时显式指定 `ExecutionProvider` 优先级：CUDA > DirectML > CoreML > CPU
- 每个平台只启用可用 EP，未启用 EP 用 `tracing::warn!` 记录而非 panic
- `ort::Session` 用 `Arc<Session>` 共享，**不要**每次推理新建 session

### 推理调用

- 输入/输出 `ort::Value` 用 `Cow<[_]>` 或预分配 buffer 复用，避免每帧 alloc
- 张量维度错误用 `thiserror` 定义专用 `InferError::ShapeMismatch`，**不要** `unwrap`
- 推理耗时用 `tracing::span!(Level::TRACE, "infer")` 包裹，便于 profiling

### 量化

- INT8 量化在 **离线** 完成（Python 侧脚本），Rust 只加载已量化模型
- 量化模型与 FP32 模型用同一 `InferenceSession` trait，调用方无感知

## 跨平台音频设备

```rust
pub trait AudioDevice: Send {
    fn name(&self) -> &str;
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
}

pub trait DeviceEnumerator {
    fn list_inputs(&self) -> Vec<Box<dyn AudioDevice>>;
    fn list_outputs(&self) -> Vec<Box<dyn AudioDevice>>;
    fn default_input(&self) -> Option<Box<dyn AudioDevice>>;
    fn default_output(&self) -> Option<Box<dyn AudioDevice>>;
}
```

- 平台实现：Windows(WASAPI) / macOS(CoreAudio) / Linux(ALSA+Pulse)
- 设备热插拔：监听 `cpal` 的 `device_change` 回调，通过 `tracing` 通知 GUI
- 采样率不匹配时用 `rubato` 重采样，**不要**依赖 cpal 自动转换（行为不一致）

## 音色库设计

```rust
pub struct Timbre {
    pub id: TimbreId,           // newtype(u64)
    pub name: String,
    pub embedding: Vec<f32>,    // 或 Box<[f32]>
    pub f0_offset_semitones: f32,
    pub tags: Vec<String>,
}
```

- `TimbreId` 用 newtype 防止与模型 ID 混淆（`type-newtype-ids`）
- embedding 用 `Box<[f32]>` 而非 `Vec<f32>`（加载后不变，`mem-boxed-slice`）
- 音色文件格式：`.toml` 元数据 + `.bin` embedding，**不要**塞进单个大 JSON
- 用户自训练流程由 Python 脚本完成，Rust 只负责加载已生成音色

## 错误处理

### 错误类型分层

```
vox-core:    VoxError (thiserror, #[non_exhaustive])
vox-io:      AudioError (thiserror)
vox-infer:   InferError (thiserror)
vox-convert: ConvertError (thiserror, #[from] AudioError/InferError)
vox-ui:      anyhow::Result (tauri command 边界)
```

- 每个 crate 自定义错误，用 `#[from]` 链接（`err-from-impl`、`err-source-chain`）
- 错误消息小写、无句号（`err-lowercase-msg`）
- Tauri 命令返回 `Result<T, String>`，在边界把 `anyhow::Error` 用 `?` + `to_string()` 转换
- **音频线程内** 错误降级为静音 + `tracing::error!`，不向上传播（避免线程退出）

## 日志与可观测性

- 用 `tracing`，**不用** `println!`/`log`（`obs-tracing-over-log`）
- 结构化字段：`frame_len_ms`, `buffer_level`, `infer_ms`, `dropped_frames`（`obs-structured-fields`）
- 实时管线 span：`#[tracing::instrument(skip(self, frame))]`，**skip 大 buffer** 避免日志爆炸
- 日志级别：实时音频线程默认 `WARN`，调试时临时升 `TRACE`，**不**写文件（IO 抖动）
- GUI 仪表盘数据走 `tauri::emit`，不走日志通道
- **禁止** 日志输出原始音频样本（可能含语音 PII，`obs-no-sensitive-data`）

## 性能配置

### Cargo.toml（workspace 根）

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"        # 实时线程不该 unwind
strip = true

[profile.dev.package."*"]
opt-level = 3          # 依赖在 dev 也优化，否则音频跑不动
```

- `panic = "abort"`：音频线程 panic 不应 unwind，整体退出比静默错误更易诊断
- 但 GUI 主线程用 `catch_unwind` 包裹 Tauri 命令，避免单次命令 panic 杀进程

### 热路径规则

- DSP 内层循环：`#[inline]` 小函数，避免 `as` 转换用 `try_from`（`num-cast-try-from`）
- 推理前后 buffer 用 `mem::take`/`mem::replace` 复用（`mem-take-replace`）
- 浮点比较用 `total_cmp` 或容差，**不用** `==`（`num-float-compare`）
- 整数算术用 `saturating_*`/`checked_*`，避免溢出 panic（`num-overflow-explicit`）

## 测试约定

- 单元测试在 `#[cfg(test)] mod tests` 内，`use super::*`（`test-cfg-test-module`）
- DSP 算法用 `proptest` 做属性测试（如重采样后长度 = 输入长度 * 比率）
- 推理封装用假 `InferenceSession` mock，**不**在 CI 跑真实 ONNX 模型（`test-mock-traits`）
- 实时管线并发用 `loom` 测试 ringbuf/channel 死锁（`test-loom-concurrency`）
- 性能基准用 `criterion`，关键路径：特征提取、推理、vocoder（`test-criterion-bench`）
- 音频质量用 `insta` 快照测试频谱图（`test-snapshot-testing`）

## GUI / Tauri 约束

- 所有 `#[tauri::command]` 返回 `Result<T, String>`，**不**返回 `anyhow::Result`（序列化问题）
- 重型操作（文件处理、模型加载）在 `tokio::task::spawn_blocking` 内执行（`async-spawn-blocking`）
- 实时音频状态查询用 `tauri::Manager::emit` 推送，**不**用前端轮询
- 前端状态机：idle / loading-model / ready / running / error，用 enum 不用 string（`type-enum-states`）

## 禁止清单（Anti-patterns）

| 禁止 | 原因 |
|------|------|
| 音频线程内 `Vec::new()`/`format!` | alloc 抖动导致爆音 |
| `ort::Session` 泄露到 `vox-convert` | 耦合推理后端 |
| 用 `String` 表示音色 ID/状态 | stringly-typed，用 newtype/enum |
| `unwrap()` 在音频/推理路径 | 实时线程 panic = 进程退出 |
| 跨 `.await` 持 `Mutex` | 死锁风险 |
| 硬编码模型绝对路径 | 跨机器不可用 |
| 在 Rust 内重写模型训练 | 训练留 Python，Rust 只推理 |
| 用 `egui`/`iced` 替换 tauri | 违反技术栈锁定 |
| 日志输出原始音频样本 | PII 风险 |
| 实时管线用 tokio mpsc | 调度抖动，用 crossbeam |

## 提交前自检清单

每次 commit 前确认：

- [ ] `cargo fmt --check` 通过
- [ ] `cargo clippy --workspace -- -D warnings` 通过
- [ ] `cargo test --workspace` 通过
- [ ] 无 `unwrap()`/`expect()` 在音频/推理热路径（grep 检查）
- [ ] 无 `format!`/`Vec::new()` 在标 `#[inline]` 的 DSP 内层循环
- [ ] 新增依赖已在技术栈表或经用户确认
- [ ] 错误类型用 `thiserror`，消息小写无句号
- [ ] 日志用 `tracing`，结构化字段，无 PII
- [ ] 实时管线阶段间用 crossbeam/ringbuf，无 tokio
- [ ] 跨平台代码用 trait 抽象，无 `#[cfg]` 散落业务逻辑

## 与通用 rust-skills 的关系

本 skill 不重复 rust-skills 的 265 条规则，仅在 VoxMorph 上下文中**强化**或**特化**它们：
- `own-`、`err-`、`mem-`、`unsafe-`：完全继承，无例外
- `async-`：仅 GUI/文件 IO 层适用，实时管线禁用 async
- `conc-`：强化为"音频线程专用 OS 线程 + 无锁队列"
- `opt-`：强化为"release 用 fat LTO + codegen-units=1 + panic=abort"
- `test-`：补充音频特定测试（频谱快照、loom 并发、criterion 基准）

冲突时以本 skill 为准。
