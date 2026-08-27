//! VoxMorph Tauri GUI 后端：命令、事件、状态机。
//!
//! # 命令约定
//!
//! - 所有 `#[tauri::command]` 返回 `Result<T, String>`，**不**返回 `anyhow::Result`
//!   （serde 序列化问题）。在边界把 `anyhow::Error` 用 `?` + `to_string()` 转换。
//! - 重型操作（文件处理、模型加载）在 `tokio::task::spawn_blocking` 内执行
//!   （`async-spawn-blocking`）。
//!
//! # 状态机
//!
//! 前端状态用 enum 不用 string（`type-enum-states`）：
//! `idle / loading-model / ready / running / error`
//!
//! # 事件推送
//!
//! 实时音频状态（缓冲水位、推理耗时、丢帧数）通过 `tauri::Manager::emit` 推送，
//! **不**用前端轮询。仪表盘数据走事件通道，不走日志通道。
//!
//! # 实现里程碑
//!
//! - M1: Tauri 工程脚手架 + 命令骨架
//! - M5: 音色库管理 UI、参数滑块、波形/频谱可视化
//! - M7: 打包发布

mod commands;
mod state;

use serde::{Deserialize, Serialize};

pub use commands::*;
pub use state::AppState;

/// GUI 状态机。用 enum 而非 string，避免 stringly-typed（`type-enum-states`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UiState {
    /// 空闲，未加载模型。
    Idle,
    /// 正在加载模型/音色。
    LoadingModel,
    /// 就绪，可启动变声。
    Ready,
    /// 变声运行中。
    Running,
    /// 错误状态。
    Error,
}

/// 推送给前端仪表盘的实时指标。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PipelineMetrics {
    /// 当前缓冲水位（帧数）。
    pub buffer_level: u32,
    /// 上一帧推理耗时（毫秒）。
    pub infer_ms: f32,
    /// 累计丢帧数。
    pub dropped_frames: u64,
    /// 当前状态。
    pub state: UiState,
}

/// Tauri 应用入口。由 `main.rs` 调用（`proj-lib-main-split`）。
///
/// 注册命令、事件，启动 Tauri 窗口。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("vox_ui=info")),
        )
        .init();

    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::get_state,
            commands::list_audio_devices,
            commands::load_timbre_library,
            commands::list_timbres,
            commands::start_engine,
            commands::stop_engine,
            commands::is_engine_running,
            commands::get_metrics,
            commands::emit_metrics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
