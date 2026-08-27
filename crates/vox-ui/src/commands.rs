//! Tauri 命令：前端通过 `invoke('command_name')` 调用。
//!
//! 所有命令返回 `Result<T, String>`（`tauri-command-result-string`）。
//! 重型操作在 `tokio::task::spawn_blocking` 内执行（`async-spawn-blocking`）。

use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use vox_core::DeviceEnumerator;
use vox_io::cpal::CpalEnumerator;

use crate::{AppState, PipelineMetrics, UiState};

// ── 状态查询 ──────────────────────────────────────────────────────────

/// 查询当前 UI 状态。
#[tauri::command]
pub fn get_state(state: State<'_, AppState>) -> Result<UiState, String> {
    Ok(state.get())
}

// ── 音频设备 ──────────────────────────────────────────────────────────

/// 音频设备的序列化视图（供前端展示）。
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub is_default: bool,
}

/// 列出所有可用音频输入/输出设备。
#[tauri::command]
pub fn list_audio_devices() -> Result<Vec<DeviceInfo>, String> {
    let en = CpalEnumerator::new();
    let mut devices = Vec::new();

    let default_input = en.default_input();
    let default_output = en.default_output();

    for dev in en.list_inputs() {
        let is_default = default_input
            .as_ref()
            .map(|d| d.name() == dev.name())
            .unwrap_or(false);
        devices.push(DeviceInfo {
            name: dev.name().to_string(),
            sample_rate: dev.sample_rate(),
            channels: dev.channels(),
            is_default,
        });
    }
    for dev in en.list_outputs() {
        let is_default = default_output
            .as_ref()
            .map(|d| d.name() == dev.name())
            .unwrap_or(false);
        devices.push(DeviceInfo {
            name: dev.name().to_string(),
            sample_rate: dev.sample_rate(),
            channels: dev.channels(),
            is_default,
        });
    }

    Ok(devices)
}

// ── 音色库 ────────────────────────────────────────────────────────────

/// 音色信息（供前端展示，不含 embedding 数据）。
#[derive(Debug, Clone, Serialize)]
pub struct TimbreInfo {
    pub id: u64,
    pub name: String,
    pub f0_offset_semitones: f32,
    pub tags: Vec<String>,
}

/// 加载音色库目录。
///
/// 扫描目录下所有 `.toml` + `.bin` 文件对，构建音色库索引。
#[tauri::command]
pub async fn load_timbre_library(
    app: AppHandle,
    state: State<'_, AppState>,
    dir: String,
) -> Result<Vec<TimbreInfo>, String> {
    state.set(UiState::LoadingModel);
    let _ = app.emit("state-changed", state.get());

    let dir_path = PathBuf::from(&dir);
    let dir_path_for_load = dir_path.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        vox_models::TimbreLibrary::load_from_dir(&dir_path_for_load)
    })
    .await
    .map_err(|e| format!("failed to load timbre library: {e}"))?;

    let library = result.map_err(|e| e.to_string())?;
    state.set_timbre_dir(dir_path);
    let timbres: Vec<TimbreInfo> = library
        .list()
        .iter()
        .map(|t| TimbreInfo {
            id: t.id.get(),
            name: t.name.clone(),
            f0_offset_semitones: t.f0_offset_semitones,
            tags: t.tags.clone(),
        })
        .collect();

    state.set_timbre_library(library);
    state.set(UiState::Ready);
    let _ = app.emit("state-changed", state.get());

    Ok(timbres)
}

/// 列出已加载的音色库中的所有音色。
#[tauri::command]
pub fn list_timbres(state: State<'_, AppState>) -> Result<Vec<TimbreInfo>, String> {
    let list = state.timbre_list().ok_or("no timbre library loaded")?;
    Ok(list
        .into_iter()
        .map(|(id, name, f0, tags)| TimbreInfo {
            id,
            name,
            f0_offset_semitones: f0,
            tags,
        })
        .collect())
}

// ── 实时引擎控制 ──────────────────────────────────────────────────────

/// 启动实时变声引擎。
///
/// 使用默认 cpal 设备 + 空管线（M7 框架，后续接入完整 Stage 链）。
#[tauri::command]
pub async fn start_engine(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if state.is_engine_running() {
        return Err("engine already running".into());
    }

    state.set(UiState::Running);
    let _ = app.emit("state-changed", state.get());

    // M7 框架：用空管线启动（无 Stage，passthrough 效果）。
    // 完整实现需加载 HuBERT/Converter/Vocoder ONNX 模型并构建 Stage 链。
    let pipeline = vox_convert::Pipeline::new(vox_convert::PipelineConfig::default_16k_mono());
    let config = vox_convert::RealtimeEngineConfig::default();

    // 引擎在专用线程中启动（cpal 流创建 + 保活）。
    state.start_engine(pipeline, config);

    if !state.is_engine_running() {
        state.set(UiState::Error);
        let _ = app.emit("state-changed", state.get());
        return Err("engine failed to start".into());
    }

    let _ = app.emit("state-changed", state.get());
    Ok(())
}

/// 停止实时变声引擎。
#[tauri::command]
pub async fn stop_engine(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if !state.is_engine_running() {
        return Err("engine not running".into());
    }

    state.stop_engine();

    state.set(UiState::Ready);
    let _ = app.emit("state-changed", state.get());

    Ok(())
}

/// 查询引擎是否正在运行。
#[tauri::command]
pub fn is_engine_running(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(state.is_engine_running())
}

// ── 指标 ──────────────────────────────────────────────────────────────

/// 查询当前管线指标。
#[tauri::command]
pub fn get_metrics(state: State<'_, AppState>) -> Result<PipelineMetrics, String> {
    Ok(PipelineMetrics {
        buffer_level: 0,
        infer_ms: 0.0,
        dropped_frames: 0,
        state: state.get(),
    })
}

/// 推送指标快照到前端（供定时器调用）。
#[tauri::command]
pub fn emit_metrics(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let metrics = PipelineMetrics {
        buffer_level: 0,
        infer_ms: 0.0,
        dropped_frames: 0,
        state: state.get(),
    };
    app.emit("metrics-update", &metrics)
        .map_err(|e| e.to_string())
}
