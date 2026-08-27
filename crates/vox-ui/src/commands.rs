//! Tauri 命令：前端通过 `invoke('command_name')` 调用。
//!
//! 所有命令返回 `Result<T, String>`（`tauri-command-result-string`）。
//! 重型操作在 `tokio::task::spawn_blocking` 内执行（`async-spawn-blocking`）。

use serde::Serialize;
use tauri::State;
use vox_core::DeviceEnumerator;
use vox_io::cpal::CpalEnumerator;

use crate::{AppState, PipelineMetrics, UiState};

/// 查询当前 UI 状态。
#[tauri::command]
pub fn get_state(state: State<'_, AppState>) -> Result<UiState, String> {
    Ok(state.get())
}

/// 音频设备的序列化视图（供前端展示）。
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub is_default: bool,
}

/// 列出所有可用音频输入/输出设备。
///
/// 枚举在调用线程同步执行（cpal 枚举通常 <10ms，不需 spawn_blocking）。
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

/// 查询当前管线指标（M1 骨架返回默认值，后续里程碑接入实时数据）。
#[tauri::command]
pub fn get_metrics(state: State<'_, AppState>) -> Result<PipelineMetrics, String> {
    Ok(PipelineMetrics {
        buffer_level: 0,
        infer_ms: 0.0,
        dropped_frames: 0,
        state: state.get(),
    })
}
