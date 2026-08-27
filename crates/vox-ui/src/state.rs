//! Tauri 应用状态：通过 `tauri::State` 在命令间共享。
//!
//! M1 骨架阶段仅持有 `UiState`。后续里程碑加入模型路径、音色选择、
//! 实时管线句柄等。状态变更通过 `tauri::Manager::emit` 通知前端，
//! 不用前端轮询（`obs-structured-fields`）。

use std::sync::Mutex;

use crate::UiState;

/// 应用全局状态，由 `tauri::Builder::manage` 注册。
///
/// 用 `Mutex` 保护（非音频线程，锁开销可接受）。实时管线的指标
/// 通过事件推送，不通过此 State 轮询。
pub struct AppState {
    state: Mutex<UiState>,
}

impl AppState {
    /// 构造初始状态（`Idle`）。
    pub fn new() -> Self {
        Self {
            state: Mutex::new(UiState::Idle),
        }
    }

    /// 读取当前 UI 状态。
    pub fn get(&self) -> UiState {
        *self.state.lock().expect("state mutex poisoned")
    }

    /// 设置 UI 状态。
    pub fn set(&self, new_state: UiState) {
        *self.state.lock().expect("state mutex poisoned") = new_state;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
