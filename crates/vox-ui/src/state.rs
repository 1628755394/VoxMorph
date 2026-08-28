//! Tauri 应用状态：通过 `tauri::State` 在命令间共享。
//!
//! 持有 UI 状态机、音色库、引擎控制句柄。
//! 状态变更通过 `tauri::Manager::emit` 通知前端，不用前端轮询。
//!
//! # 引擎句柄
//!
//! `cpal::Stream` 是 `!Send`，不能直接存入 `AppState`（需 `Send + Sync`）。
//! 故引擎句柄由专用"引擎管理线程"持有，`AppState` 只存 `EngineControl`
//!（原子标志 + channel），通过消息控制引擎线程。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use vox_convert::{LiveParamsHandle, PipelineMetrics};
use vox_models::TimbreLibrary;

/// 音色摘要信息（id, name, f0_offset, tags），供命令层提取。
pub type TimbreSummary = (u64, String, f32, Vec<String>);

use crate::UiState;

/// 引擎控制句柄：通过原子标志和 channel 控制引擎管理线程。
///
/// `Send + Sync`，可安全存入 `AppState`。
pub struct EngineControl {
    /// 停止标志：设为 true → 引擎管理线程停止引擎并退出。
    stop_flag: Arc<AtomicBool>,
    /// 引擎是否运行中。
    running: Arc<AtomicBool>,
    /// 引擎管理线程的 JoinHandle（用 Mutex 包装以便 take）。
    join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl EngineControl {
    /// 构造控制句柄（未启动状态）。
    pub fn new() -> Self {
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            join_handle: Mutex::new(None),
        }
    }

    /// 启动引擎：在专用线程中运行 RealtimeEngine。
    ///
    /// 引擎管理线程持有 cpal::Stream（!Send），故不能跨线程移动。
    /// 线程内创建流、运行引擎、等待 stop 信号。
    pub fn start(
        &self,
        pipeline: vox_convert::Pipeline,
        config: vox_convert::RealtimeEngineConfig,
    ) {
        let stop_flag = Arc::clone(&self.stop_flag);
        let running = Arc::clone(&self.running);

        stop_flag.store(false, Ordering::Relaxed);
        running.store(true, Ordering::Relaxed);

        let handle = std::thread::Builder::new()
            .name("voxmorph-engine-manager".to_string())
            .spawn(move || {
                // 在此线程内创建 cpal 流并运行引擎。
                let engine_handle =
                    match vox_convert::RealtimeEngine::start_with_default_devices(pipeline, config)
                    {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to start realtime engine");
                            running.store(false, Ordering::Relaxed);
                            return;
                        }
                    };

                // 等待 stop 信号。
                while !stop_flag.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                // 停止引擎（在此线程内 drop cpal::Stream）。
                engine_handle.stop();
                running.store(false, Ordering::Relaxed);
            })
            .expect("failed to spawn engine manager thread");

        *self.join_handle.lock().expect("join handle mutex poisoned") = Some(handle);
    }

    /// 停止引擎：设置 stop flag → 等待引擎管理线程退出。
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self
            .join_handle
            .lock()
            .expect("join handle mutex poisoned")
            .take()
        {
            let _ = handle.join();
        }
        self.running.store(false, Ordering::Relaxed);
    }

    /// 引擎是否正在运行。
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Default for EngineControl {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EngineControl {
    fn drop(&mut self) {
        if self.is_running() {
            self.stop();
        }
    }
}

/// 应用全局状态，由 `tauri::Builder::manage` 注册。
///
/// 用 `Mutex` 保护（非音频线程，锁开销可接受）。实时管线的指标
/// 通过事件推送，不通过此 State 轮询。
pub struct AppState {
    state: Mutex<UiState>,
    /// 已加载的音色库（可选，加载模型后才有）。
    timbre_library: Mutex<Option<TimbreLibrary>>,
    /// 引擎控制句柄。
    engine_control: Mutex<EngineControl>,
    /// 音色库目录路径（用户配置）。
    timbre_dir: Mutex<Option<PathBuf>>,
    /// 当前引擎的管线指标（引擎运行时有值，停止后清空）。
    pipeline_metrics: Mutex<Option<Arc<PipelineMetrics>>>,
    /// RVC 实时参数共享句柄（RVC 引擎运行时有值）。
    live_params: Mutex<Option<LiveParamsHandle>>,
}

impl AppState {
    /// 构造初始状态（`Idle`）。
    pub fn new() -> Self {
        Self {
            state: Mutex::new(UiState::Idle),
            timbre_library: Mutex::new(None),
            engine_control: Mutex::new(EngineControl::new()),
            timbre_dir: Mutex::new(None),
            pipeline_metrics: Mutex::new(None),
            live_params: Mutex::new(None),
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

    /// 设置音色库。
    pub fn set_timbre_library(&self, library: TimbreLibrary) {
        *self
            .timbre_library
            .lock()
            .expect("timbre library mutex poisoned") = Some(library);
    }

    /// 获取音色列表（在锁内提取信息，避免长期持锁）。
    pub fn timbre_list(&self) -> Option<Vec<TimbreSummary>> {
        self.timbre_library
            .lock()
            .expect("timbre library mutex poisoned")
            .as_ref()
            .map(|lib| {
                lib.list()
                    .iter()
                    .map(|t| {
                        (
                            t.id.get(),
                            t.name.clone(),
                            t.f0_offset_semitones,
                            t.tags.clone(),
                        )
                    })
                    .collect()
            })
    }

    /// 设置音色库目录路径。
    pub fn set_timbre_dir(&self, dir: PathBuf) {
        *self.timbre_dir.lock().expect("timbre dir mutex poisoned") = Some(dir);
    }

    /// 获取音色库目录路径。
    pub fn timbre_dir(&self) -> Option<PathBuf> {
        self.timbre_dir
            .lock()
            .expect("timbre dir mutex poisoned")
            .clone()
    }

    /// 启动引擎。
    pub fn start_engine(
        &self,
        pipeline: vox_convert::Pipeline,
        config: vox_convert::RealtimeEngineConfig,
    ) {
        // 在 pipeline 移入引擎线程前，克隆 metrics Arc 供 GUI 查询。
        let metrics = pipeline.metrics();
        *self
            .pipeline_metrics
            .lock()
            .expect("pipeline metrics mutex poisoned") = Some(metrics);

        self.engine_control
            .lock()
            .expect("engine control mutex poisoned")
            .start(pipeline, config);
    }

    /// 停止引擎。
    pub fn stop_engine(&self) {
        self.engine_control
            .lock()
            .expect("engine control mutex poisoned")
            .stop();

        *self
            .pipeline_metrics
            .lock()
            .expect("pipeline metrics mutex poisoned") = None;
    }

    /// 获取管线指标快照（引擎运行时才有）。
    pub fn pipeline_metrics_snapshot(&self) -> Option<vox_convert::MetricsSnapshot> {
        self.pipeline_metrics
            .lock()
            .expect("pipeline metrics mutex poisoned")
            .as_ref()
            .map(|m| m.snapshot())
    }

    /// 设置 RVC 实时参数句柄（启动 RVC 引擎时调用）。
    pub fn set_live_params_handle(&self, handle: LiveParamsHandle) {
        *self.live_params.lock().expect("live params mutex poisoned") = Some(handle);
    }

    /// 获取 RVC 实时参数句柄（供 `set_live_params` 命令使用）。
    pub fn live_params_handle(&self) -> Option<LiveParamsHandle> {
        self.live_params
            .lock()
            .expect("live params mutex poisoned")
            .clone()
    }

    /// 清除 RVC 实时参数句柄（停止引擎时调用）。
    pub fn clear_live_params_handle(&self) {
        *self.live_params.lock().expect("live params mutex poisoned") = None;
    }

    /// 引擎是否正在运行。
    pub fn is_engine_running(&self) -> bool {
        self.engine_control
            .lock()
            .expect("engine control mutex poisoned")
            .is_running()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
