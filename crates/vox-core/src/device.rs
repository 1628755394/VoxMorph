//! 跨平台音频设备抽象。
//!
//! 平台实现位于 `vox-io`：Windows(WASAPI) / macOS(CoreAudio) / Linux(ALSA+Pulse)。
//! 业务逻辑通过 trait 操作设备，避免 `#[cfg]` 散落到各 crate。

/// 单个音频设备的只读视图。
pub trait AudioDevice: Send {
    /// 人类可读设备名（如 "Default - Microphone (Realtek)"）。
    fn name(&self) -> &str;

    /// 设备原生采样率 (Hz)。
    fn sample_rate(&self) -> u32;

    /// 声道数。
    fn channels(&self) -> u16;
}

/// 设备枚举器：列出输入/输出设备。
///
/// 实现者应缓存设备列表，热插拔通过 `cpal` 的 `device_change` 回调 +
/// `tracing` 通知 GUI，而非每次调用都重新枚举。
pub trait DeviceEnumerator: Send {
    /// 列出所有可用输入设备。
    fn list_inputs(&self) -> Vec<Box<dyn AudioDevice>>;

    /// 列出所有可用输出设备。
    fn list_outputs(&self) -> Vec<Box<dyn AudioDevice>>;

    /// 系统默认输入设备。
    fn default_input(&self) -> Option<Box<dyn AudioDevice>>;

    /// 系统默认输出设备。
    fn default_output(&self) -> Option<Box<dyn AudioDevice>>;
}
