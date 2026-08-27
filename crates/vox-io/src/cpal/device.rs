//! cpal 设备抽象：把 `cpal::Device` 包装成 [`vox_core::AudioDevice`]，
//! 把 `cpal::Host` 包装成 [`vox_core::DeviceEnumerator`]。
//!
//! 构造发生在主线程（枚举器调用），允许分配 `String` 缓存设备名。
//! `sample_rate` / `channels` 取设备默认配置并缓存，避免热路径重复查询。
//!
//! 设备枚举失败（如设备断开）通过 `tracing::warn!` 记录并跳过，不向上
//! 传播错误——`DeviceEnumerator` trait 返回 `Vec`/`Option`，无 `Result`
//! 变体（`err-result-over-panic`、`obs-tracing-over-log`）。

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host};
use vox_core::{AudioDevice, DeviceEnumerator};

use crate::AudioError;

/// cpal 设备的只读视图，缓存名称、采样率与声道数。
///
/// 一个 `CpalDevice` 只代表一个方向（输入或输出），因为 `AudioDevice` trait
/// 不区分方向，而 cpal 的默认配置查询是按方向进行的。
pub struct CpalDevice {
    // `CpalSource`/`CpalSink` 通过 `into_device` 取走底层设备打开流。
    device: Device,
    name: String,
    sample_rate: u32,
    channels: u16,
}

impl CpalDevice {
    /// 由输入设备构造，查询其默认输入配置。
    ///
    /// # Errors
    /// 设备名查询失败返回 [`AudioError::Device`]；默认配置查询失败（设备
    /// 断开或不支持任何配置）返回 [`AudioError::Unavailable`]。
    pub fn new_input(device: Device) -> Result<Self, AudioError> {
        Self::new(device, Direction::Input)
    }

    /// 由输出设备构造，查询其默认输出配置。
    ///
    /// # Errors
    /// 同 [`CpalDevice::new_input`]。
    pub fn new_output(device: Device) -> Result<Self, AudioError> {
        Self::new(device, Direction::Output)
    }

    /// 取底层 cpal 设备，供 `CpalSource` / `CpalSink` 打开流时使用。
    pub(crate) fn into_device(self) -> Device {
        self.device
    }

    fn new(device: Device, dir: Direction) -> Result<Self, AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::Device(e.to_string()))?;
        let cfg = match dir {
            Direction::Input => device.default_input_config(),
            Direction::Output => device.default_output_config(),
        }
        .map_err(|e| AudioError::Unavailable(e.to_string()))?;
        let sample_rate = cfg.sample_rate();
        let channels = cfg.channels();
        Ok(Self {
            device,
            name,
            sample_rate: sample_rate.0,
            channels,
        })
    }
}

impl AudioDevice for CpalDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}

// `cpal::Device` 在所有支持平台均为 `Send`，`String` 亦然。
unsafe impl Send for CpalDevice {}

#[derive(Clone, Copy)]
enum Direction {
    Input,
    Output,
}

/// cpal `Host` 的设备枚举器封装。
///
/// M1 每次调用都重新枚举设备；设备列表缓存与热插拔回调留待后续里程碑
/// （见 `voxmorph` skill 的 `DeviceEnumerator` 注释）。
pub struct CpalEnumerator {
    host: Host,
}

impl CpalEnumerator {
    /// 以系统默认 host 构造枚举器。
    pub fn new() -> Self {
        Self {
            host: cpal::default_host(),
        }
    }

    /// 默认输入设备，返回具体类型 [`CpalDevice`] 供 `CpalSource::new` 使用。
    ///
    /// 与 [`DeviceEnumerator::default_input`] 的区别：后者返回 `Box<dyn AudioDevice>`
    /// 无法 downcast 回 `CpalDevice`，故提供此具体方法。
    pub fn default_input_cpal(&self) -> Option<CpalDevice> {
        let dev = self.host.default_input_device()?;
        match CpalDevice::new_input(dev) {
            Ok(cd) => Some(cd),
            Err(e) => {
                tracing::warn!(error = %e, "default input device unavailable");
                None
            }
        }
    }

    /// 默认输出设备，返回具体类型 [`CpalDevice`] 供 `CpalSink::new` 使用。
    pub fn default_output_cpal(&self) -> Option<CpalDevice> {
        let dev = self.host.default_output_device()?;
        match CpalDevice::new_output(dev) {
            Ok(cd) => Some(cd),
            Err(e) => {
                tracing::warn!(error = %e, "default output device unavailable");
                None
            }
        }
    }
}

impl Default for CpalEnumerator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceEnumerator for CpalEnumerator {
    fn list_inputs(&self) -> Vec<Box<dyn AudioDevice>> {
        let mut out: Vec<Box<dyn AudioDevice>> = Vec::new();
        let devices = match self.host.input_devices() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "enumerate input devices failed");
                return out;
            }
        };
        for dev in devices {
            match CpalDevice::new_input(dev) {
                Ok(cd) => out.push(Box::new(cd)),
                Err(e) => tracing::warn!(error = %e, "skip input device"),
            }
        }
        out
    }

    fn list_outputs(&self) -> Vec<Box<dyn AudioDevice>> {
        let mut out: Vec<Box<dyn AudioDevice>> = Vec::new();
        let devices = match self.host.output_devices() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "enumerate output devices failed");
                return out;
            }
        };
        for dev in devices {
            match CpalDevice::new_output(dev) {
                Ok(cd) => out.push(Box::new(cd)),
                Err(e) => tracing::warn!(error = %e, "skip output device"),
            }
        }
        out
    }

    fn default_input(&self) -> Option<Box<dyn AudioDevice>> {
        let dev = self.host.default_input_device()?;
        match CpalDevice::new_input(dev) {
            Ok(cd) => Some(Box::new(cd)),
            Err(e) => {
                tracing::warn!(error = %e, "default input device unavailable");
                None
            }
        }
    }

    fn default_output(&self) -> Option<Box<dyn AudioDevice>> {
        let dev = self.host.default_output_device()?;
        match CpalDevice::new_output(dev) {
            Ok(cd) => Some(Box::new(cd)),
            Err(e) => {
                tracing::warn!(error = %e, "default output device unavailable");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerator_constructs_with_default_host() {
        // 仅验证构造不 panic；不假设 CI 环境一定有音频设备。
        let _ = CpalEnumerator::new();
        let _ = CpalEnumerator::default();
    }

    #[test]
    fn list_inputs_does_not_panic_without_devices() {
        let en = CpalEnumerator::new();
        // 在无音频设备的 CI 容器中应返回空 vec 而非 panic。
        let _ = en.list_inputs();
        let _ = en.list_outputs();
    }
}
