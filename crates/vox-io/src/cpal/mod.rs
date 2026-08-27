//! cpal 后端：设备枚举、输入源、输出汇。
//!
//! 模块按 cpal 这一"feature"聚合，内部再按职责拆 `device` / `source` / `sink`
//! （`proj-mod-by-feature`）。所有平台差异由 cpal 内部处理，本模块不出现
//! `#[cfg]` 业务分支。
//!
//! - [`device`]：[`CpalDevice`] / [`CpalEnumerator`] 实现 `vox_core` 的
//!   `AudioDevice` / `DeviceEnumerator`。
//! - [`source`]：`CpalSource` 实现 `AudioSource`（M1 Slice 2）。
//! - [`sink`]：`CpalSink` 实现 `AudioSink`（M1 Slice 3）。

pub mod device;
pub mod sink;
pub mod source;

pub use device::{CpalDevice, CpalEnumerator};
pub use sink::CpalSink;
pub use source::CpalSource;
