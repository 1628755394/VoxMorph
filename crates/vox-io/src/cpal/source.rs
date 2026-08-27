//! cpal 输入源：把 cpal 输入流的回调式数据桥接成 [`vox_core::AudioSource`]
//! 的拉取式接口。
//!
//! 线程模型：cpal 在专用音频线程回调 `push_slice` 写入无锁 ringbuf 的生产端，
//! 调用方在工作线程通过 [`CpalSource::read`] 从消费端 `pop_slice`。两侧均无
//! 堆分配、无锁（`mem-reuse-collections`、`conc-` 无锁队列）。
//!
//! `cpal::Stream` 被 cpal 标记为 `!Send`（`NotSendSyncAcrossAllPlatforms`），
//! 故 [`CpalSource::new`] 把流句柄单独返回，由调用方在创建线程保活；`CpalSource`
//! 本身只持有 `Send` 的 ringbuf 消费端，可在工作线程间移动。
//!
//! 采样格式仅支持 `F32`；设备默认配置非 F32 时返回 [`AudioError::Unsupported`]，
//! 重采样留给 M2 的 `vox-dsp`。ringbuf 容量约 100ms，溢出时丢弃溢出部分并
//! `tracing::trace!` 上报（背压丢帧，非致命）。

use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::SampleFormat;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapCons, HeapRb};
use vox_core::{AudioSource, VoxError};

use crate::cpal::device::CpalDevice;
use crate::AudioError;

/// 基于 cpal 输入流的 [`AudioSource`]。
///
/// 不持有 cpal `Stream`（见模块文档）；流句柄由 [`CpalSource::new`] 返回，
/// 调用方负责保活。drop `CpalSource` 不会停止采集，停止采集需 drop 流句柄。
pub struct CpalSource {
    consumer: HeapCons<f32>,
    sample_rate: u32,
    channels: u16,
}

impl CpalSource {
    /// 由 [`CpalDevice`] 打开输入流，返回 (`source`, `stream`)。
    ///
    /// `stream` 必须由调用方在创建它的线程上保活（drop 即停止采集）。
    ///
    /// # Errors
    /// 默认输入配置查询失败返回 [`AudioError::Unavailable`]；采样格式非 F32
    /// 返回 [`AudioError::Unsupported`]；建流或启动失败返回 [`AudioError::Device`]。
    pub fn new(device: CpalDevice) -> Result<(Self, cpal::Stream), AudioError> {
        let dev = device.into_device();
        let cfg = dev
            .default_input_config()
            .map_err(|e| AudioError::Unavailable(e.to_string()))?;
        if cfg.sample_format() != SampleFormat::F32 {
            return Err(AudioError::Unsupported(format!(
                "input sample format {:?} not supported (require f32)",
                cfg.sample_format()
            )));
        }
        let sample_rate = cfg.sample_rate().0;
        let channels = cfg.channels();
        let stream_cfg: cpal::StreamConfig = cfg.into();

        // 容量 ≈ 100ms，下限 4096 样本，吸收回调抖动而不无限增长。
        let cap = ((sample_rate as usize) * (channels as usize) / 10).max(4096);
        let rb = HeapRb::<f32>::new(cap);
        let (mut prod, cons) = rb.split();

        let stream = dev
            .build_input_stream(
                &stream_cfg,
                move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                    let pushed = prod.push_slice(data);
                    if pushed < data.len() {
                        // ringbuf 满 → 丢弃溢出（背压）。trace 级别默认关闭，无 alloc。
                        tracing::trace!(dropped = data.len() - pushed, "input ringbuf overflow");
                    }
                },
                |err| tracing::error!(error = %err, "input stream error"),
                None,
            )
            .map_err(|e| AudioError::Device(e.to_string()))?;
        stream
            .play()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        Ok((
            Self {
                consumer: cons,
                sample_rate,
                channels,
            },
            stream,
        ))
    }
}

impl AudioSource for CpalSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize, VoxError> {
        // 非阻塞：返回当前可用的样本数；0 表示欠载（mic 永不到达 EOF）。
        Ok(self.consumer.pop_slice(out))
    }
}
