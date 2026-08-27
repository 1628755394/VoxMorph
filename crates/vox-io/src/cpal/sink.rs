//! cpal 输出汇：把 [`vox_core::AudioSink`] 的推送式接口桥接成 cpal 输出流
//! 的回调式数据。
//!
//! 线程模型：调用方在工作线程通过 [`CpalSink::write`] 写入无锁 ringbuf 的
//! 生产端，cpal 在专用音频线程回调 `pop_slice` 从消费端拉取。两侧均无堆
//! 分配、无锁（`mem-reuse-collections`、`conc-` 无锁队列）。
//!
//! 与 [`crate::cpal::source::CpalSource`] 同理，`cpal::Stream` 被 cpal 标记
//! 为 `!Send`，故 [`CpalSink::new`] 把流句柄单独返回，由调用方在创建线程
//! 保活；`CpalSink` 本身只持有 `Send` 的 ringbuf 生产端。
//!
//! 采样格式仅支持 `F32`；ringbuf 容量约 100ms。输出回调欠载（ringbuf 空）
//! 时填零静音，不阻塞音频线程（`err-result-over-panic`：欠载非错误，降级
//! 为静音）。`write` 在 ringbuf 满时返回 [`VoxError::Dropped`] 触发背压
//! 统计，而非阻塞或无限增长。

use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::SampleFormat;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapProd, HeapRb};
use vox_core::{AudioSink, VoxError};

use crate::cpal::device::CpalDevice;
use crate::AudioError;

/// 基于 cpal 输出流的 [`AudioSink`]。
///
/// 不持有 cpal `Stream`（见模块文档）；流句柄由 [`CpalSink::new`] 返回，
/// 调用方负责保活。drop `CpalSink` 不会停止播放，停止播放需 drop 流句柄。
pub struct CpalSink {
    producer: HeapProd<f32>,
    sample_rate: u32,
    channels: u16,
}

impl CpalSink {
    /// 由 [`CpalDevice`] 打开输出流，返回 (`sink`, `stream`)。
    ///
    /// `stream` 必须由调用方在创建它的线程上保活（drop 即停止播放）。
    ///
    /// # Errors
    /// 默认输出配置查询失败返回 [`AudioError::Unavailable`]；采样格式非 F32
    /// 返回 [`AudioError::Unsupported`]；建流或启动失败返回 [`AudioError::Device`]。
    pub fn new(device: CpalDevice) -> Result<(Self, cpal::Stream), AudioError> {
        let dev = device.into_device();
        let cfg = dev
            .default_output_config()
            .map_err(|e| AudioError::Unavailable(e.to_string()))?;
        if cfg.sample_format() != SampleFormat::F32 {
            return Err(AudioError::Unsupported(format!(
                "output sample format {:?} not supported (require f32)",
                cfg.sample_format()
            )));
        }
        let sample_rate = cfg.sample_rate().0;
        let channels = cfg.channels();
        let stream_cfg: cpal::StreamConfig = cfg.into();

        // 容量 ≈ 100ms，下限 4096 样本，吸收回调抖动而不无限增长。
        let cap = ((sample_rate as usize) * (channels as usize) / 10).max(4096);
        let rb = HeapRb::<f32>::new(cap);
        let (prod, mut cons) = rb.split();

        let stream = dev
            .build_output_stream(
                &stream_cfg,
                move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    let popped = cons.pop_slice(data);
                    // 欠载部分填零静音，不阻塞音频线程。
                    for s in data[popped..].iter_mut() {
                        *s = 0.0;
                    }
                    if popped < data.len() {
                        tracing::trace!(underrun = data.len() - popped, "output ringbuf underrun");
                    }
                },
                |err| tracing::error!(error = %err, "output stream error"),
                None,
            )
            .map_err(|e| AudioError::Device(e.to_string()))?;
        stream
            .play()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        Ok((
            Self {
                producer: prod,
                sample_rate,
                channels,
            },
            stream,
        ))
    }
}

impl AudioSink for CpalSink {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn write(&mut self, samples: &[f32]) -> Result<(), VoxError> {
        let pushed = self.producer.push_slice(samples);
        if pushed < samples.len() {
            // ringbuf 满 → 背压丢帧，上报而非阻塞或无限增长。
            tracing::trace!(dropped = samples.len() - pushed, "output ringbuf overflow");
            return Err(VoxError::Dropped);
        }
        Ok(())
    }
}
