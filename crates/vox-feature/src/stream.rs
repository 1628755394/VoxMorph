//! RVC 流式状态：滚动缓冲 + 流式重采样。
//!
//! 借鉴 vc-rs 的 `stream.rs`，管理 RVC 管线的跨 chunk 状态。
//! 每个 chunk 到来时：
//! 1. 追加新音频到设备采样率缓冲
//! 2. 流式重采样到 16kHz（ContentVec/RMVPE 要求）
//! 3. 维护 F0 缓冲（与特征帧对齐）
//! 4. 计算转换窗口大小（convert_size）和输出大小（out_size）
//!
//! # 不变量
//!
//! - `audio_buffer` 和 `audio_16k_buffer` 保持尾部 `convert_size` / `convert_size_16k` 样本
//! - `pitchf_buffer` 保持尾部 `feature_size` 帧
//! - 采样率变化时重置所有状态

use vox_core::VoiceProcessor;
use vox_dsp::resample::Resampler;
use vox_dsp::DspError;

use crate::rvc::{
    align_up, feature_len_for_samples, keep_tail_in_place, samples_between_rates, Rounding,
    CONTENTVEC_CONTEXT_ALIGN_SAMPLES, EMBEDDER_SAMPLE_RATE,
};

/// 流式重采样错误。
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// 重采样错误。
    #[error(transparent)]
    Dsp(#[from] DspError),
    /// 状态未初始化。
    #[error("stream state not initialized: {0}")]
    NotInitialized(String),
}

/// RVC 流式输入参数（每个 chunk 生成时计算）。
#[derive(Debug, Clone)]
pub struct RvcStreamInput {
    /// 转换窗口大小（设备采样率）。
    pub convert_size: usize,
    /// 输出大小（RVC 采样率）。
    pub out_size: usize,
    /// 新增 16kHz 音频的 RMS（用于静音检测）。
    pub input_rms: f32,
}

/// RVC 流式状态：跨 chunk 维护滚动缓冲和重采样器。
///
/// 由模型 worker 线程持有，**不**在音频回调中使用。
pub struct RvcStreamState {
    /// 设备采样率音频缓冲（滚动窗口）。
    pub audio_buffer: Vec<f32>,
    /// 16kHz 音频缓冲（滚动窗口，ContentVec/RMVPE 输入）。
    pub audio_16k_buffer: Vec<f32>,
    /// F0 缓冲（与特征帧对齐，初始填零）。
    pub pitchf_buffer: Vec<f32>,
    /// 当前设备采样率（0 = 未初始化）。
    pub sample_rate: u32,
    /// RVC 模型输出采样率（通常 48kHz）。
    pub rvc_sample_rate: u32,
    /// 流式重采样器（设备采样率 → 16kHz）。
    resampler_16k: Option<Resampler>,
    /// 重采样残余输出缓冲。
    resample_scratch: Vec<f32>,
}

impl RvcStreamState {
    /// 构造流式状态。
    pub fn new(rvc_sample_rate: u32) -> Self {
        Self {
            audio_buffer: Vec::new(),
            audio_16k_buffer: Vec::new(),
            pitchf_buffer: Vec::new(),
            sample_rate: 0,
            rvc_sample_rate,
            resampler_16k: None,
            resample_scratch: Vec::new(),
        }
    }

    /// 重置所有状态（采样率变化或 passthrough 切换时调用）。
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.audio_16k_buffer.clear();
        self.pitchf_buffer.clear();
        self.sample_rate = 0;
        self.resampler_16k = None;
        self.resample_scratch.clear();
    }

    /// 处理新音频 chunk，生成 RVC 推理所需的输入参数。
    ///
    /// # 参数
    /// - `new_audio`: 新增设备采样率音频（单声道）
    /// - `sample_rate`: 设备采样率
    /// - `crossfade_and_search_samples`: SOLA 交叉淡化+搜索样本数（RVC 采样率）
    /// - `volume_excluded_samples`: 音量计算排除的尾部样本数（RVC 采样率）
    /// - `extra_convert_samples`: 额外转换样本数（RVC 采样率，用于 SOLA 余量）
    pub fn generate_input(
        &mut self,
        new_audio: &[f32],
        sample_rate: u32,
        _crossfade_and_search_samples: usize,
        _volume_excluded_samples: usize,
        extra_convert_samples: usize,
    ) -> Result<RvcStreamInput, StreamError> {
        // 采样率变化 → 重置。
        if self.sample_rate != sample_rate {
            self.reset();
            self.sample_rate = sample_rate;
            self.resampler_16k = Some(Resampler::new(sample_rate, EMBEDDER_SAMPLE_RATE, 1)?);
        }

        // 计算新增 16kHz 样本数。
        let new_audio_16k_samples = samples_between_rates(
            new_audio.len(),
            sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let new_feature_len = feature_len_for_samples(new_audio_16k_samples, EMBEDDER_SAMPLE_RATE);

        // 追加新音频到设备采样率缓冲。
        self.audio_buffer.extend_from_slice(new_audio);

        // 流式重采样到 16kHz。
        let frame = vox_core::Frame {
            samples: new_audio.to_vec(),
            sample_rate,
            channels: 1,
            timestamp: 0,
        };
        let mut output = vox_core::Frame::zero(EMBEDDER_SAMPLE_RATE, 1, 0);
        if let Some(resampler) = &mut self.resampler_16k {
            resampler
                .process(&frame, &mut output)
                .map_err(|e| StreamError::Dsp(DspError::Compute(e.to_string())))?;
        } else {
            return Err(StreamError::NotInitialized(
                "16kHz resampler not initialized".into(),
            ));
        }
        self.resample_scratch.extend_from_slice(&output.samples);

        // flush 残余（如果有）。
        if let Some(resampler) = &mut self.resampler_16k {
            let flushed = resampler.flush()?;
            self.resample_scratch.extend_from_slice(&flushed);
        }

        // 追加到 16kHz 缓冲。
        let new_16k_start = self.audio_16k_buffer.len();
        self.audio_16k_buffer
            .extend_from_slice(&self.resample_scratch);
        self.resample_scratch.clear();

        // 计算新增 16kHz 增量的 RMS（用于静音检测）。
        let input_rms = if new_16k_start < self.audio_16k_buffer.len() {
            vox_dsp::rvc::rms(&self.audio_16k_buffer[new_16k_start..])
        } else {
            0.0
        };

        // 扩展 F0 缓冲（新帧填零，后续 RMVPE 填充）。
        self.pitchf_buffer
            .extend(std::iter::repeat(0.0).take(new_feature_len));

        // 计算转换窗口大小。
        let extra_16k_samples = samples_between_rates(
            extra_convert_samples,
            self.rvc_sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let convert_size_16k = align_up(
            new_audio_16k_samples + extra_16k_samples,
            CONTENTVEC_CONTEXT_ALIGN_SAMPLES,
        );
        let convert_size = samples_between_rates(
            convert_size_16k,
            EMBEDDER_SAMPLE_RATE,
            sample_rate,
            Rounding::Ceil,
        );

        // 计算输出大小。
        let out_size = samples_between_rates(
            convert_size_16k.saturating_sub(extra_16k_samples),
            EMBEDDER_SAMPLE_RATE,
            self.rvc_sample_rate,
            Rounding::Floor,
        )
        .max(1);

        let feature_size = feature_len_for_samples(convert_size_16k, EMBEDDER_SAMPLE_RATE);

        // 左侧零填充（启动期或 passthrough 切换后）。
        left_pad_to_len_in_place(&mut self.audio_buffer, convert_size);
        left_pad_to_len_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        left_pad_to_len_in_place(&mut self.pitchf_buffer, feature_size);

        // 保留尾部窗口。
        keep_tail_in_place(&mut self.audio_buffer, convert_size);
        keep_tail_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        keep_tail_in_place(&mut self.pitchf_buffer, feature_size);

        Ok(RvcStreamInput {
            convert_size,
            out_size,
            input_rms,
        })
    }

    /// 获取当前 16kHz 音频窗口（ContentVec/RMVPE 输入）。
    pub fn audio_16k_window(&self) -> &[f32] {
        &self.audio_16k_buffer
    }

    /// 获取当前设备采样率音频窗口。
    pub fn audio_window(&self) -> &[f32] {
        &self.audio_buffer
    }
}

/// 左侧零填充到指定长度（原地）。
fn left_pad_to_len_in_place(values: &mut Vec<f32>, len: usize) {
    if values.len() >= len {
        return;
    }
    let old_len = values.len();
    let pad = len - old_len;
    values.resize(len, 0.0);
    values.copy_within(0..old_len, pad);
    values[..pad].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_empty() {
        let state = RvcStreamState::new(48000);
        assert!(state.audio_buffer.is_empty());
        assert!(state.audio_16k_buffer.is_empty());
        assert_eq!(state.sample_rate, 0);
    }

    #[test]
    fn reset_clears_buffers() {
        let mut state = RvcStreamState::new(48000);
        state.audio_buffer = vec![1.0, 2.0, 3.0];
        state.audio_16k_buffer = vec![0.5; 100];
        state.sample_rate = 48000;
        state.reset();
        assert!(state.audio_buffer.is_empty());
        assert!(state.audio_16k_buffer.is_empty());
        assert_eq!(state.sample_rate, 0);
    }

    #[test]
    fn left_pad_prepends_zeros() {
        let mut v = vec![1.0, 2.0, 3.0];
        left_pad_to_len_in_place(&mut v, 5);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn left_pad_longer_is_noop() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        left_pad_to_len_in_place(&mut v, 3);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn generate_input_initializes_resampler() {
        let mut state = RvcStreamState::new(48000);
        // 4800 samples @ 48kHz = 100ms → 1600 samples @ 16kHz
        let audio = vec![0.1; 4800];
        let input = state.generate_input(&audio, 48000, 0, 0, 0).unwrap();
        assert_eq!(state.sample_rate, 48000);
        assert!(input.convert_size > 0);
        assert!(input.out_size > 0);
        assert!(state.resampler_16k.is_some());
    }

    #[test]
    fn generate_input_sample_rate_change_resets() {
        let mut state = RvcStreamState::new(48000);
        // 先用 48kHz。
        state.generate_input(&[0.1; 4800], 48000, 0, 0, 0).unwrap();
        assert_eq!(state.sample_rate, 48000);
        // 切换到 44100 → 应重置。
        state.generate_input(&[0.1; 4410], 44100, 0, 0, 0).unwrap();
        assert_eq!(state.sample_rate, 44100);
    }
}
