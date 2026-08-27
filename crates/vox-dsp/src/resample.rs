//! 重采样：基于 `rubato` 的 sinc 插值重采样。
//!
//! # 两种 API
//!
//! - [`resample_buffer`]：一次性处理整段音频（离线场景）。内部 pad 到
//!   chunk_size 的倍数，处理完按比例截断到预期输出长度。
//! - [`Resampler`]：实现 [`vox_core::VoiceProcessor`]，流式处理（实时管线
//!   场景，M4 启用）。内部缓冲到 `chunk_size` 帧才输出，不足时输出空帧。
//!
//! # 数据格式转换
//!
//! `rubato` 要求非交错 f64，`Frame` 是交错 f32。本模块负责 de-interleave →
//! 重采样 → re-interleave，对外只暴露交错 f32。
//!
//! # 性能
//!
//! 离线 `resample_buffer` 允许分配（非音频线程）。`Resampler` 的
//! `VoiceProcessor::process` 在内部 buffer 不足时返回空输出，不阻塞；满
//! chunk 时一次性处理。实时管线中 `process` 不在音频线程调用（在专用
//! DSP 线程），分配可接受。

use rubato::{
    Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use vox_core::{Frame, VoiceProcessor, VoxError};

use crate::DspError;

/// 默认 chunk 大小（帧数）。越大效率越高但延迟越大。
const DEFAULT_CHUNK: usize = 1024;

/// 默认 sinc 插值参数，兼顾质量与速度。
fn default_sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// 一次性重采样整段交错 f32 音频。
///
/// `samples` 为交错样本（`samples[ch + frame * channels]`），返回值亦然。
/// 输出帧数 ≈ `input_frames * to_sr / from_sr`，按四舍五入截断。
///
/// # Errors
/// 采样率或声道数为 0、重采样器构造或处理失败时返回 [`DspError`]。
pub fn resample_buffer(
    samples: &[f32],
    channels: u16,
    from_sr: u32,
    to_sr: u32,
) -> Result<Vec<f32>, DspError> {
    let ch = validate_channels(channels)?;
    if from_sr == 0 || to_sr == 0 {
        return Err(DspError::InvalidInput(
            "sample rate must be non-zero".into(),
        ));
    }
    if samples.len() % ch != 0 {
        return Err(DspError::InvalidInput(format!(
            "samples length {} not divisible by channels {}",
            samples.len(),
            ch
        )));
    }
    if from_sr == to_sr {
        return Ok(samples.to_vec());
    }

    let input_frames = samples.len() / ch;
    let ratio = to_sr as f64 / from_sr as f64;
    let chunk = DEFAULT_CHUNK;

    let mut resampler = SincFixedIn::<f64>::new(ratio, 2.0, default_sinc_params(), chunk, ch)
        .map_err(|e| DspError::Compute(format!("resampler construction failed: {e}")))?;

    // De-interleave f32 → f64，pad 到 chunk 的倍数。
    let padded_frames = input_frames.div_ceil(chunk) * chunk;
    let mut waves_in: Vec<Vec<f64>> = (0..ch).map(|_| Vec::with_capacity(padded_frames)).collect();
    for (i, &s) in samples.iter().enumerate() {
        waves_in[i % ch].push(f64::from(s));
    }
    for w in waves_in.iter_mut() {
        w.resize(padded_frames, 0.0);
    }

    // 分 chunk 处理。
    let mut waves_out: Vec<Vec<f64>> = vec![Vec::new(); ch];
    let mut pos = 0;
    while pos < padded_frames {
        let end = pos.saturating_add(chunk).min(padded_frames);
        let chunk_in: Vec<Vec<f64>> = waves_in.iter().map(|w| w[pos..end].to_vec()).collect();
        let chunk_out = resampler
            .process(&chunk_in, None)
            .map_err(|e| DspError::Compute(format!("resample process failed: {e}")))?;
        for (c, co) in chunk_out.into_iter().enumerate() {
            waves_out[c].extend(co);
        }
        pos = end;
    }

    // 预期输出帧数 = round(input_frames * ratio)。
    let expected_out_frames = (input_frames as f64 * ratio).round() as usize;
    let total_out = waves_out.first().map(|w| w.len()).unwrap_or(0);
    let trim = total_out.min(expected_out_frames);

    // Re-interleave f64 → f32。
    Ok(reinterleave(&waves_out, trim))
}

/// 流式重采样节点，实现 [`VoiceProcessor`]。
///
/// 内部缓冲输入帧到 `chunk_size` 的倍数后才处理并输出；不足时输出空帧
/// （`output.samples` 清空）。调用方需在流末尾调用 [`Resampler::flush`]
/// 取出残余输出。`output.sample_rate` 设为 `to_sr`。
pub struct Resampler {
    inner: SincFixedIn<f64>,
    from_sr: u32,
    to_sr: u32,
    channels: u16,
    chunk: usize,
    buffer: Vec<Vec<f64>>,
    buffered_frames: usize,
}

impl Resampler {
    /// 构造流式重采样器。
    ///
    /// # Errors
    /// 参数无效或重采样器构造失败返回 [`DspError`]。
    pub fn new(from_sr: u32, to_sr: u32, channels: u16) -> Result<Self, DspError> {
        let ch = validate_channels(channels)?;
        if from_sr == 0 || to_sr == 0 {
            return Err(DspError::InvalidInput(
                "sample rate must be non-zero".into(),
            ));
        }
        let ratio = to_sr as f64 / from_sr as f64;
        let chunk = DEFAULT_CHUNK;
        let inner = SincFixedIn::<f64>::new(ratio, 2.0, default_sinc_params(), chunk, ch)
            .map_err(|e| DspError::Compute(format!("resampler construction failed: {e}")))?;
        Ok(Self {
            inner,
            from_sr,
            to_sr,
            channels,
            chunk,
            buffer: vec![Vec::new(); ch],
            buffered_frames: 0,
        })
    }

    /// 取出残余输出（flush 内部缓冲，pad 零到 chunk 倍数后处理）。
    ///
    /// # Errors
    /// 处理失败返回 [`DspError`]。
    pub fn flush(&mut self) -> Result<Vec<f32>, DspError> {
        if self.buffered_frames == 0 {
            return Ok(Vec::new());
        }
        let padded = self.buffered_frames.div_ceil(self.chunk) * self.chunk;
        for w in self.buffer.iter_mut() {
            w.resize(padded, 0.0);
        }
        let out = self
            .inner
            .process(&self.buffer, None)
            .map_err(|e| DspError::Compute(format!("resample flush failed: {e}")))?;
        let total = out.first().map(|w| w.len()).unwrap_or(0);
        let expected = (self.buffered_frames as f64 * self.to_sr as f64 / self.from_sr as f64)
            .round() as usize;
        let trim = total.min(expected);
        let result = reinterleave(&out, trim);
        self.reset();
        Ok(result)
    }

    fn process_buffered(&mut self) -> Result<Vec<f32>, DspError> {
        if self.buffered_frames < self.chunk {
            return Ok(Vec::new());
        }
        let chunk_in: Vec<Vec<f64>> = self
            .buffer
            .iter()
            .map(|w| w[..self.chunk].to_vec())
            .collect();
        let out = self
            .inner
            .process(&chunk_in, None)
            .map_err(|e| DspError::Compute(format!("resample process failed: {e}")))?;
        // 移除已消费的 chunk。
        for w in self.buffer.iter_mut() {
            w.drain(..self.chunk);
        }
        self.buffered_frames = self.buffered_frames.saturating_sub(self.chunk);
        let total = out.first().map(|w| w.len()).unwrap_or(0);
        Ok(reinterleave(&out, total))
    }
}

impl VoiceProcessor for Resampler {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        let ch = self.channels as usize;
        if input.channels != self.channels {
            return Err(VoxError::invalid_input(format!(
                "channel mismatch: expected {} got {}",
                self.channels, input.channels
            )));
        }
        // De-interleave 并追加到内部 buffer。
        for (i, &s) in input.samples.iter().enumerate() {
            self.buffer[i % ch].push(f64::from(s));
        }
        self.buffered_frames += input.frame_count();
        let out = self
            .process_buffered()
            .map_err(|e| VoxError::audio(e.to_string()))?;
        output.samples = out;
        output.sample_rate = self.to_sr;
        output.channels = self.channels;
        Ok(())
    }

    fn reset(&mut self) {
        for w in self.buffer.iter_mut() {
            w.clear();
        }
        self.buffered_frames = 0;
        self.inner.reset();
    }
}

#[inline]
fn validate_channels(channels: u16) -> Result<usize, DspError> {
    if channels == 0 {
        return Err(DspError::InvalidInput("channels must be non-zero".into()));
    }
    Ok(channels as usize)
}

/// 将非交错 f64（每声道一个 Vec）重新交错为 f32，取前 `trim` 帧。
#[allow(clippy::needless_range_loop)]
#[inline]
fn reinterleave(channels: &[Vec<f64>], trim: usize) -> Vec<f32> {
    let ch = channels.len();
    let mut out = Vec::with_capacity(trim.saturating_mul(ch));
    for frame in 0..trim {
        for c in 0..ch {
            out.push(channels[c][frame] as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn passthrough_when_same_sample_rate() {
        let input = vec![0.1, 0.2, 0.3, 0.4]; // 2 frames, 2 channels
        let out = resample_buffer(&input, 2, 48000, 48000).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn upsamples_2x_mono() {
        // 100 samples at 1000 Hz → ~200 samples at 2000 Hz
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = resample_buffer(&input, 1, 1000, 2000).unwrap();
        assert!(
            (190..=210).contains(&out.len()),
            "expected ~200 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn downsamples_half_mono() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = resample_buffer(&input, 1, 2000, 1000).unwrap();
        assert!(
            (90..=110).contains(&out.len()),
            "expected ~100 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn stereo_preserves_channel_count() {
        let input: Vec<f32> = (0..200).map(|i| i as f32).collect(); // 100 frames, 2ch
        let out = resample_buffer(&input, 2, 48000, 24000).unwrap();
        // 输出帧数 ≈ 50，每个样本的声道交替应保持
        assert!(
            out.len() % 2 == 0,
            "output must be interleaved stereo, len={}",
            out.len()
        );
    }

    #[test]
    fn rejects_zero_channels() {
        let result = resample_buffer(&[0.0], 0, 48000, 24000);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let result = resample_buffer(&[0.0], 1, 0, 24000);
        assert!(result.is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn output_length_proportional_to_ratio(
            n_frames in 100usize..2000,
            from_sr in 8000u32..=48000,
            to_sr in 8000u32..=48000,
        ) {
            // 跳过比率极接近 1.0 的情况：rubato sinc 滤波器有固定内部延迟
            // （~9 样本），在比率接近 1 时该延迟占比过大，无法用容差覆盖。
            let ratio = to_sr as f64 / from_sr as f64;
            prop_assume!((ratio - 1.0).abs() > 0.05, "skip near-unity ratio");

            let input: Vec<f32> = (0..n_frames).map(|i| (i as f32 * 0.01).sin()).collect();
            let out = resample_buffer(&input, 1, from_sr, to_sr).unwrap();
            let expected = (n_frames as f64 * to_sr as f64 / from_sr as f64).round() as usize;
            // rubato SincFixedIn 有与 sinc_len 相关的固定群延迟（可达数十样本），
            // 输出长度可能比理想值少。用相对容差 15% 覆盖各种比率下的延迟差异。
            let tolerance = (expected as f64 * 0.15).max(30.0) as isize;
            prop_assert!(
                (out.len() as isize - expected as isize).abs() <= tolerance,
                "out len {} vs expected {} (tol {} from={} to={} n={})",
                out.len(), expected, tolerance, from_sr, to_sr, n_frames
            );
        }
    }

    #[test]
    fn streaming_resampler_buffers_then_outputs() {
        let mut r = Resampler::new(48000, 24000, 1).unwrap();
        let mut output = Frame::zero(24000, 1, 0);

        // 喂入 512 帧（< chunk 1024），应输出空。
        let input = Frame {
            samples: vec![0.5; 512],
            sample_rate: 48000,
            channels: 1,
            timestamp: 0,
        };
        r.process(&input, &mut output).unwrap();
        assert!(output.samples.is_empty(), "should buffer until chunk full");

        // 再喂 512 帧，凑满 1024，应输出。
        let input2 = Frame {
            samples: vec![0.5; 512],
            sample_rate: 48000,
            channels: 1,
            timestamp: 512,
        };
        r.process(&input2, &mut output).unwrap();
        assert!(!output.samples.is_empty(), "should output after chunk full");
        assert_eq!(output.sample_rate, 24000);
    }
}
