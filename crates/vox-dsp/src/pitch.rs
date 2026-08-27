//! Phase vocoder pitch shift。
//!
//! # 算法
//!
//! 基于 STFT 的 phase vocoder：对每帧做 Hann 窗 FFT，在频域按 `semitones`
//! 移动幅度谱 bin，相位用"相位累积"法保持连续性，再 IFFT + overlap-add
//! 还原。合成 hop = 分析 hop，故时长不变，仅音高移动。
//!
//! # 局限
//!
//! - 不做共振峰保持（formant preservation），升调会变"花栗鼠"、降调变"巨人"
//! - 离线实现，整段处理，非流式（M2 demo 用途）
//! - 单声道处理；多声道按声道分别处理
//!
//! # 性能
//!
//! 离线场景允许分配。FFT 大小 2048，hop 512（75% 重叠），兼顾质量与速度。

use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};
use vox_core::{Frame, VoiceProcessor, VoxError};

use crate::DspError;

/// 默认 FFT 大小（2 的幂）。
const FFT_SIZE: usize = 2048;
/// 默认 hop（分析与合成相同，保持时长）。
const HOP: usize = FFT_SIZE / 4; // 75% overlap
/// 半音到频率比的转换因子。
const SEMITONE_RATIO: f64 = 1.0594630943592953; // 2^(1/12)

/// Phase vocoder pitch shifter，实现 [`VoiceProcessor`]。
///
/// 离线用法：构造后调用 [`PitchShifter::shift_buffer`] 处理整段音频。
/// `VoiceProcessor::process` 逐帧调用时内部缓冲，流末尾用 `flush` 取残余。
pub struct PitchShifter {
    fft_size: usize,
    hop: usize,
    shift_ratio: f64,
    // FFT planner 与预规划的变换器。
    r2c: std::sync::Arc<dyn RealToComplex<f64>>,
    c2r: std::sync::Arc<dyn ComplexToReal<f64>>,
    // Hann 窗。
    window: Vec<f64>,
    // 重叠累加缓冲（输出）。
    output_buffer: Vec<f64>,
    // 输入环形缓冲。
    input_buffer: Vec<f64>,
    // 前一帧的频谱相位，用于相位累积。
    prev_phase: Vec<f64>,
    // 输出缓冲写入位置。
    output_pos: usize,
    // 输入缓冲已填充位置。
    input_pos: usize,
    // 总已输出样本数。
    total_output: usize,
    // 总已输入样本数。
    total_input: usize,
    channels: u16,
    sample_rate: u32,
}

impl PitchShifter {
    /// 构造 pitch shifter，`semitones` 正数升调、负数降调。
    ///
    /// # Errors
    /// 参数无效返回 [`DspError`]。
    pub fn new(sample_rate: u32, channels: u16, semitones: f64) -> Result<Self, DspError> {
        if channels == 0 {
            return Err(DspError::InvalidInput("channels must be non-zero".into()));
        }
        if sample_rate == 0 {
            return Err(DspError::InvalidInput(
                "sample rate must be non-zero".into(),
            ));
        }
        let shift_ratio = SEMITONE_RATIO.powf(semitones);

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let c2r = planner.plan_fft_inverse(FFT_SIZE);

        let window = hann_window(FFT_SIZE);

        Ok(Self {
            fft_size: FFT_SIZE,
            hop: HOP,
            shift_ratio,
            r2c,
            c2r,
            window,
            output_buffer: vec![0.0; FFT_SIZE],
            input_buffer: vec![0.0; FFT_SIZE],
            prev_phase: vec![0.0; FFT_SIZE / 2 + 1],
            output_pos: 0,
            input_pos: 0,
            total_output: 0,
            total_input: 0,
            channels,
            sample_rate,
        })
    }

    /// 一次性处理整段单声道交错 f32 音频，返回 pitch-shifted 结果。
    ///
    /// 多声道时按声道分别处理再交错合并。
    ///
    /// # Errors
    /// 处理失败返回 [`DspError`]。
    pub fn shift_buffer(&mut self, samples: &[f32]) -> Result<Vec<f32>, DspError> {
        let ch = self.channels as usize;
        if samples.len() % ch != 0 {
            return Err(DspError::InvalidInput(format!(
                "samples length {} not divisible by channels {}",
                samples.len(),
                ch
            )));
        }
        let n_frames = samples.len() / ch;

        // De-interleave。
        let mut channel_data: Vec<Vec<f32>> = vec![Vec::with_capacity(n_frames); ch];
        for (i, &s) in samples.iter().enumerate() {
            channel_data[i % ch].push(s);
        }

        // 每声道分别 pitch shift：用独立的单声道 shifter 避免状态污染。
        let mut processed: Vec<Vec<f32>> = Vec::with_capacity(ch);
        for chan in channel_data.iter_mut() {
            let mut mono_shifter = PitchShifter::new(self.sample_rate, 1, 0.0)?;
            mono_shifter.shift_ratio = self.shift_ratio;
            processed.push(mono_shifter.shift_mono(&chan[..])?);
        }

        // Re-interleave。
        let max_len = processed.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut out = Vec::with_capacity(max_len.saturating_mul(ch));
        for frame in 0..max_len {
            for chan in processed.iter().take(ch) {
                out.push(chan.get(frame).copied().unwrap_or(0.0));
            }
        }
        Ok(out)
    }

    fn shift_mono(&mut self, samples: &[f32]) -> Result<Vec<f32>, DspError> {
        // 重置流式状态，离线整段处理。
        self.reset();

        let total_in = samples.len();
        let mut result: Vec<f32> = Vec::new();

        // 逐 hop 喂入样本，触发处理。
        for chunk in samples.chunks(self.hop) {
            let frame = Frame {
                samples: chunk.to_vec(),
                sample_rate: self.sample_rate,
                channels: 1,
                timestamp: 0,
            };
            let mut out_frame = Frame::zero(self.sample_rate, 1, 0);
            self.process(&frame, &mut out_frame)
                .map_err(|e| DspError::Compute(e.to_string()))?;
            if !out_frame.samples.is_empty() {
                result.extend_from_slice(&out_frame.samples);
            }
        }

        // Flush 残余：补零填满最后一帧 + 输出延迟的尾部样本。
        let flushed = self.flush().map_err(|e| DspError::Compute(e.to_string()))?;
        result.extend_from_slice(&flushed);

        // Pitch shift 保持时长，但 phase vocoder 有 fft_size 的前向延迟。
        // 输出比输入多出延迟样本，截断到 total_in。
        if result.len() > total_in {
            result.truncate(total_in);
        } else {
            // 若输出不足（极短输入），补零到 total_in。
            result.resize(total_in, 0.0);
        }
        Ok(result)
    }

    /// 取出残余输出（flush 内部缓冲）。
    ///
    /// # Errors
    /// 处理失败返回 [`DspError`]。
    pub fn flush(&mut self) -> Result<Vec<f32>, DspError> {
        if self.input_pos == 0 && self.total_output >= self.total_input {
            return Ok(Vec::new());
        }
        // 用零填满剩余 input_buffer，处理最后一帧。
        while self.input_pos < self.fft_size {
            self.input_buffer[self.input_pos] = 0.0;
            self.input_pos += 1;
            self.total_input += 1;
        }
        let out = self.process_frame()?;
        let result: Vec<f32> = out.iter().map(|&s| s as f32).collect();
        Ok(result)
    }

    fn process_frame(&mut self) -> Result<Vec<f64>, DspError> {
        let n = self.fft_size;
        let half = n / 2 + 1;

        // 加窗 + FFT。
        let mut frame_data: Vec<f64> = self.input_buffer[..n]
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        let mut spectrum = self.r2c.make_output_vec();
        self.r2c
            .process(&mut frame_data, &mut spectrum)
            .map_err(|e| DspError::Compute(format!("fft forward failed: {e}")))?;

        // Phase vocoder：频域 bin 移位 + 相位累积。
        let mut new_spectrum: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); half];
        let omega = 2.0 * std::f64::consts::PI * self.hop as f64 / n as f64;

        for (k, spec_k) in new_spectrum.iter_mut().enumerate() {
            // 目标 bin（按 shift_ratio 移位）。
            let src_k = k as f64 / self.shift_ratio;
            let src_k_low = src_k.floor() as usize;

            // 源 bin 超出频谱范围时置零（降调时高频 bin 无对应源）。
            if src_k_low >= half {
                *spec_k = Complex::new(0.0, 0.0);
                continue;
            }
            let src_k_high = (src_k_low + 1).min(half - 1);
            let frac = src_k - src_k_low as f64;

            if src_k_high >= half {
                *spec_k = Complex::new(0.0, 0.0);
                continue;
            }

            // 线性插值幅度。
            let mag_low = spectrum[src_k_low].norm();
            let mag_high = spectrum[src_k_high].norm();
            let mag = mag_low * (1.0 - frac) + mag_high * frac;

            // 相位：从源 bin 取相位，加上相位累积增量。
            let phase_low = spectrum[src_k_low].arg();
            let phase_high = spectrum[src_k_high].arg();
            let phase = phase_low * (1.0 - frac) + phase_high * frac;

            // 相位累积：期望相位增量 = omega * k（对于 hop 大小的帧移）。
            let expected_phase_adv = omega * k as f64;
            let actual_phase_adv = phase - self.prev_phase[k];
            let mut delta = actual_phase_adv - expected_phase_adv;
            // wrap to [-pi, pi]。
            while delta > std::f64::consts::PI {
                delta -= 2.0 * std::f64::consts::PI;
            }
            while delta < -std::f64::consts::PI {
                delta += 2.0 * std::f64::consts::PI;
            }
            let accumulated_phase = self.prev_phase[k] + expected_phase_adv + delta;
            self.prev_phase[k] = accumulated_phase;

            *spec_k = Complex::from_polar(mag, accumulated_phase);
        }

        // realfft ComplexToRealEven 要求 bin 0 和 bin N/2 的虚部为零
        //（实信号频谱的对称性约束），相位累积可能破坏这一点，强制归零。
        new_spectrum[0].im = 0.0;
        new_spectrum[half - 1].im = 0.0;

        // IFFT。
        let mut time_domain = self.c2r.make_output_vec();
        self.c2r
            .process(&mut new_spectrum, &mut time_domain)
            .map_err(|e| DspError::Compute(format!("fft inverse failed: {e}")))?;

        // 归一化（realfft 不归一化，forward+inverse 需除以 n）。
        let scale = 1.0 / n as f64;
        for s in time_domain.iter_mut() {
            *s *= scale;
        }

        // 加窗（合成窗）+ overlap-add 到输出缓冲。
        for (i, &td) in time_domain.iter().enumerate().take(n) {
            let out_idx = (self.output_pos + i) % self.output_buffer.len();
            self.output_buffer[out_idx] += td * self.window[i];
        }

        // 提取 hop 个样本作为输出。
        let mut result = Vec::with_capacity(self.hop);
        for _ in 0..self.hop {
            let val = self.output_buffer[self.output_pos];
            self.output_buffer[self.output_pos] = 0.0;
            result.push(val);
            self.output_pos = (self.output_pos + 1) % self.output_buffer.len();
            self.total_output += 1;
        }

        // 滑动 input_buffer：移除 hop 个样本。
        self.input_buffer.drain(..self.hop);
        self.input_buffer.resize(n, 0.0);
        self.input_pos = self.input_pos.saturating_sub(self.hop);

        Ok(result)
    }
}

impl VoiceProcessor for PitchShifter {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        if input.channels != self.channels {
            return Err(VoxError::invalid_input(format!(
                "channel mismatch: expected {} got {}",
                self.channels, input.channels
            )));
        }

        // 简化：VoiceProcessor 流式接口按单声道处理。
        // 多声道流式 pitch shift 留待后续，离线用 shift_buffer。
        let ch = self.channels as usize;
        let mut all_out = Vec::new();

        for c in 0..ch {
            // 提取该声道样本。
            let chan_samples: Vec<f64> = input
                .samples
                .iter()
                .enumerate()
                .filter(|(i, _)| i % ch == c)
                .map(|(_, &s)| f64::from(s))
                .collect();

            // 追加到 input_buffer。
            for &s in &chan_samples {
                if self.input_pos < self.fft_size {
                    self.input_buffer[self.input_pos] = s;
                    self.input_pos += 1;
                }
                self.total_input += 1;
            }

            // 每凑满 fft_size 处理一帧。
            let mut chan_out = Vec::new();
            while self.input_pos >= self.fft_size {
                let out = self
                    .process_frame()
                    .map_err(|e| VoxError::audio(e.to_string()))?;
                chan_out.extend(out);
            }
            all_out.push(chan_out);
        }

        // 交错输出（简化：单声道直接输出，多声道取最长补零）。
        if ch == 1 {
            output.samples = all_out.into_iter().flatten().map(|s| s as f32).collect();
        } else {
            let max_len = all_out.iter().map(|v| v.len()).max().unwrap_or(0);
            output.samples = Vec::with_capacity(max_len.saturating_mul(ch));
            for frame in 0..max_len {
                for chan in all_out.iter().take(ch) {
                    output
                        .samples
                        .push(chan.get(frame).copied().unwrap_or(0.0) as f32);
                }
            }
        }
        output.sample_rate = self.sample_rate;
        output.channels = self.channels;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_buffer.fill(0.0);
        self.input_buffer.fill(0.0);
        self.prev_phase.fill(0.0);
        self.output_pos = 0;
        self.input_pos = 0;
        self.total_output = 0;
        self.total_input = 0;
    }
}

/// 生成 Hann 窗。
#[inline]
fn hann_window(n: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
        w.push(val);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_pitch_preserves_duration() {
        // 1 秒 440Hz 正弦波 at 16000 Hz。
        let sr = 16000u32;
        let n = sr as usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut shifter = PitchShifter::new(sr, 1, 5.0).unwrap(); // +5 semitones
        let output = shifter.shift_buffer(&input).unwrap();

        // 时长应保持（±hop 容差）。
        assert!(
            (output.len() as isize - n as isize).abs() <= HOP as isize,
            "duration changed: in {} out {}",
            n,
            output.len()
        );
    }

    #[test]
    fn shifts_frequency_up() {
        let sr = 16000u32;
        let n = sr as usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut shifter = PitchShifter::new(sr, 1, 12.0).unwrap(); // +12 semitones = 2x freq
        let output = shifter.shift_buffer(&input).unwrap();

        // 检测输出主频：+12 semitones 应使 200Hz → 400Hz。
        // 用过零率粗略估计频率。
        let out_f32 = &output[..output.len().min(n)];
        let zcr = count_zero_crossings(out_f32);
        let est_freq = zcr as f32 * sr as f32 / (2.0 * out_f32.len() as f32);

        // 允许 20% 误差（phase vocoder 非精确）。
        assert!(
            est_freq > 320.0 && est_freq < 480.0,
            "expected ~400Hz, got ~{est_freq:.0}Hz"
        );
    }

    #[test]
    fn shifts_frequency_down() {
        let sr = 16000u32;
        let n = sr as usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 400.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut shifter = PitchShifter::new(sr, 1, -12.0).unwrap(); // -12 semitones = 0.5x freq
        let output = shifter.shift_buffer(&input).unwrap();

        let out_f32 = &output[..output.len().min(n)];
        let zcr = count_zero_crossings(out_f32);
        let est_freq = zcr as f32 * sr as f32 / (2.0 * out_f32.len() as f32);

        assert!(
            est_freq > 160.0 && est_freq < 240.0,
            "expected ~200Hz, got ~{est_freq:.0}Hz"
        );
    }

    #[test]
    fn zero_semitones_approximates_passthrough() {
        let sr = 16000u32;
        let n = 4096usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut shifter = PitchShifter::new(sr, 1, 0.0).unwrap();
        let output = shifter.shift_buffer(&input).unwrap();

        // 0 semitones → shift_ratio=1.0，输出应近似输入（幅度可能有窗效应差异）。
        // 检查中间段相关性（跳过边界效应）。
        let start = FFT_SIZE;
        let end = output.len().min(n).saturating_sub(FFT_SIZE / 2);
        if end > start {
            let mut corr = 0.0;
            let mut energy = 0.0;
            for i in start..end {
                corr += input[i] * output[i];
                energy += input[i] * input[i];
            }
            let normalized = if energy > 0.0 { corr / energy } else { 0.0 };
            assert!(
                normalized > 0.8,
                "0-semitone output should correlate with input, got {normalized:.3}"
            );
        }
    }

    #[test]
    fn rejects_zero_channels() {
        assert!(PitchShifter::new(16000, 0, 5.0).is_err());
    }

    #[test]
    fn rejects_zero_sample_rate() {
        assert!(PitchShifter::new(0, 1, 5.0).is_err());
    }

    #[test]
    fn stereo_shift_preserves_channels() {
        let sr = 16000u32;
        let n = 4096;
        let input: Vec<f32> = (0..n * 2).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut shifter = PitchShifter::new(sr, 2, 3.0).unwrap();
        let output = shifter.shift_buffer(&input).unwrap();
        assert!(output.len() % 2 == 0, "stereo output must be even length");
    }

    fn count_zero_crossings(samples: &[f32]) -> usize {
        let mut count = 0;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                count += 1;
            }
        }
        count
    }
}
