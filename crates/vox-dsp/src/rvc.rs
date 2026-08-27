//! RVC 管线 DSP 工具：SOLA 搜索、coarse pitch 量化、RMS、noise gate。
//!
//! 这些函数是 RVC 变声管线的纯 DSP 组件，不依赖 ONNX 推理。
//! 借鉴 vc-rs 的实现，适配 VoxMorph 的类型和规范。
//!
//! # 性能约束
//!
//! - 内层循环 `#[inline]`（`opt-inline-small`）
//! - 浮点比较用容差（`num-float-compare`）
//! - 整数算术用 `saturating_*`（`num-overflow-explicit`）

/// 计算 RMS（均方根）能量。
///
/// `samples` 为单声道样本。空输入返回 0.0。
#[inline]
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .iter()
        .map(|&s| {
            let d = f64::from(s);
            d * d
        })
        .sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// 归一化互相关：衡量 `a` 和 `b` 在等长窗口内的相似度。
///
/// 返回值范围 [-1.0, 1.0]。`a` 和 `b` 必须等长，空输入返回 0.0。
#[inline]
pub fn normalized_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..a.len() {
        let av = f64::from(a[i]);
        let bv = f64::from(b[i]);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (dot / denom) as f32
}

/// SOLA 偏移搜索：在 `candidate` 的前 `max_offset + crossfade_len` 样本中，
/// 找到与 `reference`（长度 = `crossfade_len`）互相关最大的偏移。
///
/// 返回 `[0, max_offset]` 范围内的最佳偏移。`threshold` 为最小能量阈值，
/// 低于此值跳过搜索（避免静音段误匹配）。
pub fn sola_offset(
    candidate: &[f32],
    reference: &[f32],
    max_offset: usize,
    threshold: f32,
) -> usize {
    let crossfade_len = reference.len();
    if crossfade_len == 0 || max_offset == 0 {
        return 0;
    }
    let candidate_len = (crossfade_len + max_offset).min(candidate.len());
    if candidate_len < crossfade_len {
        return 0;
    }

    let mut best_offset = 0usize;
    let mut best_corr = f32::NEG_INFINITY;
    for offset in 0..=max_offset {
        let end = offset + crossfade_len;
        if end > candidate_len {
            break;
        }
        let window = &candidate[offset..end];
        // 跳过低能量窗口。
        let energy = rms(window);
        if energy < threshold {
            continue;
        }
        let corr = normalized_correlation(window, reference);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }
    best_offset
}

/// 带 threshold 的 SOLA 偏移搜索（与 vc-rs `sola_offset_with_threshold` 对齐）。
///
/// 当所有窗口能量都低于 `threshold` 时返回 0（不搜索）。
#[inline]
pub fn sola_offset_with_threshold(
    candidate: &[f32],
    reference: &[f32],
    max_offset: usize,
    threshold: f32,
) -> usize {
    sola_offset(candidate, reference, max_offset, threshold)
}

/// VCClient 风格的前置强度衰减：对 reference 做 `1.0 - i/len` 的线性衰减。
///
/// 用于 SOLA 交叉淡化时的加权参考信号。
pub fn vcclient_prev_strength_into(reference: &[f32], output: &mut Vec<f32>) {
    output.clear();
    let len = reference.len();
    if len == 0 {
        return;
    }
    output.reserve(len);
    for (i, &v) in reference.iter().enumerate() {
        let strength = 1.0 - (i as f32 / len as f32);
        output.push(v * strength);
    }
}

/// VCClient 风格交叉淡化：将 `output[..len]` 与 `reference` 做线性交叉淡化。
///
/// `output` 的前 `len` 样本被原地修改：`out[i] = out[i] * (i/len) + ref[i] * (1 - i/len)`。
pub fn vcclient_crossfade(reference: &[f32], output: &mut [f32]) {
    let len = reference.len().min(output.len());
    if len == 0 {
        return;
    }
    for i in 0..len {
        let ratio = i as f32 / len as f32;
        output[i] = output[i] * ratio + reference[i] * (1.0 - ratio);
    }
}

/// 将连续 F0（Hz）量化为 RVC coarse pitch bin（1..=255）。
///
/// 使用 mel 刻度映射：`mel = 1127 * ln(1 + f0/700)`，
/// 然后线性映射到 `[1, 255]`。`f0 <= 0`（无声）映射为 1。
pub fn coarse_pitch(pitchf: &[f32]) -> Vec<i64> {
    const F0_MIN: f32 = 50.0;
    const F0_MAX: f32 = 1100.0;
    let mel_min = 1127.0 * (1.0 + F0_MIN / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + F0_MAX / 700.0).ln();
    let mut out = Vec::with_capacity(pitchf.len());
    for &f0 in pitchf {
        let mel = if f0 > 0.0 {
            1127.0 * (1.0 + f0 / 700.0).ln()
        } else {
            0.0
        };
        let coarse = if mel > 0.0 {
            (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0
        } else {
            1.0
        };
        out.push(coarse.clamp(1.0, 255.0).round() as i64);
    }
    out
}

/// 简单 noise gate：低于阈值的样本衰减到 floor 倍数。
///
/// 带 attack/release 平滑，避免咔哒声。`process_in_place` 原地处理。
pub struct NoiseGate {
    threshold: f32,
    floor: f32,
    // 当前增益（0..=1），平滑过渡。
    current_gain: f32,
    attack_coef: f32,
    release_coef: f32,
}

impl NoiseGate {
    /// 构造 noise gate。
    pub fn new(
        sample_rate: f32,
        threshold: f32,
        attack_ms: f32,
        release_ms: f32,
        floor: f32,
    ) -> Self {
        let attack_coef = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };
        let release_coef = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };
        Self {
            threshold,
            floor,
            current_gain: 1.0,
            attack_coef,
            release_coef,
        }
    }

    /// 原地处理音频样本。
    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let abs_s = s.abs();
            let target_gain = if abs_s < self.threshold {
                self.floor
            } else {
                1.0
            };
            // 平滑过渡：attack（gain 下降）用 attack_coef，release（gain 上升）用 release_coef。
            if target_gain < self.current_gain {
                self.current_gain =
                    self.current_gain * self.attack_coef + target_gain * (1.0 - self.attack_coef);
            } else {
                self.current_gain =
                    self.current_gain * self.release_coef + target_gain * (1.0 - self.release_coef);
            }
            *s *= self.current_gain;
        }
    }

    /// 重置内部状态。
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

/// 取 `audio` 的末尾 `len` 样本（不足则取全部）。
#[inline]
pub(crate) fn tail_slice(audio: &[f32], len: usize) -> &[f32] {
    let start = audio.len().saturating_sub(len);
    &audio[start..]
}

/// 将 `audio` 的末尾部分（或全部）填入 `output`，不足时 pad 零。
pub(crate) fn last_or_pad_into(audio: &[f32], target_len: usize, output: &mut Vec<f32>) {
    output.clear();
    if audio.len() >= target_len {
        output.extend_from_slice(&audio[audio.len() - target_len..]);
    } else {
        output.resize(target_len - audio.len(), 0.0);
        output.extend_from_slice(audio);
    }
}

/// Pad `values` 到 `len` 长度（末尾补零）。
pub(crate) fn pad_to_len_in_place(values: &mut Vec<f32>, len: usize) {
    if values.len() < len {
        values.resize(len, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(rms(&[0.0; 100]), 0.0);
    }

    #[test]
    fn rms_of_constant_signal() {
        // RMS of [0.5; N] = 0.5
        let r = rms(&[0.5; 1000]);
        assert!((r - 0.5).abs() < 1e-5, "expected 0.5, got {r}");
    }

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn normalized_correlation_identical_signals() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        let corr = normalized_correlation(&a, &a);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "identical signals should have corr 1.0, got {corr}"
        );
    }

    #[test]
    fn normalized_correlation_anticorrelated() {
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = vec![-1.0, 1.0, -1.0, 1.0];
        let corr = normalized_correlation(&a, &b);
        assert!(
            (corr - (-1.0)).abs() < 1e-5,
            "anticorrelated should have corr -1.0, got {corr}"
        );
    }

    #[test]
    fn normalized_correlation_empty_returns_zero() {
        assert_eq!(normalized_correlation(&[], &[]), 0.0);
    }

    #[test]
    fn sola_offset_finds_best_alignment() {
        // reference = sine wave chunk
        let reference: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        // candidate = [noise..] + reference + [more..], best offset should be where reference starts
        let mut candidate = vec![0.0; 32]; // 32 samples of silence before
        candidate.extend_from_slice(&reference);
        candidate.extend_from_slice(&[0.0; 32]);

        let offset = sola_offset(&candidate, &reference, 64, 1e-4);
        assert_eq!(
            offset, 32,
            "should find offset at 32 where reference starts"
        );
    }

    #[test]
    fn sola_offset_zero_max_offset_returns_zero() {
        let reference = vec![0.5; 32];
        let candidate = vec![0.5; 64];
        assert_eq!(sola_offset(&candidate, &reference, 0, 1e-4), 0);
    }

    #[test]
    fn coarse_pitch_unvoiced_maps_to_one() {
        let pitchf = vec![0.0, 0.0, 0.0];
        let coarse = coarse_pitch(&pitchf);
        assert_eq!(coarse, vec![1, 1, 1]);
    }

    #[test]
    fn coarse_pitch_voiced_in_range() {
        // 200Hz should map to some bin in [1, 255]
        let pitchf = vec![200.0];
        let coarse = coarse_pitch(&pitchf);
        assert_eq!(coarse.len(), 1);
        assert!(
            (1..=255).contains(&coarse[0]),
            "coarse pitch should be in [1,255], got {}",
            coarse[0]
        );
    }

    #[test]
    fn coarse_pitch_high_frequency() {
        let pitchf = vec![1000.0];
        let coarse = coarse_pitch(&pitchf);
        assert!(
            coarse[0] > 200,
            "high freq should map to high bin, got {}",
            coarse[0]
        );
    }

    #[test]
    fn noise_gate_attenuates_below_threshold() {
        let mut gate = NoiseGate::new(16000.0, 0.1, 5.0, 50.0, 0.0);
        let mut buf = vec![0.01; 100]; // well below threshold 0.1
        gate.process_in_place(&mut buf);
        // After processing, samples should be attenuated (gain < 1)
        // With attack smoothing, first few samples may not be fully attenuated
        let avg: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
        assert!(
            avg < 0.01,
            "attenuated signal should be much smaller, avg={avg}"
        );
    }

    #[test]
    fn noise_gate_passes_above_threshold() {
        let mut gate = NoiseGate::new(16000.0, 0.1, 5.0, 50.0, 0.0);
        let mut buf = vec![0.5; 100]; // well above threshold
        gate.process_in_place(&mut buf);
        let avg: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
        assert!(
            (avg - 0.5).abs() < 0.1,
            "passed signal should be near original, avg={avg}"
        );
    }

    #[test]
    fn vcclient_crossfade_blends_signals() {
        let reference = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0, 0.0, 0.0, 0.0];
        vcclient_crossfade(&reference, &mut output);
        // At i=0: out = 0*0 + 1*1 = 1.0
        // At i=3: out = 0*0.75 + 1*0.25 = 0.25
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "start should be reference, got {}",
            output[0]
        );
        assert!(
            (output[3] - 0.25).abs() < 1e-5,
            "end should be mostly output, got {}",
            output[3]
        );
    }

    #[test]
    fn vcclient_prev_strength_decreasing() {
        let reference = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = Vec::new();
        vcclient_prev_strength_into(&reference, &mut output);
        assert_eq!(output.len(), 4);
        // strength at i=0: 1.0, at i=3: 0.25
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[3] - 0.25).abs() < 1e-5);
    }
}
