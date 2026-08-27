//! SOLA（Similarity Overlap-Add）chunk 平滑：消除 chunk 边界不连续。
//!
//! 借鉴 vc-rs 的 `sola.rs`，适配 VoxMorph 的类型。
//!
//! # 原理
//!
//! RVC 每个 chunk 独立生成，边界处可能有数样本偏移，直接拼接会产生
//! 咔哒声。SOLA 在额外输出范围内搜索与上一 chunk 尾部最相似的偏移，
//! 在该位置做交叉淡化，输出长度保持固定。
//!
//! # PSOLA
//!
//! 当输出有稳定 F0 时，优先选择 pitch 周期边界附近的偏移（PSOLA），
//! 避免在母音持续段切到波形周期中间导致不稳定。

use crate::rvc::{
    last_or_pad_into, normalized_correlation, pad_to_len_in_place, rms, sola_offset, tail_slice,
    vcclient_crossfade, vcclient_prev_strength_into,
};

/// 平滑类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingKind {
    /// 通用 SOLA（相似度搜索）。
    Sola,
    /// Pitch-Synchronous SOLA（pitch 周期对齐）。
    Psola,
}

/// chunk join 诊断信息（仅用于 telemetry，不影响音频）。
#[derive(Clone, Copy, Debug, Default)]
pub struct JoinDiagnostics {
    /// 使用的平滑类型（None = 首个 chunk，无 join）。
    pub kind: Option<SmoothingKind>,
    /// 选择的 SOLA 偏移（样本数）。
    pub sola_offset: usize,
    /// 最大允许偏移。
    pub max_offset: usize,
    /// 选择偏移处的归一化互相关。
    pub correlation: f32,
    /// 实际交叉淡化长度。
    pub crossfade_len: usize,
    /// PSOLA pitch 周期（样本数），有声且稳定时才有。
    pub pitch_period: Option<usize>,
    /// PSOLA 回退到 SOLA。
    pub psola_fallback: bool,
}

/// SOLA chunk joiner 配置。
#[derive(Clone, Debug)]
pub struct SolaConfig {
    pub kind: SmoothingKind,
    /// 输出 chunk 样本数。
    pub chunk_samples: usize,
    /// 采样率（Hz），用于 PSOLA pitch 周期转换。
    pub sample_rate: u32,
    /// 交叉淡化样本数。
    pub crossfade_samples: usize,
    /// SOLA 搜索范围样本数。
    pub sola_search_samples: usize,
    /// 尾部丢弃样本数（RVC 输出末尾不稳定区）。
    pub tail_discard_samples: usize,
}

impl SolaConfig {
    /// 从毫秒构造配置。
    pub fn from_ms(
        kind: SmoothingKind,
        chunk_samples: usize,
        sample_rate: u32,
        crossfade_ms: u32,
        sola_search_ms: u32,
        tail_discard_ms: u32,
    ) -> Self {
        let to_samples = |ms: u32| ((sample_rate as u64 * ms as u64) / 1000) as usize;
        Self {
            kind,
            chunk_samples,
            sample_rate,
            crossfade_samples: to_samples(crossfade_ms),
            sola_search_samples: to_samples(sola_search_ms),
            tail_discard_samples: to_samples(tail_discard_ms),
        }
    }
}

/// SOLA chunk joiner：在 chunk 间做相似度搜索 + 交叉淡化。
pub struct SolaChunkJoiner {
    chunk_samples: usize,
    sample_rate: u32,
    kind: SmoothingKind,
    crossfade_samples: usize,
    sola_search_samples: usize,
    tail_discard_samples: usize,
    /// 上一 chunk 尾部（交叉淡化参考）。
    sola_buffer: Vec<f32>,
    /// 加权后的参考信号（复用缓冲）。
    weighted_reference: Vec<f32>,
    /// 输出缓冲（复用，避免每 chunk alloc）。
    output_buffer: Vec<f32>,
    /// 最近一次 join 诊断。
    last_diagnostics: JoinDiagnostics,
}

impl SolaChunkJoiner {
    /// 构造 SOLA joiner。
    pub fn new(config: &SolaConfig) -> Self {
        Self {
            chunk_samples: config.chunk_samples,
            sample_rate: config.sample_rate,
            kind: config.kind,
            crossfade_samples: config.crossfade_samples,
            sola_search_samples: config.sola_search_samples,
            tail_discard_samples: config.tail_discard_samples,
            sola_buffer: Vec::new(),
            weighted_reference: Vec::new(),
            output_buffer: Vec::new(),
            last_diagnostics: JoinDiagnostics::default(),
        }
    }

    /// 初始化（prime）：用首个 chunk 的尾部填充 sola_buffer。
    pub fn prime(&mut self, audio: &[f32]) {
        let audio = self.candidate_audio(audio);
        if self.crossfade_samples == 0 || audio.is_empty() {
            self.sola_buffer.clear();
            return;
        }
        self.sola_buffer.clear();
        self.sola_buffer
            .extend_from_slice(tail_slice(audio, self.crossfade_samples));
    }

    /// 处理一个 chunk：与历史做 SOLA join，输出到 `output_buffer`。
    ///
    /// 返回选择的 SOLA 偏移。用 [`Self::output`] 读取结果。
    pub fn process(&mut self, audio: &[f32]) -> usize {
        self.process_with_kind(audio, None, self.kind())
    }

    /// 处理一个 chunk（带 F0，用于 PSOLA）。
    pub fn process_with_pitchf(&mut self, audio: &[f32], pitchf: &[f32]) -> usize {
        let kind = self.kind();
        self.process_with_kind(audio, Some(pitchf), kind)
    }

    fn kind(&self) -> SmoothingKind {
        self.kind
    }

    fn process_with_kind(
        &mut self,
        audio: &[f32],
        pitchf: Option<&[f32]>,
        kind: SmoothingKind,
    ) -> usize {
        let target_len = self.chunk_samples.max(1);
        let audio = self.candidate_audio(audio);

        if self.crossfade_samples == 0 || audio.is_empty() {
            last_or_pad_into(audio, target_len, &mut self.output_buffer);
            self.sola_buffer.clear();
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
        }

        if self.sola_buffer.is_empty() {
            self.prime(audio);
            self.output_buffer.clear();
            self.output_buffer.resize(target_len, 0.0);
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
        }

        let crossfade_len = self
            .sola_buffer
            .len()
            .min(self.crossfade_samples)
            .min(audio.len())
            .min(target_len);
        if crossfade_len == 0 {
            last_or_pad_into(audio, target_len, &mut self.output_buffer);
            self.update_sola_buffer(audio, 0);
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
        }

        let max_offset = self
            .sola_search_samples
            .min(audio.len().saturating_sub(target_len));
        let candidate_len = (crossfade_len + max_offset).min(audio.len());
        let reference = &self.sola_buffer[self.sola_buffer.len() - crossfade_len..];
        vcclient_prev_strength_into(reference, &mut self.weighted_reference);
        let weighted_reference = self.weighted_reference.as_slice();

        // PSOLA：如果有稳定 F0，优先 pitch 周期边界。
        let (sola_offset, psola_fallback, pitch_period) = if kind == SmoothingKind::Psola {
            if let Some(pitchf) = pitchf {
                if let Some(period) = stable_pitch_period_samples(pitchf, self.sample_rate) {
                    let psola_offset = psola_offset_select(
                        &audio[..candidate_len],
                        weighted_reference,
                        max_offset,
                        period,
                    );
                    if let Some(off) = psola_offset {
                        (off, false, Some(period))
                    } else {
                        // PSOLA 无合适偏移 → 回退 SOLA。
                        let off = sola_offset(
                            &audio[..candidate_len],
                            weighted_reference,
                            max_offset,
                            1e-4,
                        );
                        (off, true, Some(period))
                    }
                } else {
                    // 无稳定 F0 → 纯 SOLA。
                    let off = sola_offset(
                        &audio[..candidate_len],
                        weighted_reference,
                        max_offset,
                        1e-4,
                    );
                    (off, true, None)
                }
            } else {
                // 无 F0 数据 → 纯 SOLA。
                let off = sola_offset(
                    &audio[..candidate_len],
                    weighted_reference,
                    max_offset,
                    1e-4,
                );
                (off, true, None)
            }
        } else {
            let off = sola_offset(
                &audio[..candidate_len],
                weighted_reference,
                max_offset,
                1e-4,
            );
            (off, false, None)
        };

        let sola_offset = sola_offset.min(max_offset);
        let correlation = normalized_correlation(
            &audio[sola_offset..sola_offset + crossfade_len],
            weighted_reference,
        );

        let output_end = sola_offset.saturating_add(target_len).min(audio.len());
        let output = &mut self.output_buffer;
        output.clear();
        output.extend_from_slice(&audio[sola_offset..output_end]);
        pad_to_len_in_place(output, target_len);
        output.truncate(target_len);
        vcclient_crossfade(reference, &mut output[..crossfade_len]);
        self.update_sola_buffer(audio, sola_offset);

        self.last_diagnostics = JoinDiagnostics {
            kind: Some(kind),
            sola_offset,
            max_offset,
            correlation,
            crossfade_len,
            pitch_period,
            psola_fallback,
        };

        sola_offset
    }

    /// 获取最近一次 join 的输出。
    pub fn output(&self) -> &[f32] {
        &self.output_buffer
    }

    /// 获取最近一次 join 的诊断。
    pub fn last_diagnostics(&self) -> &JoinDiagnostics {
        &self.last_diagnostics
    }

    /// 重置状态（passthrough 切换或采样率变化时调用）。
    pub fn reset(&mut self) {
        self.sola_buffer.clear();
        self.output_buffer.clear();
        self.weighted_reference.clear();
        self.last_diagnostics = JoinDiagnostics::default();
    }

    /// 候选音频：丢弃不稳定尾部，取搜索窗口。
    fn candidate_audio<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        let stable_len = audio.len().saturating_sub(self.tail_discard_samples);
        let audio = &audio[..stable_len];
        let window_len = self
            .chunk_samples
            .max(1)
            .saturating_add(self.crossfade_samples)
            .saturating_add(self.sola_search_samples);
        if audio.len() > window_len {
            &audio[audio.len() - window_len..]
        } else {
            audio
        }
    }

    fn update_sola_buffer(&mut self, audio: &[f32], sola_offset: usize) {
        if self.crossfade_samples == 0 {
            self.sola_buffer.clear();
            return;
        }
        let candidate = if sola_offset < self.sola_search_samples {
            let start = audio
                .len()
                .saturating_sub(self.sola_search_samples + self.crossfade_samples - sola_offset);
            let end = audio
                .len()
                .saturating_sub(self.sola_search_samples - sola_offset);
            if start < end && end <= audio.len() {
                &audio[start..end]
            } else {
                tail_slice(audio, self.crossfade_samples)
            }
        } else {
            tail_slice(audio, self.crossfade_samples)
        };
        self.sola_buffer.clear();
        self.sola_buffer.extend_from_slice(candidate);
    }
}

/// PSOLA 常量。
const PSOLA_MIN_F0_HZ: f32 = 50.0;
const PSOLA_MAX_F0_HZ: f32 = 1_100.0;
const PSOLA_MAX_RELATIVE_F0_STDDEV: f32 = 0.20;
const PSOLA_MIN_RMS: f32 = 1e-4;

/// 估计稳定 pitch 周期（样本数）。
///
/// 当 F0 稳定（标准差 < 20% 均值）且有声时返回周期 = `sample_rate / mean_f0`。
fn stable_pitch_period_samples(pitchf: &[f32], sample_rate: u32) -> Option<usize> {
    if pitchf.is_empty() || sample_rate == 0 {
        return None;
    }
    let voiced: Vec<f32> = pitchf.iter().copied().filter(|&f| f > 0.0).collect();
    if voiced.is_empty() {
        return None;
    }
    let voiced_ratio = voiced.len() as f32 / pitchf.len() as f32;
    if voiced_ratio < 0.5 {
        return None;
    }
    // 均值。
    let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
    if !(PSOLA_MIN_F0_HZ..=PSOLA_MAX_F0_HZ).contains(&mean) {
        return None;
    }
    // 标准差。
    let variance: f32 =
        voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / voiced.len() as f32;
    let stddev = variance.sqrt();
    if stddev / mean > PSOLA_MAX_RELATIVE_F0_STDDEV {
        return None;
    }
    // 周期 = sample_rate / f0（四舍五入到整数样本）。
    let period = (sample_rate as f32 / mean).round() as usize;
    if period == 0 {
        return None;
    }
    Some(period)
}

/// PSOLA 偏移选择：优先 pitch 周期边界。
fn psola_offset_select(
    candidate: &[f32],
    reference: &[f32],
    max_offset: usize,
    period: usize,
) -> Option<usize> {
    if period == 0 || max_offset == 0 {
        return None;
    }
    // 在 pitch 周期倍数处搜索。
    let mut best_offset = None;
    let mut best_corr = f32::NEG_INFINITY;
    let crossfade_len = reference.len();
    let mut offset = 0;
    while offset <= max_offset {
        let end = offset + crossfade_len;
        if end > candidate.len() {
            break;
        }
        let window = &candidate[offset..end];
        if rms(window) < PSOLA_MIN_RMS {
            offset += period;
            continue;
        }
        let corr = normalized_correlation(window, reference);
        if corr > best_corr {
            best_corr = corr;
            best_offset = Some(offset);
        }
        offset += period;
    }
    best_offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sola_joiner_first_chunk_primes() {
        let config = SolaConfig {
            kind: SmoothingKind::Sola,
            chunk_samples: 100,
            sample_rate: 48000,
            crossfade_samples: 20,
            sola_search_samples: 10,
            tail_discard_samples: 5,
        };
        let mut joiner = SolaChunkJoiner::new(&config);
        let audio = vec![0.5; 200];
        let offset = joiner.process(&audio);
        assert_eq!(offset, 0, "first chunk should prime, not join");
        assert_eq!(joiner.output().len(), 100);
    }

    #[test]
    fn sola_joiner_finds_best_alignment() {
        let config = SolaConfig {
            kind: SmoothingKind::Sola,
            chunk_samples: 64,
            sample_rate: 48000,
            crossfade_samples: 32,
            sola_search_samples: 16,
            tail_discard_samples: 0,
        };
        let mut joiner = SolaChunkJoiner::new(&config);

        // 第一个 chunk：正弦波。
        let sine: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        joiner.process(&sine);

        // 第二个 chunk：同样的正弦波但前面有偏移。
        let mut chunk2 = vec![0.0; 16]; // 16 样本偏移
        chunk2.extend_from_slice(&sine);
        chunk2.extend_from_slice(&[0.0; 64]);

        let offset = joiner.process(&chunk2);
        // 应该找到 ~16 的偏移。
        assert!(
            offset > 0,
            "should find non-zero offset for misaligned chunk, got {offset}"
        );
        assert_eq!(
            joiner.output().len(),
            64,
            "output should be chunk_samples long"
        );
    }

    #[test]
    fn sola_joiner_reset_clears_state() {
        let config = SolaConfig {
            kind: SmoothingKind::Sola,
            chunk_samples: 100,
            sample_rate: 48000,
            crossfade_samples: 20,
            sola_search_samples: 10,
            tail_discard_samples: 5,
        };
        let mut joiner = SolaChunkJoiner::new(&config);
        joiner.prime(&[0.5; 200]);
        assert!(!joiner.sola_buffer.is_empty());
        joiner.reset();
        assert!(joiner.sola_buffer.is_empty());
    }

    #[test]
    fn sola_config_from_ms() {
        let config = SolaConfig::from_ms(SmoothingKind::Sola, 4800, 48000, 85, 12, 10);
        assert_eq!(config.chunk_samples, 4800);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.crossfade_samples, 4080); // 85ms * 48 = 4080
        assert_eq!(config.sola_search_samples, 576); // 12ms * 48 = 576
        assert_eq!(config.tail_discard_samples, 480); // 10ms * 48 = 480
    }

    #[test]
    fn stable_pitch_period_returns_samples() {
        // 200Hz @ 48000 → 240 samples
        let pitchf = vec![200.0; 100];
        let period = stable_pitch_period_samples(&pitchf, 48000);
        assert_eq!(period, Some(240));
    }

    #[test]
    fn stable_pitch_period_unvoiced_returns_none() {
        let pitchf = vec![0.0; 100];
        assert_eq!(stable_pitch_period_samples(&pitchf, 48000), None);
    }

    #[test]
    fn stable_pitch_period_unstable_returns_none() {
        // F0 跳变：100Hz 和 500Hz 交替，标准差大
        let pitchf: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 100.0 } else { 500.0 })
            .collect();
        assert_eq!(stable_pitch_period_samples(&pitchf, 48000), None);
    }

    #[test]
    fn stable_pitch_period_low_voiced_ratio_returns_none() {
        // 只有 30% 有声
        let mut pitchf = vec![0.0; 70];
        pitchf.extend(vec![200.0; 30]);
        assert_eq!(stable_pitch_period_samples(&pitchf, 48000), None);
    }

    #[test]
    fn stable_pitch_period_empty_returns_none() {
        assert_eq!(stable_pitch_period_samples(&[], 48000), None);
    }

    #[test]
    fn stable_pitch_period_zero_sample_rate_returns_none() {
        let pitchf = vec![200.0; 100];
        assert_eq!(stable_pitch_period_samples(&pitchf, 0), None);
    }

    #[test]
    fn psola_joiner_uses_pitch_period() {
        // PSOLA 配置：chunk=64, crossfade=32, search=16
        let config = SolaConfig {
            kind: SmoothingKind::Psola,
            chunk_samples: 64,
            sample_rate: 48000,
            crossfade_samples: 32,
            sola_search_samples: 16,
            tail_discard_samples: 0,
        };
        let mut joiner = SolaChunkJoiner::new(&config);

        // 第一个 chunk：正弦波（200Hz @ 48kHz → 周期 240 样本）
        let sine: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 200.0 / 48000.0).sin())
            .collect();
        let pitchf = [200.0f32; 10];
        joiner.process_with_pitchf(&sine, &pitchf);

        // 第二个 chunk：同样的正弦波但前面有偏移
        let mut chunk2 = vec![0.0; 16];
        chunk2.extend_from_slice(&sine);
        chunk2.extend_from_slice(&[0.0; 64]);

        let offset = joiner.process_with_pitchf(&chunk2, &pitchf);
        // PSOLA 应该不 panic 并返回有效偏移
        let _ = offset;
        assert_eq!(joiner.output().len(), 64);
        let diag = joiner.last_diagnostics();
        assert_eq!(diag.kind, Some(SmoothingKind::Psola));
    }

    #[test]
    fn tail_slice_returns_last_n() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(tail_slice(&audio, 3), &[3.0, 4.0, 5.0]);
        assert_eq!(tail_slice(&audio, 10), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn last_or_pad_into_pads_front() {
        let audio = vec![1.0, 2.0, 3.0];
        let mut output = Vec::new();
        last_or_pad_into(&audio, 5, &mut output);
        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn last_or_pad_into_trims_front() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = Vec::new();
        last_or_pad_into(&audio, 3, &mut output);
        assert_eq!(output, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn diagnostics_default_is_empty() {
        let d = JoinDiagnostics::default();
        assert!(d.kind.is_none());
        assert_eq!(d.sola_offset, 0);
    }
}
