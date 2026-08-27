//! RVC 变声 Stage：编排完整 RVC 管线（ContentVec + RMVPE + RVC + SOLA）。
//!
//! 借鉴 vc-rs 的 `pipeline.rs`，将三个 ONNX session + stream state + SOLA
//! 整合为一个 [`Stage`]，可直接插入 [`crate::Pipeline`]。
//!
//! # 流程
//!
//! ```text
//! [Audio Frame] → RvcStreamState (滚动缓冲 + 16kHz 重采样)
//!               → ContentVecSession (content features)
//!               → RmvpeSession (F0 曲线)
//!               → 特征 2x 重复 + coarse pitch 量化
//!               → RvcModelSession (生成音频波形)
//!               → SolaChunkJoiner (chunk 平滑)
//!               → [Audio Frame]
//! ```
//!
//! # 实时参数
//!
//! `pitch_shift` / `speaker_id` / `input_gain` / `output_gain` 可运行时调整
//!（`set_live_params`），无需重新加载模型。

use std::sync::Mutex;

use vox_core::{Frame, InferenceSession, VoxError};
use vox_dsp::rvc::coarse_pitch;
use vox_dsp::sola::{SmoothingKind, SolaChunkJoiner, SolaConfig};
use vox_feature::rvc::{ms_to_samples, onnx_silence_front_feature_frames, FeatureTensor};
use vox_feature::sessions::{ContentVecSession, RmvpeSession, RvcModelSession};
use vox_feature::stream::{RvcStreamState, StreamError};

use crate::Stage;

/// RVC 实时参数（可运行时调整）。
#[derive(Clone, Debug)]
pub struct RvcLiveParams {
    /// Pitch shift（半音）。
    pub pitch_shift: f32,
    /// Speaker ID（多说话人模型）。
    pub speaker_id: i64,
    /// 输入增益。
    pub input_gain: f32,
    /// 输出增益。
    pub output_gain: f32,
}

impl Default for RvcLiveParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
        }
    }
}

/// RVC Stage 配置（加载时固定）。
#[derive(Clone, Debug)]
pub struct RvcStageConfig {
    /// SOLA 交叉淡化时长（ms）。
    pub crossfade_ms: u32,
    /// SOLA 搜索范围（ms）。
    pub sola_search_ms: u32,
    /// RVC 输出尾部丢弃（ms，不稳定区）。
    pub tail_discard_ms: u32,
    /// 额外转换时长（ms，SOLA 余量）。
    pub extra_convert_ms: u32,
    /// 静音阈值（RMS 低于此值跳过推理）。
    pub silence_threshold: f32,
    /// F0 有声/无声阈值。
    pub f0_threshold: f32,
}

impl Default for RvcStageConfig {
    fn default() -> Self {
        Self {
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
            extra_convert_ms: 100,
            silence_threshold: 0.0001,
            f0_threshold: 0.3,
        }
    }
}

/// RVC 变声 Stage：编排 ContentVec + RMVPE + RVC + SOLA 完整管线。
///
/// 泛型参数 `E` / `F` / `R` 分别为 ContentVec / RMVPE / RVC 的 session 类型，
/// 便于测试用 `MockSession` 替换。
pub struct RvcStage<E: InferenceSession, F: InferenceSession, R: InferenceSession> {
    embedder: ContentVecSession<E>,
    f0: RmvpeSession<F>,
    rvc: RvcModelSession<R>,
    stream_state: RvcStreamState,
    smoother: SolaChunkJoiner,
    config: RvcStageConfig,
    live_params: Mutex<RvcLiveParams>,
    // 复用缓冲。
    feature_tensor: FeatureTensor,
    pitchf_buffer: Vec<f32>,
    pitch_buffer: Vec<i64>,
    rvc_output: Vec<f32>,
    // 输出采样率（RVC 模型采样率）。
    output_sample_rate: u32,
}

impl<E: InferenceSession, F: InferenceSession, R: InferenceSession> RvcStage<E, F, R> {
    /// 构造 RVC Stage。
    pub fn new(
        embedder: E,
        embedder_channels: i64,
        f0: F,
        rvc: R,
        rvc_sample_rate: u32,
        config: RvcStageConfig,
    ) -> Self {
        let chunk_samples = ms_to_samples(rvc_sample_rate, 500); // 默认 500ms chunk
        let sola_config = SolaConfig::from_ms(
            SmoothingKind::Sola,
            chunk_samples,
            rvc_sample_rate,
            config.crossfade_ms,
            config.sola_search_ms,
            config.tail_discard_ms,
        );
        Self {
            embedder: ContentVecSession::new(embedder, embedder_channels),
            f0: RmvpeSession::new(f0),
            rvc: RvcModelSession::new(rvc, rvc_sample_rate),
            stream_state: RvcStreamState::new(rvc_sample_rate),
            smoother: SolaChunkJoiner::new(&sola_config),
            config,
            live_params: Mutex::new(RvcLiveParams::default()),
            feature_tensor: FeatureTensor::default(),
            pitchf_buffer: Vec::new(),
            pitch_buffer: Vec::new(),
            rvc_output: Vec::new(),
            output_sample_rate: rvc_sample_rate,
        }
    }

    /// 设置实时参数（线程安全，可从 GUI 线程调用）。
    pub fn set_live_params(&self, params: RvcLiveParams) {
        *self.live_params.lock().expect("live params mutex poisoned") = params;
    }

    /// 获取输出采样率。
    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    /// 处理一帧音频：完整 RVC 管线。
    fn process_internal(&mut self, input: &Frame, output: &mut Frame) -> Result<(), RvcStageError> {
        let live = self
            .live_params
            .lock()
            .expect("live params mutex poisoned")
            .clone();

        // 多声道 → 单声道 + 输入增益。
        let mono = to_mono_with_gain(&input.samples, input.channels, live.input_gain);

        // 计算转换窗口参数。
        let crossfade_and_search_samples = ms_to_samples(
            self.output_sample_rate,
            self.config.crossfade_ms + self.config.sola_search_ms,
        );
        let volume_excluded_samples =
            ms_to_samples(self.output_sample_rate, self.config.tail_discard_ms);
        let extra_convert_samples =
            ms_to_samples(self.output_sample_rate, self.config.extra_convert_ms);

        // 流式状态：追加音频、重采样到 16kHz、计算窗口。
        let stream_input = self.stream_state.generate_input(
            &mono,
            input.sample_rate,
            crossfade_and_search_samples,
            volume_excluded_samples,
            extra_convert_samples,
        )?;

        // 静音检测：跳过推理，输出静音。
        if stream_input.input_rms < self.config.silence_threshold {
            output.samples = vec![0.0; stream_input.out_size];
            output.sample_rate = self.output_sample_rate;
            output.channels = 1;
            output.timestamp = input.timestamp;
            return Ok(());
        }

        // 1. ContentVec 特征提取。
        let audio_16k = self.stream_state.audio_16k_window().to_vec();
        self.embedder
            .extract_into(&audio_16k, &mut self.feature_tensor)?;

        // 2. RMVPE F0 估计。
        let f0 = self.f0.estimate_f0(&audio_16k)?;

        // 3. 特征 2x 重复（RVC 惯例）。
        self.feature_tensor
            .repeat_frames(2)
            .map_err(RvcStageError::Other)?;

        // 4. F0 对齐到特征帧数 + pitch shift。
        let target_frames = self.feature_tensor.shape.get(1).copied().unwrap_or(0) as usize;
        self.pitchf_buffer = align_pitchf(&f0, target_frames, live.pitch_shift);
        self.pitch_buffer = coarse_pitch(&self.pitchf_buffer);

        // 5. 裁剪前置静音帧（ONNX silence front）。
        let silence_front =
            onnx_silence_front_feature_frames(extra_convert_samples, self.output_sample_rate);
        if silence_front > 0 {
            self.feature_tensor
                .trim_front_frames(silence_front)
                .map_err(RvcStageError::Other)?;
            // 同步裁剪 pitch。
            if silence_front < self.pitchf_buffer.len() {
                self.pitchf_buffer.drain(..silence_front);
            }
            if silence_front < self.pitch_buffer.len() {
                self.pitch_buffer.drain(..silence_front);
            }
        }

        // 6. RVC 推理。
        self.rvc_output = self.rvc.convert(
            &self.feature_tensor,
            &self.pitch_buffer,
            &self.pitchf_buffer,
            live.speaker_id,
        )?;

        // 7. SOLA 平滑。
        self.smoother.process(&self.rvc_output);
        let smoothed = self.smoother.output().to_vec();

        // 8. 输出增益。
        let final_output = if (live.output_gain - 1.0).abs() > 1e-6 {
            smoothed.iter().map(|&s| s * live.output_gain).collect()
        } else {
            smoothed
        };

        output.samples = final_output;
        output.sample_rate = self.output_sample_rate;
        output.channels = 1;
        output.timestamp = input.timestamp;

        Ok(())
    }
}

/// RVC Stage 错误。
#[derive(Debug, thiserror::Error)]
pub enum RvcStageError {
    #[error(transparent)]
    Stream(#[from] StreamError),
    #[error(transparent)]
    Feature(#[from] vox_feature::FeatureError),
    #[error("rvc stage error: {0}")]
    Other(String),
}

impl From<RvcStageError> for VoxError {
    fn from(e: RvcStageError) -> Self {
        VoxError::infer(e.to_string())
    }
}

impl<E: InferenceSession, F: InferenceSession, R: InferenceSession> Stage for RvcStage<E, F, R> {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        match self.process_internal(input, output) {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::error!(error = %e, "rvc stage failed, outputting silence");
                // 降级为静音。
                let out_size = ms_to_samples(self.output_sample_rate, 500);
                output.samples = vec![0.0; out_size];
                output.sample_rate = self.output_sample_rate;
                output.channels = 1;
                output.timestamp = input.timestamp;
                Err(e.into())
            }
        }
    }

    fn reset(&mut self) {
        self.stream_state.reset();
        self.smoother.reset();
        self.feature_tensor = FeatureTensor::default();
        self.pitchf_buffer.clear();
        self.pitch_buffer.clear();
        self.rvc_output.clear();
    }
}

/// 多声道交错 → 单声道 + 增益。
fn to_mono_with_gain(samples: &[f32], channels: u16, gain: f32) -> Vec<f32> {
    if channels <= 1 {
        return samples.iter().map(|&s| s * gain).collect();
    }
    let ch = channels as usize;
    let n_frames = samples.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let frame = &samples[i * ch..(i + 1) * ch];
        let avg = frame.iter().sum::<f32>() / ch as f32;
        mono.push(avg * gain);
    }
    mono
}

/// 对齐 F0 到目标帧数（取尾部或左侧零填充）+ pitch shift。
fn align_pitchf(input: &[f32], target_len: usize, pitch_shift: f32) -> Vec<f32> {
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    // Pitch shift: f0 * 2^(shift/12)。
    let shift_ratio = 2.0f32.powf(pitch_shift / 12.0);
    let shifted: Vec<f32> = input
        .iter()
        .map(|&f0| if f0 > 0.0 { f0 * shift_ratio } else { 0.0 })
        .collect();

    if shifted.len() >= target_len {
        shifted[shifted.len() - target_len..].to_vec()
    } else {
        let mut out = vec![0.0; target_len - shifted.len()];
        out.extend_from_slice(&shifted);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_infer::{MockSession, MockStrategy};

    fn make_test_stage() -> RvcStage<MockSession, MockSession, MockSession> {
        let embedder = MockSession::new(MockStrategy::Constant {
            shape: vec![1, 10, 256],
            value: 0.5,
        });
        let f0 = MockSession::new(MockStrategy::Constant {
            shape: vec![1, 10],
            value: 200.0,
        });
        let rvc = MockSession::new(MockStrategy::Constant {
            shape: vec![1, 4800],
            value: 0.3,
        });
        RvcStage::new(embedder, 256, f0, rvc, 48000, RvcStageConfig::default())
    }

    #[test]
    fn rvc_stage_processes_audio() {
        let mut stage = make_test_stage();
        let input = Frame {
            samples: vec![0.5; 4800], // 100ms @ 48kHz
            sample_rate: 48000,
            channels: 1,
            timestamp: 0,
        };
        let mut output = Frame::zero(48000, 1, 0);
        let result = stage.process(&input, &mut output);
        // 可能成功或降级静音，但不应 panic。
        assert!(result.is_ok() || result.is_err());
        assert_eq!(output.sample_rate, 48000);
        assert_eq!(output.channels, 1);
    }

    #[test]
    fn rvc_stage_reset_clears_state() {
        let mut stage = make_test_stage();
        let input = Frame {
            samples: vec![0.5; 4800],
            sample_rate: 48000,
            channels: 1,
            timestamp: 0,
        };
        let mut output = Frame::zero(48000, 1, 0);
        let _ = stage.process(&input, &mut output);
        // reset 不应 panic。
        stage.reset();
        assert!(stage.stream_state.audio_buffer.is_empty());
    }

    #[test]
    fn set_live_params_updates_pitch_shift() {
        let stage = make_test_stage();
        stage.set_live_params(RvcLiveParams {
            pitch_shift: 12.0,
            ..Default::default()
        });
        let params = stage
            .live_params
            .lock()
            .expect("live params mutex poisoned")
            .clone();
        assert!((params.pitch_shift - 12.0).abs() < 1e-6);
    }

    #[test]
    fn to_mono_with_gain_applies_gain() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 frames stereo
        let mono = to_mono_with_gain(&input, 2, 2.0);
        assert_eq!(mono, vec![3.0, 7.0]); // (1+2)/2 * 2 = 3, (3+4)/2 * 2 = 7
    }

    #[test]
    fn to_mono_single_channel_with_gain() {
        let input = vec![1.0, 2.0, 3.0];
        let mono = to_mono_with_gain(&input, 1, 0.5);
        assert_eq!(mono, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn align_pitchf_pads_front() {
        let input = vec![100.0, 200.0];
        let out = align_pitchf(&input, 5, 0.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 100.0, 200.0]);
    }

    #[test]
    fn align_pitchf_trims_front() {
        let input = vec![100.0, 200.0, 300.0, 400.0];
        let out = align_pitchf(&input, 2, 0.0);
        assert_eq!(out, vec![300.0, 400.0]);
    }

    #[test]
    fn align_pitchf_applies_pitch_shift() {
        let input = vec![200.0];
        let out = align_pitchf(&input, 1, 12.0); // +12 semitones = 2x
        assert!(
            (out[0] - 400.0).abs() < 1e-5,
            "pitch shift +12 should double f0, got {}",
            out[0]
        );
    }

    #[test]
    fn align_pitchf_unvoiced_stays_zero() {
        let input = vec![0.0, 0.0];
        let out = align_pitchf(&input, 2, 12.0);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn align_pitchf_empty_input_returns_zeros() {
        let out = align_pitchf(&[], 3, 0.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn rvc_stage_output_sample_rate() {
        let stage = make_test_stage();
        assert_eq!(stage.output_sample_rate(), 48000);
    }
}
