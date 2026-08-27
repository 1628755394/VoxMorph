//! RVC 管线形状常量与特征张量操作。
//!
//! 这些常量和操作是 RVC 变声管线的核心数学基础，借鉴 vc-rs 的 `shape.rs`
//! 和 `feature.rs`，适配 VoxMorph 的类型。
//!
//! # 关键常量
//!
//! - `EMBEDDER_SAMPLE_RATE`: ContentVec 输入采样率 = 16kHz
//! - `RVC_SAMPLE_RATE`: RVC 模型输出采样率 = 48kHz（默认，可由模型元数据覆盖）
//! - `CONTENTVEC_CONTEXT_ALIGN_SAMPLES`: ContentVec 上下文对齐 = 320 样本（20ms@16kHz）
//! - `RMVPE_FRAME_SAMPLES_16K`: RMVPE 每帧 = 160 样本（10ms@16kHz）

/// ContentVec（HuBERT）输入采样率：16 kHz。
pub const EMBEDDER_SAMPLE_RATE: u32 = 16_000;

/// RVC 模型默认输出采样率：48 kHz（可由模型元数据覆盖）。
pub const RVC_SAMPLE_RATE: u32 = 48_000;

/// ContentVec 上下文对齐样本数：320 = 20ms @ 16kHz。
pub const CONTENTVEC_CONTEXT_ALIGN_SAMPLES: usize = 320;

/// RMVPE 每帧样本数：160 = 10ms @ 16kHz。
pub const RMVPE_FRAME_SAMPLES_16K: usize = 160;

/// RMVPE bucket 帧数（mel2hidden 对齐）。
pub const RMVPE_BUCKET_FRAMES: usize = 32;

/// RMVPE 上下文保护帧数。
pub const RMVPE_GUARD_FRAMES: usize = 5;

/// 毫秒 → 样本数转换。
#[inline]
pub fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

/// 从样本数计算 ContentVec 特征帧数（10ms hop @ 16kHz → frames = samples * 100 / sr）。
#[inline]
pub fn feature_len_for_samples(samples: usize, sample_rate: u32) -> usize {
    (samples as u64 * 100 / sample_rate as u64) as usize
}

/// 舍入模式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    Floor,
    Ceil,
}

/// 向上取整除法（兼容 MSRV 1.75，不用 `div_ceil`）。
#[inline]
fn ceil_div(numerator: u64, denominator: u64) -> u64 {
    let q = numerator / denominator;
    if numerator % denominator != 0 {
        q + 1
    } else {
        q
    }
}

/// 跨采样率样本数转换：`samples` 在 `from_sr` 下的时长 → `to_sr` 下的样本数。
#[inline]
pub fn samples_between_rates(
    samples: usize,
    from_sr: u32,
    to_sr: u32,
    rounding: Rounding,
) -> usize {
    let numerator = samples as u64 * to_sr as u64;
    let denominator = from_sr as u64;
    match rounding {
        Rounding::Floor => (numerator / denominator) as usize,
        Rounding::Ceil => ceil_div(numerator, denominator) as usize,
    }
}

/// 向上对齐到 `align` 的倍数。
#[inline]
pub fn align_up(value: usize, align: usize) -> usize {
    if align == 0 || value % align == 0 {
        value
    } else {
        value + (align - value % align)
    }
}

/// 保留 `values` 的末尾 `len` 个元素（原地 drain 前部）。
pub fn keep_tail_in_place<T>(values: &mut Vec<T>, len: usize) {
    if values.len() > len {
        values.drain(..values.len() - len);
    }
}

/// RMVPE 模型输入样本数（16kHz）：10ms hop + bucket 对齐 + guard 帧。
pub fn rmvpe_model_input_samples_16k(chunk_samples: usize, sample_rate: u32) -> usize {
    let chunk_frames = ceil_div(chunk_samples as u64 * 100, sample_rate as u64) as usize;
    let required_frames = chunk_frames.saturating_add(RMVPE_GUARD_FRAMES).max(1);
    let bucket_frames = align_up(required_frames, RMVPE_BUCKET_FRAMES);
    bucket_frames.saturating_sub(1) * RMVPE_FRAME_SAMPLES_16K
}

/// ONNX silence front feature frames：extra_convert 对应的前置静音帧数。
pub fn onnx_silence_front_feature_frames(
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    let extra_16k_samples = (extra_convert_samples as u64 * EMBEDDER_SAMPLE_RATE as u64
        / rvc_sample_rate as u64) as usize;
    (extra_16k_samples / 360) * 2
}

/// 特征张量：ContentVec 输出 `[1, frames, channels]`，可复用缓冲。
#[derive(Default, Debug)]
pub struct FeatureTensor {
    /// 展平的特征数据（`frames * channels` 个 f32）。
    pub data: Vec<f32>,
    /// 形状 `[1, frames, channels]`。
    pub shape: Vec<i64>,
}

impl FeatureTensor {
    /// 重复每帧 `factor` 次（RVC 的 2x 上采样惯例）。
    ///
    /// 原地操作，复用已分配缓冲。从后向前遍历避免覆盖未复制帧。
    ///
    /// # Errors
    /// 非 rank-3 或 batch != 1 返回错误。
    pub fn repeat_frames(&mut self, factor: usize) -> Result<(), String> {
        if factor <= 1 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            return Err("feature tensor must be rank-3 [1, frames, channels]".into());
        }
        let batch = usize::try_from(self.shape[0]).map_err(|_| "invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).map_err(|_| "invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).map_err(|_| "invalid feature channels")?;
        if batch != 1 {
            return Err(format!("feature batch must be 1, got {batch}"));
        }

        let old_len = self.data.len();
        self.data.resize(old_len * factor, 0.0);
        for frame in (0..frames).rev() {
            let src = frame * channels;
            for repeat in (0..factor).rev() {
                let dst = (frame * factor + repeat) * channels;
                self.data.copy_within(src..src + channels, dst);
            }
        }
        self.shape[1] = (frames * factor) as i64;
        Ok(())
    }

    /// 裁剪前 `frames_to_drop` 帧。
    ///
    /// # Errors
    /// 非 rank-3 或 batch != 1 返回错误。
    pub fn trim_front_frames(&mut self, frames_to_drop: usize) -> Result<(), String> {
        if frames_to_drop == 0 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            return Err("feature tensor must be rank-3 [1, frames, channels]".into());
        }
        let batch = usize::try_from(self.shape[0]).map_err(|_| "invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).map_err(|_| "invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).map_err(|_| "invalid feature channels")?;
        if batch != 1 {
            return Err(format!("feature batch must be 1, got {batch}"));
        }
        if frames_to_drop >= frames {
            self.data.clear();
            self.shape[1] = 0;
            return Ok(());
        }
        let sample_offset = frames_to_drop * channels;
        self.data.drain(..sample_offset);
        self.shape[1] = (frames - frames_to_drop) as i64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_to_samples_basic() {
        assert_eq!(ms_to_samples(16000, 10), 160);
        assert_eq!(ms_to_samples(48000, 100), 4800);
    }

    #[test]
    fn feature_len_for_samples_16k() {
        // 320 samples @ 16kHz = 20ms → 2 frames (10ms hop)
        assert_eq!(feature_len_for_samples(320, 16000), 2);
        // 1600 samples @ 16kHz = 100ms → 10 frames
        assert_eq!(feature_len_for_samples(1600, 16000), 10);
    }

    #[test]
    fn samples_between_rates_downsample() {
        // 48000 samples @ 48kHz → 16000 samples @ 16kHz
        assert_eq!(
            samples_between_rates(48000, 48000, 16000, Rounding::Floor),
            16000
        );
    }

    #[test]
    fn samples_between_rates_upsample_ceil() {
        // 160 samples @ 16kHz → 480 samples @ 48kHz (exact)
        assert_eq!(
            samples_between_rates(160, 16000, 48000, Rounding::Ceil),
            480
        );
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(100, 320), 320);
        assert_eq!(align_up(320, 320), 320);
        assert_eq!(align_up(640, 320), 640);
        assert_eq!(align_up(321, 320), 640);
    }

    #[test]
    fn align_up_zero_align_is_noop() {
        assert_eq!(align_up(100, 0), 100);
    }

    #[test]
    fn keep_tail_in_place_trims_front() {
        let mut v = vec![1, 2, 3, 4, 5];
        keep_tail_in_place(&mut v, 3);
        assert_eq!(v, vec![3, 4, 5]);
    }

    #[test]
    fn keep_tail_in_place_shorter_is_noop() {
        let mut v = vec![1, 2];
        keep_tail_in_place(&mut v, 5);
        assert_eq!(v, vec![1, 2]);
    }

    #[test]
    fn rmvpe_input_samples_basic() {
        // 1600 samples @ 16kHz = 100ms = 10 frames
        // required = 10 + 5 guard = 15, bucket = 32
        // samples = (32 - 1) * 160 = 4960
        let s = rmvpe_model_input_samples_16k(1600, 16000);
        assert_eq!(s, 31 * 160);
    }

    #[test]
    fn repeat_frames_duplicates_each_frame() {
        let mut t = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![1, 3, 2],
        };
        t.repeat_frames(2).unwrap();
        assert_eq!(
            t.data,
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(t.shape, vec![1, 6, 2]);
    }

    #[test]
    fn repeat_frames_factor_one_is_noop() {
        let mut t = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
        };
        t.repeat_frames(1).unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape, vec![1, 2, 2]);
    }

    #[test]
    fn trim_front_frames_removes_front() {
        let mut t = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![1, 3, 2],
        };
        t.trim_front_frames(1).unwrap();
        assert_eq!(t.data, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape, vec![1, 2, 2]);
    }

    #[test]
    fn trim_front_all_clears() {
        let mut t = FeatureTensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2],
        };
        t.trim_front_frames(5).unwrap();
        assert!(t.data.is_empty());
        assert_eq!(t.shape[1], 0);
    }

    #[test]
    fn repeat_frames_wrong_rank_errors() {
        let mut t = FeatureTensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
        };
        assert!(t.repeat_frames(2).is_err());
    }
}
