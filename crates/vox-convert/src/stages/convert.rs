//! 变声转换 Stage：content features + timbre embedding → converted features。
//!
//! # 流程
//!
//! 1. 接收 `Frame`（音频样本）
//! 2. 持有当前选定的 `Timbre`（音色 embedding）
//! 3. 把 audio samples + timbre embedding 组合成输入张量
//! 4. 调用 Converter ONNX 模型推理
//! 5. 输出 converted features（仍是 `Frame`，但 samples 存放展平的特征向量）
//!
//! # 设计说明
//!
//! - 本 Stage 假设输入已经是 16kHz 单声道（由前置 FeatureStage 或预处理保证）
//! - 输出 `Frame` 的 `samples` 存放展平的 converted features，供下游 VocoderStage 消费
//! - `sample_rate` 字段复用为"特征帧率"标识（下游识别）
//! - 音色可运行时切换（`set_timbre`），线程安全用 `Mutex`（非音频线程调用）
//! - 推理错误降级为静音（输出零向量），不 panic

use std::sync::Mutex;

use vox_core::{Frame, InferenceSession, Tensor, Timbre, VoxError};

use crate::Stage;

/// Converter 模型输入布局。
#[derive(Debug, Clone)]
pub struct ConvertInputLayout {
    /// content features 的形状（不含 batch 维），如 `[T, 768]`。
    pub content_shape: Vec<usize>,
    /// timbre embedding 的形状（不含 batch 维），如 `[256]`。
    pub embedding_shape: Vec<usize>,
}

impl ConvertInputLayout {
    /// 默认布局：content `[T, 768]` + embedding `[256]`。
    pub fn default_hubert() -> Self {
        Self {
            content_shape: vec![/* T */ 0, 768],
            embedding_shape: vec![256],
        }
    }
}

/// 变声转换 Stage：content features + timbre embedding → converted features。
///
/// 持有 Converter 模型的 `InferenceSession` 和当前选定的 `Timbre`。
/// 音色可运行时切换（`set_timbre`）。
pub struct ConvertStage<S: InferenceSession> {
    session: S,
    /// 当前音色（可运行时切换）。
    timbre: Mutex<Option<Timbre>>,
    /// 输入布局描述。
    layout: ConvertInputLayout,
    /// 预分配的输出 buffer（复用，避免每帧 alloc）。
    output_buffer: Mutex<Vec<f32>>,
}

impl<S: InferenceSession> ConvertStage<S> {
    /// 构造 ConvertStage。
    ///
    /// 初始无音色，需调用 `set_timbre` 后才能正常处理。
    pub fn new(session: S, layout: ConvertInputLayout) -> Self {
        Self {
            session,
            timbre: Mutex::new(None),
            layout,
            output_buffer: Mutex::new(Vec::new()),
        }
    }

    /// 设置当前音色（运行时切换）。
    ///
    /// 传入 `None` 清除音色（Stage 将输出静音）。
    pub fn set_timbre(&self, timbre: Option<Timbre>) {
        *self.timbre.lock().expect("timbre mutex poisoned") = timbre;
    }

    /// 获取当前音色的 ID（如有）。
    pub fn current_timbre_id(&self) -> Option<vox_core::TimbreId> {
        self.timbre
            .lock()
            .expect("timbre mutex poisoned")
            .as_ref()
            .map(|t| t.id)
    }

    /// 构造 Converter 模型的输入张量。
    ///
    /// 输入 0: content features `[1, T, feat_dim]`（从 audio samples 展平重塑）
    /// 输入 1: timbre embedding `[1, embed_dim]`
    fn build_inputs(&self, audio: &[f32], timbre: &Timbre) -> Vec<Tensor> {
        // content features: 把 audio samples 作为展平的 content features。
        // 实际部署中，这里应接收 FeatureStage 的输出而非原始 audio。
        // 为 M7 框架，我们假设 input Frame.samples 已是展平的 content features。
        let feat_dim = self.layout.content_shape.get(1).copied().unwrap_or(768);
        let t_len = if audio.len() % feat_dim == 0 {
            audio.len() / feat_dim
        } else {
            audio.len()
        };

        let content_tensor = Tensor::f32(audio.to_vec(), vec![1, t_len, feat_dim]);

        let embedding_tensor = Tensor::f32(timbre.embedding.to_vec(), {
            let mut shape = vec![1];
            shape.extend_from_slice(&self.layout.embedding_shape);
            shape
        });

        vec![content_tensor, embedding_tensor]
    }

    /// 静音输出（错误降级）。
    fn silent_output(&self, input: &Frame) -> Frame {
        Frame {
            samples: vec![0.0; input.samples.len()],
            sample_rate: input.sample_rate,
            channels: input.channels,
            timestamp: input.timestamp,
        }
    }
}

impl<S: InferenceSession> Stage for ConvertStage<S> {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        // 获取当前音色。
        let timbre_guard = self.timbre.lock().expect("timbre mutex poisoned");
        let timbre = match timbre_guard.as_ref() {
            Some(t) => t,
            None => {
                // 无音色 → 静音输出。
                *output = self.silent_output(input);
                tracing::warn!("convert stage has no timbre, outputting silence");
                return Ok(());
            }
        };

        // 构造输入张量。
        let inputs = self.build_inputs(&input.samples, timbre);
        drop(timbre_guard); // 释放锁，避免推理期间持锁。

        // 推理。
        let outputs = match self.session.run(&inputs) {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "convert inference failed, outputting silence");
                *output = self.silent_output(input);
                return Ok(());
            }
        };

        // 取第一个输出张量作为 converted features。
        let converted = match outputs.into_iter().next() {
            Some(t) => t,
            None => {
                tracing::error!("convert session returned no output");
                *output = self.silent_output(input);
                return Ok(());
            }
        };

        // 复用 output buffer。
        let mut buf = self
            .output_buffer
            .lock()
            .expect("output buffer mutex poisoned");
        buf.clear();
        let converted_data = converted
            .as_f32()
            .ok_or_else(|| vox_core::VoxError::infer("convert output is not f32".to_string()))?;
        buf.extend_from_slice(converted_data);

        // 输出 Frame：samples = 展平的 converted features。
        output.samples = buf.clone();
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;

        Ok(())
    }

    fn reset(&mut self) {
        let mut buf = self
            .output_buffer
            .lock()
            .expect("output buffer mutex poisoned");
        buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::TimbreId;
    use vox_infer::{MockSession, MockStrategy};

    fn make_timbre(id: u64, embedding: Vec<f32>) -> Timbre {
        Timbre {
            id: TimbreId::new(id),
            name: format!("test-{id}"),
            embedding: embedding.into_boxed_slice(),
            f0_offset_semitones: 0.0,
            tags: vec![],
        }
    }

    fn make_frame(samples: Vec<f32>) -> Frame {
        Frame {
            samples,
            sample_rate: 16000,
            channels: 1,
            timestamp: 0,
        }
    }

    #[test]
    fn no_timbre_outputs_silence() {
        let mut stage = ConvertStage::new(
            MockSession::identity(),
            ConvertInputLayout::default_hubert(),
        );
        let input = make_frame(vec![0.5; 768]);
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert!(
            output.samples.iter().all(|&s| s == 0.0),
            "should be silence"
        );
    }

    #[test]
    fn with_timbre_runs_inference() {
        // Mock: 返回固定 shape 的输出。
        let mut stage = ConvertStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 10, 256],
                value: 0.42,
            }),
            ConvertInputLayout::default_hubert(),
        );
        stage.set_timbre(Some(make_timbre(1, vec![0.1; 256])));

        let input = make_frame(vec![0.5; 768]); // 1 frame * 768 features
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        assert_eq!(output.samples.len(), 10 * 256, "should be 10*256 features");
        assert!(output.samples.iter().all(|&s| (s - 0.42).abs() < 1e-6));
    }

    #[test]
    fn inference_error_outputs_silence() {
        // MockSession 总是返回 Ok，无法直接模拟推理错误。
        // 用空输出模拟"无输出"场景。
        let mut stage = ConvertStage::new(
            MockSession::new(MockStrategy::Custom(|_| vec![])),
            ConvertInputLayout::default_hubert(),
        );
        stage.set_timbre(Some(make_timbre(1, vec![0.1; 256])));

        let input = make_frame(vec![0.5; 768]);
        let mut output = Frame::zero(16000, 1, 0);

        stage.process(&input, &mut output).unwrap();
        // 无输出 → 降级为静音。
        assert!(output.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn set_timbre_switches_at_runtime() {
        let mut stage = ConvertStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 5, 128],
                value: 1.0,
            }),
            ConvertInputLayout::default_hubert(),
        );

        // 无音色 → 静音。
        let input = make_frame(vec![0.5; 768]);
        let mut output = Frame::zero(16000, 1, 0);
        stage.process(&input, &mut output).unwrap();
        assert!(output.samples.iter().all(|&s| s == 0.0));

        // 设置音色 → 有输出。
        stage.set_timbre(Some(make_timbre(1, vec![0.1; 256])));
        stage.process(&input, &mut output).unwrap();
        assert!(output.samples.iter().all(|&s| (s - 1.0).abs() < 1e-6));

        // 清除音色 → 静音。
        stage.set_timbre(None);
        stage.process(&input, &mut output).unwrap();
        assert!(output.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn current_timbre_id_tracks_selection() {
        let stage = ConvertStage::new(
            MockSession::identity(),
            ConvertInputLayout::default_hubert(),
        );
        assert!(stage.current_timbre_id().is_none());

        stage.set_timbre(Some(make_timbre(42, vec![0.1; 256])));
        assert_eq!(stage.current_timbre_id(), Some(TimbreId::new(42)));

        stage.set_timbre(None);
        assert!(stage.current_timbre_id().is_none());
    }

    #[test]
    fn reset_clears_output_buffer() {
        let mut stage = ConvertStage::new(
            MockSession::new(MockStrategy::Constant {
                shape: vec![1, 5, 128],
                value: 1.0,
            }),
            ConvertInputLayout::default_hubert(),
        );
        stage.set_timbre(Some(make_timbre(1, vec![0.1; 256])));

        let input = make_frame(vec![0.5; 768]);
        let mut output = Frame::zero(16000, 1, 0);
        stage.process(&input, &mut output).unwrap();
        assert!(!output.samples.is_empty());

        // reset 不应 panic。
        stage.reset();
    }

    #[test]
    fn build_inputs_shapes() {
        let stage = ConvertStage::new(
            MockSession::identity(),
            ConvertInputLayout::default_hubert(),
        );
        let timbre = make_timbre(1, vec![0.1; 256]);
        let inputs = stage.build_inputs(&[0.5; 768], &timbre);

        assert_eq!(inputs.len(), 2);
        // content: [1, T, 768]
        assert_eq!(inputs[0].shape, vec![1, 1, 768]);
        // embedding: [1, 256]
        assert_eq!(inputs[1].shape, vec![1, 256]);
    }
}
