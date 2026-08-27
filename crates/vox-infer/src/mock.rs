//! Mock 推理会话：用于测试，不依赖真实 ONNX 模型（`test-mock-traits`）。
//!
//! `MockSession` 实现 [`InferenceSession`]，按预设规则生成输出张量，
//! 便于 `vox-feature` / `vox-convert` 在 CI 中测试推理编排逻辑。

use std::sync::Mutex;

use vox_core::{InferenceSession, Tensor, TensorData, VoxError};

use crate::InferError;

/// Mock 推理策略。
#[derive(Debug, Clone)]
pub enum MockStrategy {
    /// 原样返回输入（identity）。
    Identity,
    /// 返回固定形状的零张量。
    Zeros { shape: Vec<usize> },
    /// 返回固定形状的常量张量。
    Constant { shape: Vec<usize>, value: f32 },
    /// 自定义闭包（输入 → 输出）。
    Custom(fn(&[Tensor]) -> Vec<Tensor>),
}

/// Mock 推理会话，实现 [`InferenceSession`]。
///
/// 用于测试 `vox-feature` / `vox-convert` 的推理编排，无需真实 ONNX 模型。
pub struct MockSession {
    strategy: Mutex<MockStrategy>,
}

impl MockSession {
    /// 构造 MockSession，使用指定策略。
    pub fn new(strategy: MockStrategy) -> Self {
        Self {
            strategy: Mutex::new(strategy),
        }
    }

    /// 构造 identity MockSession（原样返回输入）。
    pub fn identity() -> Self {
        Self::new(MockStrategy::Identity)
    }

    /// 构造 zeros MockSession。
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(MockStrategy::Zeros { shape })
    }

    /// 构造 constant MockSession。
    pub fn constant(shape: Vec<usize>, value: f32) -> Self {
        Self::new(MockStrategy::Constant { shape, value })
    }
}

impl Default for MockSession {
    fn default() -> Self {
        Self::identity()
    }
}

impl InferenceSession for MockSession {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, VoxError> {
        let strategy = self
            .strategy
            .lock()
            .map_err(|e| VoxError::infer(format!("mock strategy lock failed: {e}")))?
            .clone();

        let outputs = match strategy {
            MockStrategy::Identity => inputs.to_vec(),
            MockStrategy::Zeros { shape } => {
                let len = shape.iter().product();
                vec![Tensor {
                    data: TensorData::F32(vec![0.0; len]),
                    shape,
                }]
            }
            MockStrategy::Constant { shape, value } => {
                let len = shape.iter().product();
                vec![Tensor {
                    data: TensorData::F32(vec![value; len]),
                    shape,
                }]
            }
            MockStrategy::Custom(f) => f(inputs),
        };

        Ok(outputs)
    }
}

/// 将 [`InferError`] 转为 [`vox_core::VoxError`]（供调用方 `?` 传播）。
impl From<InferError> for VoxError {
    fn from(e: InferError) -> Self {
        VoxError::infer(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_returns_inputs() {
        let mut session = MockSession::identity();
        let input = Tensor::f32(vec![1.0, 2.0, 3.0], vec![3]);
        let outputs = session.run(std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_f32().unwrap(), input.as_f32().unwrap());
        assert_eq!(outputs[0].shape, input.shape);
    }

    #[test]
    fn zeros_returns_zero_tensor() {
        let mut session = MockSession::zeros(vec![2, 3]);
        let input = Tensor::f32(vec![1.0; 6], vec![2, 3]);
        let outputs = session.run(&[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![2, 3]);
        assert!(outputs[0].as_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_returns_constant_tensor() {
        let mut session = MockSession::constant(vec![4], 0.5);
        let input = Tensor::f32(vec![1.0; 4], vec![4]);
        let outputs = session.run(&[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![4]);
        assert!(outputs[0]
            .as_f32()
            .unwrap()
            .iter()
            .all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn custom_strategy_applies_closure() {
        let double = |inputs: &[Tensor]| -> Vec<Tensor> {
            inputs
                .iter()
                .map(|t| Tensor {
                    data: TensorData::F32(t.as_f32().unwrap().iter().map(|&v| v * 2.0).collect()),
                    shape: t.shape.clone(),
                })
                .collect()
        };
        let mut session = MockSession::new(MockStrategy::Custom(double));
        let input = Tensor::f32(vec![1.0, 2.0, 3.0], vec![3]);
        let outputs = session.run(&[input]).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn default_is_identity() {
        let mut session = MockSession::default();
        let input = Tensor::f32(vec![42.0], vec![1]);
        let outputs = session.run(std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), vec![42.0]);
    }

    #[test]
    fn empty_inputs_identity_returns_empty() {
        let mut session = MockSession::identity();
        let outputs = session.run(&[]).unwrap();
        assert!(outputs.is_empty());
    }
}
