//! 推理后端抽象。
//!
//! `ort::Session` 在 `vox-infer` 中实现 [`InferenceSession`]，避免把
//! `ort` 类型泄露到 `vox-convert`（解耦推理后端，`test-mock-traits`）。

use crate::VoxError;

/// 张量：推理输入/输出的最小表示。
///
/// 用 owned `Vec` 而非借用，以便跨线程在 channel 中传递。
/// 形状以行优先（C-contiguous）解释。
#[derive(Debug, Clone)]
pub struct Tensor {
    /// 元素数据，行优先展开。
    pub data: Vec<f32>,
    /// 各维度大小，如 `[batch, time, feat]`。
    pub shape: Vec<usize>,
}

impl Tensor {
    /// 构造一个张量，预分配 `shape` 乘积容量（`mem-with-capacity`）。
    #[inline]
    pub fn new(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: Vec::with_capacity(len),
            shape,
        }
    }

    /// 元素总数 = `shape` 乘积。
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// 是否为空（无元素）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 推理会话抽象。
///
/// 实现者负责加载 ONNX 模型、选择 ExecutionProvider、复用输入/输出缓冲。
/// 推理耗时建议用 `tracing::span!(Level::TRACE, "infer")` 包裹。
///
/// # Errors
/// 形状不匹配返回 [`VoxError::ShapeMismatch`]，后端错误返回 [`VoxError::Infer`]。
pub trait InferenceSession: Send {
    /// 执行一次推理。
    ///
    /// `inputs` / 返回值的所有权模型允许实现者复用内部缓冲（`mem-reuse-collections`）。
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, VoxError>;
}
