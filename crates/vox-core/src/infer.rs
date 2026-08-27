//! 推理后端抽象。
//!
//! `ort::Session` 在 `vox-infer` 中实现 [`InferenceSession`]，避免把
//! `ort` 类型泄露到 `vox-convert`（解耦推理后端，`test-mock-traits`）。

use crate::VoxError;

/// 张量数据类型。
///
/// ONNX 模型支持多种 dtype（float32、int64 等），RVC 模型的 pitch/sid
/// 输入需要 int64。用枚举统一表示，避免泄露 `ort` 类型。
#[derive(Debug, Clone)]
pub enum TensorData {
    /// float32 数据（最常见，ContentVec/RMVPE/RVC 特征和音频）。
    F32(Vec<f32>),
    /// int64 数据（RVC pitch bins、speaker ID）。
    I64(Vec<i64>),
}

impl TensorData {
    /// 元素总数。
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 尝试获取 f32 切片引用。
    #[inline]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// 尝试获取 i64 切片引用。
    #[inline]
    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            TensorData::I64(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

impl From<Vec<f32>> for TensorData {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        TensorData::F32(v)
    }
}

impl From<Vec<i64>> for TensorData {
    #[inline]
    fn from(v: Vec<i64>) -> Self {
        TensorData::I64(v)
    }
}

/// 张量：推理输入/输出的最小表示。
///
/// 用 owned `Vec` 而非借用，以便跨线程在 channel 中传递。
/// 形状以行优先（C-contiguous）解释。
#[derive(Debug, Clone)]
pub struct Tensor {
    /// 元素数据，行优先展开。
    pub data: TensorData,
    /// 各维度大小，如 `[batch, time, feat]`。
    pub shape: Vec<usize>,
}

impl Tensor {
    /// 构造一个 f32 张量，预分配 `shape` 乘积容量（`mem-with-capacity`）。
    #[inline]
    pub fn new(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: TensorData::F32(Vec::with_capacity(len)),
            shape,
        }
    }

    /// 构造一个 f32 张量，直接传入数据。
    #[inline]
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::F32(data),
            shape,
        }
    }

    /// 构造一个 i64 张量，直接传入数据。
    #[inline]
    pub fn i64(data: Vec<i64>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::I64(data),
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

    /// 尝试获取 f32 数据切片。
    #[inline]
    pub fn as_f32(&self) -> Option<&[f32]> {
        self.data.as_f32()
    }

    /// 尝试获取 i64 数据切片。
    #[inline]
    pub fn as_i64(&self) -> Option<&[i64]> {
        self.data.as_i64()
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
