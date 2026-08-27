//! ONNX Runtime session 封装：实现 [`vox_core::InferenceSession`]。
//!
//! # EP 自动选择
//!
//! 按优先级 CUDA > DirectML > CoreML > CPU 注册 EP。每平台只启用可用 EP，
//! 未启用 EP 用 `tracing::warn!` 记录而非 panic（规范要求）。
//!
//! # Session 复用
//!
//! `ort::Session` 内部用 `Arc` 共享，**禁止**每次推理新建 session。
//! `OrtSession` 持有 `ort::Session`，通过 `&mut self` 的 `run` 方法推理。
//!
//! # 张量转换
//!
//! `vox_core::Tensor`（`Vec<f32>` + `shape`）↔ `ort::value::Tensor`：
//! - 输入：用 `Tensor::from_array(([shape], data.into_boxed_slice()))` 构造
//! - 输出：用 `try_extract_tensor::<f32>()` 提取为 `(&Shape, &[f32])`，再转为 `Vec<f32>`

#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use tracing::{info, span, warn, Level};
use vox_core::{InferenceSession, Tensor, VoxError};

use crate::InferError;

/// ONNX Runtime session 封装，实现 [`InferenceSession`]。
///
/// 用 [`OrtSession::load`] 从文件加载模型，EP 按平台自动选择。
#[derive(Debug)]
pub struct OrtSession {
    session: Session,
}

impl OrtSession {
    /// 从文件加载 ONNX 模型，自动选择最优 EP。
    ///
    /// # Errors
    /// 模型加载失败返回 [`InferError::Load`]，EP 初始化失败返回 [`InferError::ExecutionProvider`]。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, InferError> {
        Self::load_with_eps(path, default_execution_providers())
    }

    /// 从文件加载模型，使用指定的 EP 列表。
    ///
    /// # Errors
    /// 同 [`OrtSession::load`]。
    pub fn load_with_eps(
        path: impl AsRef<Path>,
        eps: Vec<EpDescriptor>,
    ) -> Result<Self, InferError> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(InferError::Load(format!(
                "model file not found: {}",
                path_ref.display()
            )));
        }

        let mut builder = Session::builder()
            .map_err(|e| InferError::Load(format!("session builder failed: {e}")))?;

        // 注册 EP（按优先级顺序）。
        for ep in &eps {
            match ep {
                EpDescriptor::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        builder = builder
                            .with_execution_providers([ort::ep::CUDA::default().build()])
                            .map_err(|e| {
                                InferError::ExecutionProvider(format!("cuda init failed: {e}"))
                            })?;
                        info!("execution provider: cuda enabled");
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        warn!("execution provider: cuda requested but feature not enabled");
                    }
                }
                EpDescriptor::DirectML => {
                    #[cfg(feature = "directml")]
                    {
                        builder = builder
                            .with_execution_providers([ort::ep::DirectML::default().build()])
                            .map_err(|e| {
                                InferError::ExecutionProvider(format!("directml init failed: {e}"))
                            })?;
                        info!("execution provider: directml enabled");
                    }
                    #[cfg(not(feature = "directml"))]
                    {
                        warn!("execution provider: directml requested but feature not enabled");
                    }
                }
                EpDescriptor::CoreML => {
                    #[cfg(feature = "coreml")]
                    {
                        builder = builder
                            .with_execution_providers([ort::ep::CoreML::default().build()])
                            .map_err(|e| {
                                InferError::ExecutionProvider(format!("coreml init failed: {e}"))
                            })?;
                        info!("execution provider: coreml enabled");
                    }
                    #[cfg(not(feature = "coreml"))]
                    {
                        warn!("execution provider: coreml requested but feature not enabled");
                    }
                }
                EpDescriptor::Cpu => {
                    // CPU EP 始终可用，无需显式注册（ort 默认使用 CPU）。
                    info!("execution provider: cpu (default)");
                }
            }
        }

        let session = builder
            .commit_from_file(path_ref)
            .map_err(|e| InferError::Load(format!("model load failed: {e}")))?;

        info!(
            model_path = %path_ref.display(),
            inputs = session.inputs().len(),
            outputs = session.outputs().len(),
            "onnx model loaded"
        );

        Ok(Self { session })
    }

    /// 获取底层 `ort::Session` 的引用（供高级用例，如查询输入/输出元数据）。
    pub fn inner(&self) -> &Session {
        &self.session
    }

    /// 将 [`vox_core::Tensor`] 转换为 `ort::value::Tensor`。
    fn to_ort_value(tensor: &Tensor) -> Result<OrtTensor<f32>, InferError> {
        let shape: Vec<usize> = tensor.shape.clone();
        let data = tensor.data.clone().into_boxed_slice();
        OrtTensor::from_array((shape, data))
            .map_err(|e| InferError::Runtime(format!("tensor creation failed: {e}")))
    }

    /// 将 `ort::value::DynValue` 输出转换为 [`vox_core::Tensor`]。
    fn from_ort_value(value: &ort::value::DynValue) -> Result<Tensor, InferError> {
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("tensor extraction failed: {e}")))?;

        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let data_vec = data.to_vec();
        Ok(Tensor {
            data: data_vec,
            shape: shape_vec,
        })
    }
}

impl InferenceSession for OrtSession {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, VoxError> {
        let _span = span!(Level::TRACE, "infer", input_count = inputs.len()).entered();

        // 构造 ort 输入：用 HashMap<name, DynTensor> 匹配 session 输入。
        // 先收集输入名（避免借用冲突），再构造 map。
        let input_names: Vec<String> = self
            .session
            .inputs()
            .iter()
            .map(|outlet| outlet.name().to_string())
            .collect();
        let mut inputs_map: HashMap<String, ort::value::DynTensor> = HashMap::new();
        for (name, tensor) in input_names.iter().zip(inputs.iter()) {
            let ort_val = Self::to_ort_value(tensor).map_err(|e| VoxError::infer(e.to_string()))?;
            inputs_map.insert(name.clone(), ort_val.upcast());
        }

        // 执行推理。
        let outputs = self
            .session
            .run(inputs_map)
            .map_err(|e| VoxError::infer(format!("inference run failed: {e}")))?;

        // 转换输出。
        let mut result = Vec::with_capacity(outputs.len());
        for (_name, value) in outputs.iter() {
            let tensor =
                Self::from_ort_value(&value).map_err(|e| VoxError::infer(e.to_string()))?;
            result.push(tensor);
        }

        Ok(result)
    }
}

/// Execution Provider 描述符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpDescriptor {
    /// NVIDIA CUDA（需 `cuda` feature）。
    Cuda,
    /// Windows DirectML（需 `directml` feature）。
    DirectML,
    /// Apple CoreML（需 `coreml` feature）。
    CoreML,
    /// CPU（始终可用）。
    Cpu,
}

/// 返回默认 EP 优先级列表：CUDA > DirectML > CoreML > CPU。
fn default_execution_providers() -> Vec<EpDescriptor> {
    vec![
        EpDescriptor::Cuda,
        EpDescriptor::DirectML,
        EpDescriptor::CoreML,
        EpDescriptor::Cpu,
    ]
}

/// 共享 session 的便捷类型别名（规范要求 `Arc<Session>` 共享）。
pub type SharedOrtSession = Arc<std::sync::Mutex<OrtSession>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_eps_in_priority_order() {
        let eps = default_execution_providers();
        assert_eq!(eps[0], EpDescriptor::Cuda);
        assert_eq!(eps[1], EpDescriptor::DirectML);
        assert_eq!(eps[2], EpDescriptor::CoreML);
        assert_eq!(eps[3], EpDescriptor::Cpu);
    }

    #[test]
    fn load_nonexistent_model_returns_error() {
        let result = OrtSession::load("nonexistent_model.onnx");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, InferError::Load(_)));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn ep_descriptor_is_copy() {
        let ep = EpDescriptor::Cuda;
        let ep_copy = ep;
        assert_eq!(ep, ep_copy);
    }
}
