//! VoxMorph 模型与音色库管理。
//!
//! # 音色文件格式
//!
//! 每个音色 = 一个目录或一对文件：
//! - `*.toml`：元数据（名称、标签、F0 偏移建议）
//! - `*.bin`：embedding（原始 `f32` 字节序列）
//!
//! **不**塞进单个大 JSON（`mem-boxed-slice`：embedding 加载后用 `Box<[f32]>`）。
//!
//! # 模型路径
//!
//! 模型文件路径通过配置注入，**禁止硬编码绝对路径**。
//!
//! # 训练
//!
//! 用户自训练流程由 Python 脚本完成，Rust 只负责加载已生成音色与模型。
//!
//! # 实现里程碑
//!
//! - M5: 音色库索引、加载、列表
//! - M5: 模型注册表（HuBERT / Converter / Vocoder 三件套）

use thiserror::Error;
use vox_core::VoxError;

/// 模型/音色库错误。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// 模型或音色文件未找到。
    #[error("not found: {0}")]
    NotFound(String),
    /// 元数据解析失败（toml 反序列化）。
    #[error("metadata parse error: {0}")]
    Metadata(String),
    /// embedding 文件损坏或长度不匹配。
    #[error("embedding invalid: {0}")]
    Embedding(String),
    /// 来自核心层的错误。
    #[error(transparent)]
    Core(#[from] VoxError),
}
