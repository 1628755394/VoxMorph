//! 音色库类型。
//!
//! 音色文件格式：`.toml` 元数据 + `.bin` embedding（`mem-boxed-slice`），
//! 不塞进单个大 JSON。用户自训练由 Python 脚本完成，Rust 只负责加载。

use serde::{Deserialize, Serialize};

/// 音色 ID，newtype 防止与模型 ID 混淆（`type-newtype-ids`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimbreId(pub u64);

impl TimbreId {
    /// 构造一个新 ID。
    #[inline]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    /// 取内部值。
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TimbreId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TimbreId({})", self.0)
    }
}

/// 一个音色：embedding + 元数据。
///
/// `embedding` 加载后不变，用 `Box<[f32]>` 而非 `Vec<f32>` 省容量字段
/// （`mem-boxed-slice`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timbre {
    /// 唯一标识。
    pub id: TimbreId,
    /// 人类可读名称。
    pub name: String,
    /// 音色嵌入向量（HuBERT/ContentVec speaker embedding）。
    pub embedding: Box<[f32]>,
    /// 建议的基频偏移（半音），可被用户参数覆盖。
    pub f0_offset_semitones: f32,
    /// 标签（如 "female", "anime", "deep"）。
    pub tags: Vec<String>,
}
