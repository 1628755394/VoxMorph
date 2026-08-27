//! 音色文件 I/O：`.toml` 元数据 + `.bin` embedding 分离存储。
//!
//! # 文件格式
//!
//! 每个音色对应一对文件（共享同名 stem）：
//! - `<name>.toml`：元数据（id, name, f0_offset_semitones, tags, embedding_len）
//! - `<name>.bin`：embedding 原始 `f32` 字节序列（小端）
//!
//! **不**塞进单个大 JSON（规范要求）。embedding 加载后用 `Box<[f32]>`。
//!
//! # 约束
//!
//! - 路径通过参数注入，**不硬编码绝对路径**
//! - embedding 长度必须与 toml 中 `embedding_len` 匹配，否则返回 `Embedding` 错误

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::info;
use vox_core::timbre::{Timbre, TimbreId};

use crate::ModelError;

/// `.toml` 元数据结构（不含 embedding 数据，仅含长度校验字段）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimbreMetadata {
    pub id: TimbreId,
    pub name: String,
    pub f0_offset_semitones: f32,
    pub tags: Vec<String>,
    /// embedding 元素数（用于校验 .bin 文件长度）。
    pub embedding_len: usize,
}

impl TimbreMetadata {
    /// 从 `Timbre` 提取元数据（不含 embedding 数据）。
    pub fn from_timbre(timbre: &Timbre) -> Self {
        Self {
            id: timbre.id,
            name: timbre.name.clone(),
            f0_offset_semitones: timbre.f0_offset_semitones,
            tags: timbre.tags.clone(),
            embedding_len: timbre.embedding.len(),
        }
    }
}

/// 音色文件对：`.toml` + `.bin`。
///
/// 封装加载/存储逻辑，分离元数据与二进制 embedding。
pub struct TimbreFile;

impl TimbreFile {
    /// 从文件对加载音色。
    ///
    /// `toml_path` 指向 `.toml` 元数据文件，`.bin` 路径由 toml stem 推导。
    ///
    /// # Errors
    /// - 文件不存在返回 [`ModelError::NotFound`]
    /// - toml 解析失败返回 [`ModelError::Metadata`]
    /// - embedding 长度不匹配返回 [`ModelError::Embedding`]
    pub fn load(toml_path: &Path) -> Result<Timbre, ModelError> {
        if !toml_path.exists() {
            return Err(ModelError::NotFound(format!(
                "metadata file not found: {}",
                toml_path.display()
            )));
        }

        // 读取并解析 toml 元数据。
        let toml_content = fs::read_to_string(toml_path)
            .map_err(|e| ModelError::Metadata(format!("failed to read toml: {e}")))?;
        let metadata: TimbreMetadata = toml::from_str(&toml_content)
            .map_err(|e| ModelError::Metadata(format!("toml parse failed: {e}")))?;

        // 推导 .bin 路径。
        let bin_path = bin_path_from_toml(toml_path);
        if !bin_path.exists() {
            return Err(ModelError::NotFound(format!(
                "embedding file not found: {}",
                bin_path.display()
            )));
        }

        // 读取 embedding 二进制。
        let bin_bytes = fs::read(&bin_path)
            .map_err(|e| ModelError::Embedding(format!("failed to read bin: {e}")))?;

        // 校验长度：f32 = 4 字节。
        let expected_bytes = metadata.embedding_len * 4;
        if bin_bytes.len() != expected_bytes {
            return Err(ModelError::Embedding(format!(
                "embedding length mismatch: expected {expected_bytes} bytes ({} f32), got {} bytes",
                metadata.embedding_len,
                bin_bytes.len()
            )));
        }

        // 转换为 f32 数组（小端）。
        let embedding: Box<[f32]> = bin_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();

        info!(
            timbre_id = metadata.id.get(),
            name = %metadata.name,
            embedding_len = metadata.embedding_len,
            "timbre loaded"
        );

        Ok(Timbre {
            id: metadata.id,
            name: metadata.name,
            embedding,
            f0_offset_semitones: metadata.f0_offset_semitones,
            tags: metadata.tags,
        })
    }

    /// 保存音色到文件对。
    ///
    /// 在 `dir` 目录下创建 `<name>.toml` 和 `<name>.bin`。
    /// 文件名取自 `timbre.name`（sanitize 后）。
    ///
    /// # Errors
    /// - 目录不存在或不可写返回 [`ModelError::NotFound`]
    /// - 写入失败返回 [`ModelError::Metadata`] 或 [`ModelError::Embedding`]
    pub fn save(dir: &Path, timbre: &Timbre) -> Result<PathBuf, ModelError> {
        if !dir.exists() {
            return Err(ModelError::NotFound(format!(
                "directory not found: {}",
                dir.display()
            )));
        }

        let stem = sanitize_filename(&timbre.name);
        let toml_path = dir.join(format!("{stem}.toml"));
        let bin_path = dir.join(format!("{stem}.bin"));

        // 写 toml 元数据。
        let metadata = TimbreMetadata::from_timbre(timbre);
        let toml_content = toml::to_string_pretty(&metadata)
            .map_err(|e| ModelError::Metadata(format!("toml serialize failed: {e}")))?;
        fs::write(&toml_path, toml_content)
            .map_err(|e| ModelError::Metadata(format!("failed to write toml: {e}")))?;

        // 写 bin embedding（f32 小端）。
        let bin_bytes: Vec<u8> = timbre
            .embedding
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        fs::write(&bin_path, bin_bytes)
            .map_err(|e| ModelError::Embedding(format!("failed to write bin: {e}")))?;

        info!(
            timbre_id = timbre.id.get(),
            name = %timbre.name,
            toml_path = %toml_path.display(),
            "timbre saved"
        );

        Ok(toml_path)
    }
}

/// 从 `.toml` 路径推导 `.bin` 路径（替换扩展名）。
fn bin_path_from_toml(toml_path: &Path) -> PathBuf {
    let stem = toml_path.file_stem().unwrap_or_default();
    let mut bin = toml_path.to_path_buf();
    bin.set_file_name(stem);
    bin.set_extension("bin");
    bin
}

/// 清理文件名：只保留字母数字、`-`、`_`，其余替换为 `_`。
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_nonexistent_toml_returns_not_found() {
        let result = TimbreFile::load(Path::new("nonexistent.toml"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ModelError::NotFound(_)));
    }

    #[test]
    fn load_missing_bin_returns_not_found() {
        let dir = std::env::temp_dir();
        let toml_path = dir.join("test_no_bin.toml");
        // TimbreId 是 newtype(u64)，serde 序列化为裸整数。
        fs::write(
            &toml_path,
            "id = 1\nname = \"test\"\nf0_offset_semitones = 0.0\ntags = []\nembedding_len = 4\n",
        )
        .unwrap();
        let result = TimbreFile::load(&toml_path);
        let _ = fs::remove_file(&toml_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::NotFound(_)));
    }

    #[test]
    fn save_then_load_round_trip() {
        let dir = std::env::temp_dir().join("voxmorph_timbre_test");
        fs::create_dir_all(&dir).unwrap();

        let timbre = Timbre {
            id: TimbreId::new(42),
            name: "Test Voice".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5].into_boxed_slice(),
            f0_offset_semitones: -2.0,
            tags: vec!["female".to_string(), "anime".to_string()],
        };

        let toml_path = TimbreFile::save(&dir, &timbre).unwrap();
        let loaded = TimbreFile::load(&toml_path).unwrap();

        assert_eq!(loaded.id, timbre.id);
        assert_eq!(loaded.name, timbre.name);
        assert_eq!(loaded.f0_offset_semitones, timbre.f0_offset_semitones);
        assert_eq!(loaded.tags, timbre.tags);
        assert_eq!(loaded.embedding.len(), timbre.embedding.len());
        for (a, b) in loaded.embedding.iter().zip(timbre.embedding.iter()) {
            assert!((a - b).abs() < 1e-6, "embedding mismatch: {a} vs {b}");
        }

        // 清理。
        let _ = fs::remove_file(&toml_path);
        let _ = fs::remove_file(bin_path_from_toml(&toml_path));
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn embedding_length_mismatch_returns_error() {
        let dir = std::env::temp_dir().join("voxmorph_timbre_mismatch");
        fs::create_dir_all(&dir).unwrap();

        // 写 toml 声明 embedding_len = 5，但 bin 只含 3 个 f32。
        let toml_path = dir.join("mismatch.toml");
        let toml_content = r#"id = 1
name = "mismatch"
f0_offset_semitones = 0.0
tags = []
embedding_len = 5
"#;
        fs::write(&toml_path, toml_content).unwrap();

        let bin_path = dir.join("mismatch.bin");
        let mut bin_file = fs::File::create(&bin_path).unwrap();
        for v in [0.1_f32, 0.2, 0.3] {
            bin_file.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(bin_file);

        let result = TimbreFile::load(&toml_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::Embedding(_)));

        let _ = fs::remove_file(&toml_path);
        let _ = fs::remove_file(&bin_path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn sanitize_replaces_special_chars() {
        assert_eq!(sanitize_filename("hello world"), "hello_world");
        assert_eq!(sanitize_filename("voice-1_2"), "voice-1_2");
        assert_eq!(sanitize_filename("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_filename("纯中文"), "纯中文");
    }

    #[test]
    fn bin_path_from_toml_replaces_extension() {
        let toml = Path::new("foo/bar/timbre.toml");
        let bin = bin_path_from_toml(toml);
        assert_eq!(bin, PathBuf::from("foo/bar/timbre.bin"));
    }

    #[test]
    fn metadata_from_timbre_extracts_fields() {
        let timbre = Timbre {
            id: TimbreId::new(7),
            name: "test".to_string(),
            embedding: vec![0.0; 256].into_boxed_slice(),
            f0_offset_semitones: 3.0,
            tags: vec!["low".to_string()],
        };
        let meta = TimbreMetadata::from_timbre(&timbre);
        assert_eq!(meta.id, timbre.id);
        assert_eq!(meta.name, timbre.name);
        assert_eq!(meta.embedding_len, 256);
        assert_eq!(meta.f0_offset_semitones, 3.0);
        assert_eq!(meta.tags, timbre.tags);
    }
}
