//! 音色库：从目录扫描、加载、索引、查找音色。
//!
//! # 用法
//!
//! ```no_run
//! use vox_models::TimbreLibrary;
//! use vox_core::timbre::TimbreId;
//!
//! let library = TimbreLibrary::load_from_dir("models/timbres").unwrap();
//! let all = library.list();
//! let by_id = library.find_by_id(&TimbreId::new(1));
//! let by_name = library.find_by_name("My Voice");
//! let tagged = library.find_by_tag("female");
//! ```
//!
//! # 约束
//!
//! - 扫描目录下所有 `.toml` 文件，配对加载 `.bin`
//! - 加载失败的音色跳过并 `tracing::warn!`，不中断整个库加载
//! - ID 唯一性校验：重复 ID 跳过后者并 `tracing::warn!`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{info, warn};
use vox_core::timbre::{Timbre, TimbreId};

use crate::{ModelError, TimbreFile};

/// 音色库：已加载的音色集合，支持按 ID/名称/标签查找。
#[derive(Debug)]
pub struct TimbreLibrary {
    timbres: Vec<Timbre>,
    by_id: HashMap<TimbreId, usize>,
    by_name: HashMap<String, usize>,
}

impl TimbreLibrary {
    /// 从目录加载所有音色。
    ///
    /// 扫描 `dir` 下所有 `.toml` 文件，配对加载 `.bin`。
    /// 加载失败的音色跳过并 `tracing::warn!`。
    ///
    /// # Errors
    /// - 目录不存在返回 [`ModelError::NotFound`]
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        let dir_ref = dir.as_ref();
        if !dir_ref.exists() {
            return Err(ModelError::NotFound(format!(
                "timbre directory not found: {}",
                dir_ref.display()
            )));
        }

        let mut timbres: Vec<Timbre> = Vec::new();
        let mut by_id: HashMap<TimbreId, usize> = HashMap::new();
        let mut by_name: HashMap<String, usize> = HashMap::new();

        // 扫描 .toml 文件。
        let entries = std::fs::read_dir(dir_ref)
            .map_err(|e| ModelError::NotFound(format!("failed to read dir: {e}")))?;

        let mut toml_files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        toml_files.sort();

        for toml_path in &toml_files {
            match TimbreFile::load(toml_path) {
                Ok(timbre) => {
                    // ID 唯一性校验。
                    if by_id.contains_key(&timbre.id) {
                        warn!(
                            timbre_id = timbre.id.get(),
                            name = %timbre.name,
                            "duplicate timbre id, skipping"
                        );
                        continue;
                    }
                    let idx = timbres.len();
                    by_id.insert(timbre.id, idx);
                    by_name.insert(timbre.name.clone(), idx);
                    timbres.push(timbre);
                }
                Err(e) => {
                    warn!(
                        path = %toml_path.display(),
                        error = %e,
                        "failed to load timbre, skipping"
                    );
                }
            }
        }

        info!(
            loaded = timbres.len(),
            total_found = toml_files.len(),
            "timbre library loaded"
        );

        Ok(Self {
            timbres,
            by_id,
            by_name,
        })
    }

    /// 从已有音色列表构造库（供测试/编程式构建）。
    pub fn from_timbres(timbres: Vec<Timbre>) -> Self {
        let mut by_id = HashMap::new();
        let mut by_name = HashMap::new();
        for (idx, t) in timbres.iter().enumerate() {
            by_id.insert(t.id, idx);
            by_name.insert(t.name.clone(), idx);
        }
        Self {
            timbres,
            by_id,
            by_name,
        }
    }

    /// 列出所有音色。
    pub fn list(&self) -> &[Timbre] {
        &self.timbres
    }

    /// 音色数量。
    pub fn len(&self) -> usize {
        self.timbres.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.timbres.is_empty()
    }

    /// 按 ID 查找。
    pub fn find_by_id(&self, id: &TimbreId) -> Option<&Timbre> {
        self.by_id.get(id).map(|&idx| &self.timbres[idx])
    }

    /// 按名称查找（精确匹配）。
    pub fn find_by_name(&self, name: &str) -> Option<&Timbre> {
        self.by_name.get(name).map(|&idx| &self.timbres[idx])
    }

    /// 按标签查找（返回所有包含该标签的音色）。
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Timbre> {
        self.timbres
            .iter()
            .filter(|t| t.tags.iter().any(|tg| tg == tag))
            .collect()
    }

    /// 按标签查找（大小写不敏感）。
    pub fn find_by_tag_ci(&self, tag: &str) -> Vec<&Timbre> {
        let tag_lower = tag.to_lowercase();
        self.timbres
            .iter()
            .filter(|t| t.tags.iter().any(|tg| tg.to_lowercase() == tag_lower))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::timbre::TimbreId;

    fn make_timbre(id: u64, name: &str, tags: Vec<&str>) -> Timbre {
        Timbre {
            id: TimbreId::new(id),
            name: name.to_string(),
            embedding: vec![0.0; 8].into_boxed_slice(),
            f0_offset_semitones: 0.0,
            tags: tags.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn from_timbres_builds_indices() {
        let lib = TimbreLibrary::from_timbres(vec![
            make_timbre(1, "Alice", vec!["female"]),
            make_timbre(2, "Bob", vec!["male"]),
        ]);
        assert_eq!(lib.len(), 2);
        assert!(lib.find_by_id(&TimbreId::new(1)).is_some());
        assert!(lib.find_by_name("Bob").is_some());
        assert!(lib.find_by_name("Charlie").is_none());
    }

    #[test]
    fn find_by_tag_returns_matches() {
        let lib = TimbreLibrary::from_timbres(vec![
            make_timbre(1, "Alice", vec!["female", "anime"]),
            make_timbre(2, "Bob", vec!["male", "deep"]),
            make_timbre(3, "Carol", vec!["female", "deep"]),
        ]);
        let female = lib.find_by_tag("female");
        assert_eq!(female.len(), 2);
        let deep = lib.find_by_tag("deep");
        assert_eq!(deep.len(), 2);
        let none = lib.find_by_tag("child");
        assert!(none.is_empty());
    }

    #[test]
    fn find_by_tag_ci_case_insensitive() {
        let lib = TimbreLibrary::from_timbres(vec![make_timbre(1, "A", vec!["Female"])]);
        assert_eq!(lib.find_by_tag_ci("female").len(), 1);
        assert_eq!(lib.find_by_tag_ci("FEMALE").len(), 1);
    }

    #[test]
    fn empty_library() {
        let lib = TimbreLibrary::from_timbres(vec![]);
        assert!(lib.is_empty());
        assert_eq!(lib.len(), 0);
        assert!(lib.list().is_empty());
    }

    #[test]
    fn load_from_nonexistent_dir_returns_error() {
        let result = TimbreLibrary::load_from_dir("nonexistent_dir_xyz");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelError::NotFound(_)));
    }

    #[test]
    fn load_from_dir_with_multiple_timbres() {
        let dir = std::env::temp_dir().join("voxmorph_library_test");
        std::fs::create_dir_all(&dir).unwrap();

        let t1 = make_timbre(10, "Voice A", vec!["low"]);
        let t2 = make_timbre(20, "Voice B", vec!["high"]);
        TimbreFile::save(&dir, &t1).unwrap();
        TimbreFile::save(&dir, &t2).unwrap();

        let lib = TimbreLibrary::load_from_dir(&dir).unwrap();
        assert_eq!(lib.len(), 2);
        assert!(lib.find_by_name("Voice A").is_some());
        assert!(lib.find_by_name("Voice B").is_some());
        assert!(lib.find_by_id(&TimbreId::new(10)).is_some());
        assert_eq!(lib.find_by_tag("low").len(), 1);

        // 清理。
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_dir_skips_duplicate_ids() {
        let dir = std::env::temp_dir().join("voxmorph_library_dup_test");
        std::fs::create_dir_all(&dir).unwrap();

        let t1 = make_timbre(99, "First", vec![]);
        let t2 = make_timbre(99, "Second", vec![]); // 相同 ID
        TimbreFile::save(&dir, &t1).unwrap();
        TimbreFile::save(&dir, &t2).unwrap();

        let lib = TimbreLibrary::load_from_dir(&dir).unwrap();
        // 只加载第一个，第二个因重复 ID 跳过。
        assert_eq!(lib.len(), 1);
        assert!(lib.find_by_name("First").is_some() || lib.find_by_name("Second").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_dir_skips_corrupted_files() {
        let dir = std::env::temp_dir().join("voxmorph_library_corrupt_test");
        std::fs::create_dir_all(&dir).unwrap();

        // 正常音色。
        let good = make_timbre(1, "Good", vec![]);
        TimbreFile::save(&dir, &good).unwrap();

        // 损坏的 toml（无对应 .bin）。
        std::fs::write(
            dir.join("bad.toml"),
            "id = 2\nname = \"Bad\"\nf0_offset_semitones = 0.0\ntags = []\nembedding_len = 4\n",
        )
        .unwrap();

        let lib = TimbreLibrary::load_from_dir(&dir).unwrap();
        // 只加载好的，坏的跳过。
        assert_eq!(lib.len(), 1);
        assert!(lib.find_by_name("Good").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
