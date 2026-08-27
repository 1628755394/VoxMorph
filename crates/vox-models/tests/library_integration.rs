//! 音色库集成测试：端到端验证 save → load_from_dir → 查找。
//!
//! 模拟真实场景：
//! - 创建临时目录
//! - 保存多个音色（不同 ID、名称、标签）
//! - 用 TimbreLibrary::load_from_dir 加载
//! - 验证按 ID/名称/标签查找
//! - 验证 embedding 数据完整
//! - 清理临时目录

use vox_core::timbre::{Timbre, TimbreId};
use vox_models::{TimbreFile, TimbreLibrary};

fn make_timbre(id: u64, name: &str, embedding: Vec<f32>, tags: Vec<&str>, f0: f32) -> Timbre {
    Timbre {
        id: TimbreId::new(id),
        name: name.to_string(),
        embedding: embedding.into_boxed_slice(),
        f0_offset_semitones: f0,
        tags: tags.into_iter().map(String::from).collect(),
    }
}

#[test]
fn full_round_trip_multiple_timbres() {
    let dir = std::env::temp_dir().join("voxmorph_integration_timbre");
    // 确保干净开始。
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 创建 3 个不同音色。
    let timbres = vec![
        make_timbre(
            1,
            "Alice",
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec!["female", "anime"],
            -2.0,
        ),
        make_timbre(
            2,
            "Bob",
            vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            vec!["male", "deep"],
            2.0,
        ),
        make_timbre(
            3,
            "Carol",
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            vec!["female", "deep", "soft"],
            0.0,
        ),
    ];

    // 保存所有音色。
    for t in &timbres {
        TimbreFile::save(&dir, t).expect("save should succeed");
    }

    // 加载库。
    let lib = TimbreLibrary::load_from_dir(&dir).expect("load should succeed");
    assert_eq!(lib.len(), 3, "should load all 3 timbres");

    // 按 ID 查找。
    let alice = lib
        .find_by_id(&TimbreId::new(1))
        .expect("Alice should exist");
    assert_eq!(alice.name, "Alice");
    assert!((alice.f0_offset_semitones - (-2.0)).abs() < 1e-6);

    // 按名称查找。
    let bob = lib.find_by_name("Bob").expect("Bob should exist");
    assert_eq!(bob.id, TimbreId::new(2));

    // 按标签查找。
    let female = lib.find_by_tag("female");
    assert_eq!(female.len(), 2, "Alice and Carol are female");
    let deep = lib.find_by_tag("deep");
    assert_eq!(deep.len(), 2, "Bob and Carol are deep");

    // 验证 embedding 数据完整（round-trip）。
    for original in &timbres {
        let loaded = lib
            .find_by_id(&original.id)
            .expect("timbre should be found by id");
        assert_eq!(loaded.embedding.len(), original.embedding.len());
        for (a, b) in loaded.embedding.iter().zip(original.embedding.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "embedding mismatch for {}",
                original.name
            );
        }
    }

    // 清理。
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn empty_dir_loads_empty_library() {
    let dir = std::env::temp_dir().join("voxmorph_empty_dir_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let lib = TimbreLibrary::load_from_dir(&dir).expect("empty dir should load");
    assert!(lib.is_empty());
    assert_eq!(lib.len(), 0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mixed_valid_invalid_files_loads_only_valid() {
    let dir = std::env::temp_dir().join("voxmorph_mixed_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 保存 2 个有效音色。
    let t1 = make_timbre(1, "Valid1", vec![0.5; 4], vec!["test"], 0.0);
    let t2 = make_timbre(2, "Valid2", vec![0.3; 4], vec!["test"], 1.0);
    TimbreFile::save(&dir, &t1).unwrap();
    TimbreFile::save(&dir, &t2).unwrap();

    // 写一个损坏的 toml（无对应 .bin）。
    std::fs::write(
        dir.join("broken.toml"),
        "id = 3\nname = \"Broken\"\nf0_offset_semitones = 0.0\ntags = []\nembedding_len = 4\n",
    )
    .unwrap();

    // 写一个非 toml 文件（应被忽略）。
    std::fs::write(dir.join("readme.txt"), "not a timbre").unwrap();

    let lib = TimbreLibrary::load_from_dir(&dir).expect("should load valid timbres");
    assert_eq!(lib.len(), 2, "should load only 2 valid timbres");
    assert!(lib.find_by_name("Valid1").is_some());
    assert!(lib.find_by_name("Valid2").is_some());
    assert!(
        lib.find_by_name("Broken").is_none(),
        "broken timbre should be skipped"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tags_filter_case_insensitive() {
    let dir = std::env::temp_dir().join("voxmorph_tag_ci_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let t = make_timbre(1, "Mixed", vec![0.0; 4], vec!["Female", "ANIME"], 0.0);
    TimbreFile::save(&dir, &t).unwrap();

    let lib = TimbreLibrary::load_from_dir(&dir).unwrap();
    // find_by_tag 精确匹配（大小写敏感）。
    assert_eq!(
        lib.find_by_tag("female").len(),
        0,
        "exact match should fail"
    );
    assert_eq!(lib.find_by_tag("Female").len(), 1);
    // find_by_tag_ci 大小写不敏感。
    assert_eq!(lib.find_by_tag_ci("female").len(), 1);
    assert_eq!(lib.find_by_tag_ci("anime").len(), 1);
    assert_eq!(lib.find_by_tag_ci("ANIME").len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}
