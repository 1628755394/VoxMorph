// VoxMorph Tauri 桌面入口。保持极简，逻辑在 lib.rs（`proj-lib-main-split`）。
// 移动端入口由 `#[cfg_attr(mobile, tauri::mobile_entry_point)]` 在 lib.rs 标注。

fn main() {
    vox_ui::run()
}
