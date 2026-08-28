//! 变声管线 Stage 实现：FeatureStage / ConvertStage / VocoderStage。
//!
//! 这些 Stage 实现 [`crate::pipeline::Stage`] trait，串联成完整变声管线：
//!
//! ```text
//! [Audio Frame] → FeatureStage → [content features]
//!                → ConvertStage → [converted features]
//!                → VocoderStage → [audio waveform]
//! ```

pub mod convert;
pub mod feature;
pub mod passthrough;
pub mod rvc;
pub mod vocoder;

pub use convert::{ConvertInputLayout, ConvertStage};
pub use feature::FeatureStage;
pub use passthrough::PassthroughStage;
pub use rvc::{RvcLiveParams, RvcStage, RvcStageConfig, RvcStageError};
pub use vocoder::{VocoderInputLayout, VocoderStage};
