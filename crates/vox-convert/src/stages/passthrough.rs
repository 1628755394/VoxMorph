//! Passthrough Stage：输入帧原样输出，用于直通模式与测试。
//!
//! 不做任何 DSP 处理，仅复制样本与元数据。实时管线要求至少一个 Stage
//!（空管线会被 `Pipeline::start` 拒绝），故直通模式用此 Stage 占位。

use vox_core::{Frame, VoxError};

use crate::Stage;

/// 直通 Stage：输入即输出。
#[derive(Debug, Default)]
pub struct PassthroughStage;

impl PassthroughStage {
    /// 构造。
    pub fn new() -> Self {
        Self
    }
}

impl Stage for PassthroughStage {
    fn process(&mut self, input: &Frame, output: &mut Frame) -> Result<(), VoxError> {
        output.samples.clear();
        output.samples.extend_from_slice(&input.samples);
        output.sample_rate = input.sample_rate;
        output.channels = input.channels;
        output.timestamp = input.timestamp;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_copies_frame() {
        let mut stage = PassthroughStage::new();
        let input = Frame {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
            channels: 1,
            timestamp: 42,
        };
        let mut output = Frame {
            samples: Vec::new(),
            sample_rate: 0,
            channels: 0,
            timestamp: 0,
        };
        stage.process(&input, &mut output).unwrap();
        assert_eq!(output.samples, input.samples);
        assert_eq!(output.sample_rate, 16000);
        assert_eq!(output.channels, 1);
        assert_eq!(output.timestamp, 42);
    }
}
