//! CPU tensor contraction via reshape→GEMM→reshape.

use crate::algebra::{Algebra, Scalar};
use crate::backend::BackendScalar;
use super::Cpu;

/// Classify modes into batch, left, right, and contracted.
pub(super) fn classify_modes(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    todo!("Will be implemented in next task")
}
