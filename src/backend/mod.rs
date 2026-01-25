//! Backend abstractions for CPU and GPU execution.
//!
//! This module defines the [`Backend`] trait and implementations:
//! - [`Cpu`]: CPU backend with SIMD acceleration
//! - [`Cuda`]: CUDA backend (optional, requires `cuda` feature)

mod cpu;
mod traits;

pub use cpu::Cpu;
pub use traits::{Backend, Storage};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::Cuda;
