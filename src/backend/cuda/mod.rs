//! CUDA backend for GPU execution.
//!
//! This module provides the CUDA backend implementation using cudarc.

mod cutensor;
mod storage;
pub use storage::CudaStorage;

use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// CUDA backend for GPU tensor operations.
///
/// Wraps a CUDA device and provides methods for GPU memory management.
pub struct Cuda {
    device: Arc<CudaDevice>,
}

impl Cuda {
    /// Create a new CUDA backend on the default device (device 0).
    pub fn new() -> Result<Self, CudaError> {
        Self::on_device(0)
    }

    /// Create a new CUDA backend on a specific device.
    ///
    /// # Arguments
    /// * `ordinal` - The device ordinal (0-indexed)
    pub fn on_device(ordinal: usize) -> Result<Self, CudaError> {
        let device =
            CudaDevice::new(ordinal).map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(Self { device })
    }

    /// Get a reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Errors that can occur during CUDA operations.
#[derive(Debug)]
pub enum CudaError {
    /// Error initializing or accessing the CUDA device.
    Device(String),
    /// Error allocating GPU memory.
    Alloc(String),
    /// Error in cuTENSOR operations.
    Cutensor(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Device(msg) => write!(f, "CUDA device error: {}", msg),
            CudaError::Alloc(msg) => write!(f, "CUDA allocation error: {}", msg),
            CudaError::Cutensor(msg) => write!(f, "cuTENSOR error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}
