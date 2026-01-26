//! cuTENSOR bindings and safe wrappers.
//!
//! This module provides bindings to NVIDIA's cuTENSOR library for
//! high-performance tensor contractions on CUDA GPUs.

mod contract;
mod handle;
pub mod sys;

pub use contract::{contract, CacheKey, PlanCache};
pub use handle::{CutensorType, Handle, Plan, TensorDesc};

use sys::cutensorStatus_t;

/// Errors that can occur during cuTENSOR operations.
#[derive(Debug)]
pub enum CutensorError {
    /// cuTENSOR library not initialized.
    NotInitialized,
    /// Memory allocation failed.
    AllocFailed,
    /// Invalid parameter value.
    InvalidValue,
    /// Architecture mismatch (GPU compute capability not supported).
    ArchMismatch,
    /// Operation not supported.
    NotSupported,
    /// cuTENSOR status error with status code.
    Status(i32),
    /// Other error with a descriptive message.
    Other(String),
}

impl std::fmt::Display for CutensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CutensorError::NotInitialized => write!(f, "cuTENSOR not initialized"),
            CutensorError::AllocFailed => write!(f, "cuTENSOR allocation failed"),
            CutensorError::InvalidValue => write!(f, "cuTENSOR invalid value"),
            CutensorError::ArchMismatch => write!(f, "cuTENSOR architecture mismatch"),
            CutensorError::NotSupported => write!(f, "cuTENSOR operation not supported"),
            CutensorError::Status(code) => write!(f, "cuTENSOR status error (code {})", code),
            CutensorError::Other(msg) => write!(f, "cuTENSOR error: {}", msg),
        }
    }
}

impl std::error::Error for CutensorError {}

/// Check a cuTENSOR status and convert to Result.
///
/// # Arguments
/// * `status` - The status code returned by a cuTENSOR function
///
/// # Returns
/// * `Ok(())` if the status indicates success
/// * `Err(CutensorError)` with the appropriate error variant otherwise
pub fn check(status: cutensorStatus_t) -> Result<(), CutensorError> {
    match status {
        cutensorStatus_t::SUCCESS => Ok(()),
        cutensorStatus_t::NOT_INITIALIZED => Err(CutensorError::NotInitialized),
        cutensorStatus_t::ALLOC_FAILED => Err(CutensorError::AllocFailed),
        cutensorStatus_t::INVALID_VALUE => Err(CutensorError::InvalidValue),
        cutensorStatus_t::ARCH_MISMATCH => Err(CutensorError::ArchMismatch),
        cutensorStatus_t::NOT_SUPPORTED => Err(CutensorError::NotSupported),
        other => Err(CutensorError::Status(other as i32)),
    }
}
