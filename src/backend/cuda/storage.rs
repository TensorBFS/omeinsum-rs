//! CUDA storage implementation for GPU memory management.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, DriverError};
use std::sync::Arc;

/// GPU memory storage backed by CUDA.
///
/// Provides a wrapper around cudarc's `CudaSlice` with device reference tracking.
pub struct CudaStorage<T> {
    slice: CudaSlice<T>,
    device: Arc<CudaDevice>,
}

impl<T> CudaStorage<T> {
    /// Create a new CudaStorage from a CudaSlice and device reference.
    pub fn new(slice: CudaSlice<T>, device: Arc<CudaDevice>) -> Self {
        Self { slice, device }
    }

    /// Get a reference to the underlying CUDA slice.
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Get a reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Number of elements in storage.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: DeviceRepr + Clone> CudaStorage<T> {
    /// Copy all data from GPU to a Vec on the host.
    ///
    /// # Errors
    ///
    /// Returns a `DriverError` if the CUDA device-to-host copy fails.
    pub fn to_vec(&self) -> Result<Vec<T>, DriverError> {
        self.device.dtoh_sync_copy(&self.slice)
    }
}

// SAFETY: CudaStorage<T> can be sent between threads because:
// - CudaSlice<T> internally uses a CUDA device pointer which is thread-safe
// - Arc<CudaDevice> is Send
// The actual GPU memory is managed by the CUDA runtime which handles synchronization.
unsafe impl<T: Send> Send for CudaStorage<T> {}

// SAFETY: CudaStorage<T> can be shared between threads because:
// - CudaSlice<T> only provides immutable access through &self methods
// - Arc<CudaDevice> is Sync
// - CUDA operations are synchronized by the runtime
unsafe impl<T: Sync> Sync for CudaStorage<T> {}
