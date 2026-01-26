//! CUDA backend for GPU execution.
//!
//! This module provides the CUDA backend implementation using cudarc.

mod cutensor;
mod storage;

pub use storage::CudaStorage;

use crate::algebra::Scalar;
use cutensor::{contract, CacheKey, CutensorType, Handle, PlanCache, TensorDesc};
use cudarc::driver::CudaDevice;
use std::cell::RefCell;
use std::sync::Arc;

/// CUDA backend for GPU tensor operations.
///
/// Wraps a CUDA device and provides methods for GPU memory management
/// and tensor contractions via cuTENSOR.
pub struct Cuda {
    device: Arc<CudaDevice>,
    handle: RefCell<Option<Handle>>,
    cache: RefCell<PlanCache>,
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
        let device = CudaDevice::new(ordinal).map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(Self {
            device: Arc::new(device),
            handle: RefCell::new(None),
            cache: RefCell::new(PlanCache::new(64)),
        })
    }

    /// Get a reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a reference to the cuTENSOR handle, initializing it lazily if needed.
    fn get_handle(&self) -> Result<std::cell::Ref<Handle>, CudaError> {
        {
            let mut h = self.handle.borrow_mut();
            if h.is_none() {
                *h = Some(
                    Handle::new(self.device.clone())
                        .map_err(|e| CudaError::Cutensor(format!("{}", e)))?,
                );
            }
        }
        Ok(std::cell::Ref::map(self.handle.borrow(), |h| {
            h.as_ref().unwrap()
        }))
    }

    /// Perform a tensor contraction using cuTENSOR.
    ///
    /// Computes: C = A * B, where the contraction is specified by mode indices.
    ///
    /// # Arguments
    /// * `a` - Input tensor A storage
    /// * `shape_a` - Shape (extents) of tensor A
    /// * `strides_a` - Strides of tensor A
    /// * `modes_a` - Mode indices for tensor A
    /// * `b` - Input tensor B storage
    /// * `shape_b` - Shape (extents) of tensor B
    /// * `strides_b` - Strides of tensor B
    /// * `modes_b` - Mode indices for tensor B
    /// * `shape_c` - Shape (extents) of output tensor C
    /// * `strides_c` - Strides of output tensor C
    /// * `modes_c` - Mode indices for output tensor C
    ///
    /// # Returns
    /// * `Ok(CudaStorage<T>)` containing the contraction result
    /// * `Err(CudaError)` if the contraction fails
    #[allow(clippy::too_many_arguments)]
    pub fn contract<T>(
        &self,
        a: &CudaStorage<T>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<T>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        strides_c: &[usize],
        modes_c: &[i32],
    ) -> Result<CudaStorage<T>, CudaError>
    where
        T: Scalar + CutensorType + cudarc::driver::DeviceRepr + num_traits::One + num_traits::Zero,
    {
        // Create tensor descriptors
        let handle = self.get_handle()?;

        let desc_a = TensorDesc::new::<T>(&handle, shape_a, strides_a)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
        let desc_b = TensorDesc::new::<T>(&handle, shape_b, strides_b)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
        let desc_c = TensorDesc::new::<T>(&handle, shape_c, strides_c)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

        // Build cache key
        let key = CacheKey {
            shapes: vec![shape_a.to_vec(), shape_b.to_vec(), shape_c.to_vec()],
            strides: vec![strides_a.to_vec(), strides_b.to_vec(), strides_c.to_vec()],
            modes: vec![modes_a.to_vec(), modes_b.to_vec(), modes_c.to_vec()],
            dtype: T::DATA as u32,
        };

        // Get or create the execution plan from cache
        // Need to drop handle borrow before borrowing cache mutably
        drop(handle);

        let handle = self.get_handle()?;
        let plan = self
            .cache
            .borrow_mut()
            .get_or_create::<T>(
                &handle, key, &desc_a, modes_a, &desc_b, modes_b, &desc_c, modes_c,
            )
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

        // Allocate output storage
        let len: usize = shape_c.iter().product();
        let mut c = self
            .device
            .alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))?;

        // Execute the contraction
        contract::<T>(&handle, plan, T::one(), a.slice(), b.slice(), &mut c)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

        Ok(CudaStorage::new(c, self.device.clone()))
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
