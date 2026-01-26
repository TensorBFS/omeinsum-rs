//! CUDA backend for GPU execution.
//!
//! This module provides the CUDA backend implementation using cudarc.

mod cutensor;
mod storage;

pub use storage::CudaStorage;

use cudarc::driver::CudaDevice;
use cutensor::{contract, CacheKey, CutensorType, Handle, PlanCache, TensorDesc};
use num_complex::Complex;
use std::cell::RefCell;
use std::sync::Arc;

// ============================================================================
// CUDA-compatible complex number wrapper
// ============================================================================
//
// Due to Rust's orphan rule, we cannot implement cudarc traits for num_complex
// types directly. This generic newtype wrapper provides CUDA-compatible complex.

/// CUDA-compatible wrapper for complex numbers.
///
/// This type has the same memory layout as `num_complex::Complex<T>` and CUDA's
/// complex types, but can implement cudarc traits since it's a local type.
///
/// Use `CudaComplex<f32>` for single-precision and `CudaComplex<f64>` for double.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CudaComplex<T>(pub Complex<T>);

impl<T> CudaComplex<T> {
    /// Create a new CudaComplex from real and imaginary parts.
    pub fn new(re: T, im: T) -> Self {
        CudaComplex(Complex::new(re, im))
    }

    /// Get the real part.
    pub fn re(&self) -> T
    where
        T: Clone,
    {
        self.0.re.clone()
    }

    /// Get the imaginary part.
    pub fn im(&self) -> T
    where
        T: Clone,
    {
        self.0.im.clone()
    }
}

// SAFETY: CudaComplex<T> is repr(transparent) over Complex<T>, which is repr(C)
// with two T fields. This is compatible with CUDA's complex types.
unsafe impl<T: cudarc::driver::DeviceRepr> cudarc::driver::DeviceRepr for CudaComplex<T> {}
// SAFETY: Zero-initialized CudaComplex<T> is valid if T is valid as zero bits.
unsafe impl<T: cudarc::driver::ValidAsZeroBits> cudarc::driver::ValidAsZeroBits for CudaComplex<T> {}

// Arithmetic for CudaComplex<f32>
impl std::ops::Add for CudaComplex<f32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        CudaComplex(self.0 + rhs.0)
    }
}

impl std::ops::Mul for CudaComplex<f32> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        CudaComplex(self.0 * rhs.0)
    }
}

impl num_traits::Zero for CudaComplex<f32> {
    fn zero() -> Self {
        CudaComplex(Complex::new(0.0, 0.0))
    }
    fn is_zero(&self) -> bool {
        self.0.re == 0.0 && self.0.im == 0.0
    }
}

impl num_traits::One for CudaComplex<f32> {
    fn one() -> Self {
        CudaComplex(Complex::new(1.0, 0.0))
    }
}

// Arithmetic for CudaComplex<f64>
impl std::ops::Add for CudaComplex<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        CudaComplex(self.0 + rhs.0)
    }
}

impl std::ops::Mul for CudaComplex<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        CudaComplex(self.0 * rhs.0)
    }
}

impl num_traits::Zero for CudaComplex<f64> {
    fn zero() -> Self {
        CudaComplex(Complex::new(0.0, 0.0))
    }
    fn is_zero(&self) -> bool {
        self.0.re == 0.0 && self.0.im == 0.0
    }
}

impl num_traits::One for CudaComplex<f64> {
    fn one() -> Self {
        CudaComplex(Complex::new(1.0, 0.0))
    }
}

// CutensorType implementations
impl CutensorType for CudaComplex<f32> {
    const DATA: cutensor::sys::cutensorDataType_t = cutensor::sys::cutensorDataType_t::C_32F;
    fn compute_desc() -> cutensor::sys::cutensorComputeDescriptor_t {
        unsafe { cutensor::sys::CUTENSOR_COMPUTE_DESC_32F }
    }
}

impl CutensorType for CudaComplex<f64> {
    const DATA: cutensor::sys::cutensorDataType_t = cutensor::sys::cutensorDataType_t::C_64F;
    fn compute_desc() -> cutensor::sys::cutensorComputeDescriptor_t {
        unsafe { cutensor::sys::CUTENSOR_COMPUTE_DESC_64F }
    }
}

// Conversion traits
impl<T> From<Complex<T>> for CudaComplex<T> {
    fn from(c: Complex<T>) -> Self {
        CudaComplex(c)
    }
}

impl<T> From<CudaComplex<T>> for Complex<T> {
    fn from(c: CudaComplex<T>) -> Self {
        c.0
    }
}

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
            device,
            handle: RefCell::new(None),
            cache: RefCell::new(PlanCache::new(64)),
        })
    }

    /// Get a reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a reference to the cuTENSOR handle, initializing it lazily if needed.
    fn get_handle(&self) -> Result<std::cell::Ref<'_, Handle>, CudaError> {
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
        T: CutensorType + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + num_traits::One + num_traits::Zero,
    {
        // Create tensor descriptors
        let handle = self.get_handle()?;

        let desc_a = TensorDesc::new::<T>(&handle, shape_a, strides_a)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
        let desc_b = TensorDesc::new::<T>(&handle, shape_b, strides_b)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
        let desc_c = TensorDesc::new::<T>(&handle, shape_c, strides_c)
            .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

        // Need to drop handle borrow before borrowing cache mutably
        drop(handle);

        // Build cache key
        let key = CacheKey {
            shapes: vec![shape_a.to_vec(), shape_b.to_vec(), shape_c.to_vec()],
            strides: vec![strides_a.to_vec(), strides_b.to_vec(), strides_c.to_vec()],
            modes: vec![modes_a.to_vec(), modes_b.to_vec(), modes_c.to_vec()],
            dtype: T::DATA as u32,
        };

        // Allocate output storage
        let len: usize = shape_c.iter().product();
        let mut c = self
            .device
            .alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))?;

        // Get or create the execution plan from cache and execute contraction
        // Keep the cache borrow alive during the contract call
        {
            let handle = self.get_handle()?;
            let mut cache = self.cache.borrow_mut();
            let plan = cache
                .get_or_create::<T>(
                    &handle, key, &desc_a, modes_a, &desc_b, modes_b, &desc_c, modes_c,
                )
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

            // Execute the contraction
            contract::<T>(&handle, plan, T::one(), a.slice(), b.slice(), &mut c)
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
        }

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
