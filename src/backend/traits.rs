//! Backend trait definitions.

use crate::algebra::{Algebra, Scalar};

/// Storage trait for tensor data.
///
/// Abstracts over different storage backends (CPU memory, GPU memory).
pub trait Storage<T: Scalar>: Clone + Send + Sync + Sized {
    /// Number of elements in storage.
    fn len(&self) -> usize;

    /// Check if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index (may be slow for GPU).
    fn get(&self, index: usize) -> T;

    /// Set element at index (may be slow for GPU).
    fn set(&mut self, index: usize, value: T);

    /// Copy all data to a Vec (downloads from GPU if needed).
    fn to_vec(&self) -> Vec<T>;

    /// Create storage from slice.
    fn from_slice(data: &[T]) -> Self;

    /// Create zero-initialized storage.
    fn zeros(len: usize) -> Self;
}

/// Marker trait for scalar types supported by a specific backend.
///
/// This enables compile-time checking that a scalar type is supported
/// by a particular backend (e.g., CUDA only supports f32/f64/complex).
pub trait BackendScalar<B: Backend>: Scalar {}

/// Backend trait for tensor execution.
///
/// Defines how tensor operations are executed on different hardware.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Storage type for this backend.
    type Storage<T: Scalar>: Storage<T>;

    /// Backend name for debugging.
    fn name() -> &'static str;

    /// Synchronize all pending operations.
    fn synchronize(&self);

    /// Allocate storage.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>;

    /// Create storage from slice.
    #[allow(clippy::wrong_self_convention)]
    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T>;

    /// Copy strided data to contiguous storage.
    ///
    /// This is the core operation for making non-contiguous tensors contiguous.
    fn copy_strided<T: Scalar>(
        &self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T>;

    /// Binary tensor contraction.
    ///
    /// Computes: C[modes_c] = Σ A[modes_a] ⊗ B[modes_b]
    /// where the sum is over indices appearing in both A and B but not in C.
    ///
    /// # Arguments
    /// * `a`, `b` - Input tensor storage
    /// * `shape_a`, `shape_b` - Tensor shapes
    /// * `strides_a`, `strides_b` - Tensor strides (for non-contiguous support)
    /// * `modes_a`, `modes_b` - Index labels for each tensor dimension
    /// * `shape_c`, `modes_c` - Output shape and index labels
    fn contract<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>;

    /// Contraction with argmax tracking for tropical backpropagation.
    ///
    /// Returns (result, argmax) where argmax contains the index that "won"
    /// the reduction at each output position.
    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>;

}

// CPU supports all Scalar types
impl<T: Scalar> BackendScalar<crate::backend::Cpu> for T {}

// CUDA supports f32, f64, and complex types via cuTENSOR
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f32 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f64 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::algebra::Complex32 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::algebra::Complex64 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f32> {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f64> {}
