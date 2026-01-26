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

    /// General matrix multiplication.
    ///
    /// Computes C = A ⊗ B where ⊗ is the semiring multiplication
    /// and the reduction uses semiring addition.
    ///
    /// # Arguments
    /// * `a` - Left matrix, row-major, shape [m, k]
    /// * `b` - Right matrix, row-major, shape [k, n]
    /// * `m`, `k`, `n` - Matrix dimensions
    ///
    /// # Returns
    /// Result matrix C, row-major, shape [m, n]
    fn gemm<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        m: usize,
        k: usize,
        b: &Self::Storage<A::Scalar>,
        n: usize,
    ) -> Self::Storage<A::Scalar>;

    /// GEMM with argmax tracking for tropical backpropagation.
    ///
    /// Returns (result, argmax) where argmax[i, j] is the k index
    /// that "won" the reduction for element [i, j].
    fn gemm_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        m: usize,
        k: usize,
        b: &Self::Storage<A::Scalar>,
        n: usize,
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>);

    /// Backward pass for GEMM w.r.t. A.
    ///
    /// Given dC (gradient of output), computes dA (gradient of A).
    fn gemm_backward_a<A: Algebra>(
        &self,
        grad_c: &Self::Storage<A::Scalar>,
        argmax: &Self::Storage<u32>,
        b: &Self::Storage<A::Scalar>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage<A::Scalar>;

    /// Backward pass for GEMM w.r.t. B.
    fn gemm_backward_b<A: Algebra>(
        &self,
        grad_c: &Self::Storage<A::Scalar>,
        argmax: &Self::Storage<u32>,
        a: &Self::Storage<A::Scalar>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage<A::Scalar>;

    /// Batched GEMM: `C[b] = A[b] @ B[b]` for each batch.
    ///
    /// # Arguments
    /// * `a` - Left matrices, row-major, shape [batch_size, m, k]
    /// * `b` - Right matrices, row-major, shape [batch_size, k, n]
    /// * `batch_size` - Number of batches
    /// * `m`, `k`, `n` - Matrix dimensions
    ///
    /// # Returns
    /// Result matrices C, row-major, shape [batch_size, m, n]
    fn gemm_batched<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        batch_size: usize,
        m: usize,
        k: usize,
        b: &Self::Storage<A::Scalar>,
        n: usize,
    ) -> Self::Storage<A::Scalar>;

    /// Batched GEMM with argmax tracking.
    ///
    /// Returns (result, argmax) where argmax[b, i, j] is the k index
    /// that "won" the reduction for element [b, i, j].
    fn gemm_batched_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        batch_size: usize,
        m: usize,
        k: usize,
        b: &Self::Storage<A::Scalar>,
        n: usize,
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>);
}

// CPU supports all Scalar types
impl<T: Scalar> BackendScalar<crate::backend::Cpu> for T {}
