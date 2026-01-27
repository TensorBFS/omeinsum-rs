# Unified Backend Design

> **Status:** Approved Design
> **Date:** 2026-01-27

## Goal

Implement a unified `Backend` trait so that `Tensor<T, Cuda>` works with the same `einsum()` API as `Tensor<T, Cpu>`.

## Key Insight

The `Backend` trait should be defined at the **contraction level**, not the GEMM level:

- **CPU:** Implements `contract()` internally via permute→reshape→GEMM→reshape
- **CUDA:** Implements `contract()` directly via cuTENSOR (no reshape needed)

cuTENSOR handles arbitrary tensor contractions natively, while BLAS only understands 2D matrices. By defining the trait at the contraction level, each backend can use its optimal strategy.

## Design

### 1. New Backend Trait

```rust
/// Marker trait for scalar types supported by a specific backend.
pub trait BackendScalar<B: Backend>: Scalar {}

// CPU supports all Scalar types
impl<T: Scalar> BackendScalar<Cpu> for T {}

// CUDA only supports cuTENSOR-compatible types
#[cfg(feature = "cuda")]
impl BackendScalar<Cuda> for f32 {}
#[cfg(feature = "cuda")]
impl BackendScalar<Cuda> for f64 {}
#[cfg(feature = "cuda")]
impl BackendScalar<Cuda> for CudaComplex<f32> {}
#[cfg(feature = "cuda")]
impl BackendScalar<Cuda> for CudaComplex<f64> {}

/// Backend trait for tensor execution.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Storage type for this backend.
    type Storage<T: Scalar>: Storage<T>;

    /// Device/context type.
    type Device: Default + Clone;

    /// Backend name for debugging.
    fn name() -> &'static str;

    /// Get the device.
    fn device(&self) -> &Self::Device;

    /// Allocate storage.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>;

    /// Create storage from slice.
    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T>;

    /// Copy strided data to contiguous storage.
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
    ) -> Self::Storage<A::Scalar>;

    /// Contraction with argmax tracking (for tropical backprop).
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
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>);
}
```

### 2. CPU Backend Implementation

```rust
impl Backend for Cpu {
    type Storage<T: Scalar> = Vec<T>;
    type Device = ();

    fn contract<A: Algebra>(
        &self,
        a: &Vec<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Vec<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Vec<A::Scalar> {
        // 1. Make inputs contiguous if needed
        // 2. Classify indices into batch/left/right/contracted
        // 3. Permute and reshape to matrices
        // 4. Call internal GEMM (faer-based)
        // 5. Reshape and permute to output
    }
}
```

The reshape→GEMM→reshape logic moves from `Tensor::contract_binary()` into the CPU backend.

### 3. CUDA Backend Implementation

```rust
impl Backend for Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;
    type Device = Arc<CudaDevice>;

    fn contract<A: Algebra>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> CudaStorage<A::Scalar>
    where
        A::Scalar: CutensorType,
    {
        // Direct cuTENSOR call - no reshape needed!
        // cuTENSOR handles arbitrary shapes/strides/modes directly
    }
}
```

### 4. Simplified Tensor API

```rust
impl<T: Scalar, B: Backend> Tensor<T, B>
where
    T: BackendScalar<B>,
{
    pub fn contract_binary<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        modes_a: &[i32],
        modes_b: &[i32],
        modes_c: &[i32],
    ) -> Self {
        let shape_c = compute_output_shape(...);

        // Thin wrapper - delegate to backend
        let storage_c = self.backend.contract::<A>(
            self.storage.as_ref(), self.shape(), self.strides(), modes_a,
            other.storage.as_ref(), other.shape(), other.strides(), modes_b,
            &shape_c, modes_c,
        );

        Self::from_storage(storage_c, &shape_c, self.backend.clone())
    }
}
```

### 5. Unified einsum API

```rust
pub fn einsum<A, T, B>(
    tensors: &[&Tensor<T, B>],
    indices: &[&[usize]],
    output: &[usize],
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    // Works identically for CPU and CUDA
}
```

## File Organization

```
src/backend/
├── mod.rs              # Re-exports
├── traits.rs           # Backend, Storage, BackendScalar traits
├── cpu/
│   ├── mod.rs          # Cpu struct, Backend impl
│   ├── gemm.rs         # Internal GEMM (faer-based)
│   └── contract.rs     # Permute→reshape→GEMM→reshape logic
└── cuda/
    ├── mod.rs          # Cuda struct, Backend impl
    ├── storage.rs      # CudaStorage
    └── cutensor/       # cuTENSOR FFI (unchanged)
```

## Migration Steps

| Step | Change | Risk |
|------|--------|------|
| 1 | Add `BackendScalar<B>` trait | None (additive) |
| 2 | Add `Backend::contract()` method | None (additive) |
| 3 | Implement `contract()` for `Cpu` | Low (extract existing logic) |
| 4 | Implement `contract()` for `Cuda` | Low (wrap existing code) |
| 5 | Simplify `Tensor::contract_binary()` | Medium (refactor) |
| 6 | Remove old `Backend::gemm*` methods | Medium (breaking) |
| 7 | Add `BackendScalar<B>` bounds to public API | Low (mostly inferred) |

## Type Safety

The `BackendScalar<B>` marker trait provides compile-time safety:

- `Tensor<f64, Cpu>` ✓ compiles
- `Tensor<f64, Cuda>` ✓ compiles
- `Tensor<i32, Cpu>` ✓ compiles
- `Tensor<i32, Cuda>` ✗ compile error (i32 doesn't implement `BackendScalar<Cuda>`)

This follows Burn's approach of using marker traits for GPU element constraints.

## Testing Strategy

1. **Correctness:** Verify CPU and CUDA produce identical results
2. **Type safety:** Verify unsupported types don't compile for CUDA
3. **Algebra:** Test Standard, MaxPlus, MinPlus on both backends
4. **Edge cases:** Empty tensors, scalars, large tensors

## References

- [Burn Framework](https://github.com/tracel-ai/burn) - Backend trait design patterns
- [cuTENSOR Documentation](https://docs.nvidia.com/cuda/cutensor/) - Mode-based contraction API
