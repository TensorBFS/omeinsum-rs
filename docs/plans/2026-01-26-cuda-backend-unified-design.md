# CUDA Backend Unified Design

> **Status:** Design Document

## Goal

Implement `Backend` trait for `Cuda` so that `Tensor<T, Cuda>` works with the same API as `Tensor<T, Cpu>`.

## Current State

- `Cuda` struct provides low-level `contract` method for cuTENSOR operations
- `CudaStorage<T>` wraps `CudaSlice<T>` but doesn't implement `Storage<T>`
- `CudaComplex<T>` wrapper handles complex types on CUDA
- Tests work via direct `cuda.contract()` calls

## Design Challenge

The `Backend` trait requires:
```rust
type Storage<T: Scalar>: Storage<T>;  // where Scalar: bytemuck::Pod
```

CUDA requires:
```rust
T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits
```

These trait sets don't overlap for all types. Specifically:
- `f32`, `f64`: satisfy both
- `Complex32`, `Complex64`: satisfy `Scalar` but not cudarc traits (orphan rule)

## Solution: CudaScalar Marker Trait

Create a marker trait that unifies the requirements:

```rust
/// Types that can be used with CUDA storage.
/// Must satisfy both Scalar and cudarc requirements.
pub trait CudaScalar: Scalar + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits {}

impl CudaScalar for f32 {}
impl CudaScalar for f64 {}
impl CudaScalar for CudaComplex<f32> {}
impl CudaScalar for CudaComplex<f64> {}
```

## Implementation Steps

### Step 1: Make CudaStorage implement Storage

```rust
impl<T: CudaScalar> Storage<T> for CudaStorage<T> {
    fn len(&self) -> usize { self.slice.len() }
    fn get(&self, index: usize) -> T { /* dtoh single element */ }
    fn set(&mut self, index: usize, value: T) { /* htod single element */ }
    fn to_vec(&self) -> Vec<T> { /* existing implementation */ }
    fn from_slice(data: &[T]) -> Self { /* needs default device - tricky */ }
    fn zeros(len: usize) -> Self { /* needs default device - tricky */ }
}
```

Issue: `from_slice` and `zeros` need a device reference, but `Storage::from_slice` is a standalone function.

### Step 2: Backend Implementation

```rust
impl Backend for Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;  // Only valid for CudaScalar types

    fn alloc<T: Scalar>(&self, len: usize) -> CudaStorage<T> { ... }
    fn from_slice<T: Scalar>(&self, data: &[T]) -> CudaStorage<T> { ... }
    fn copy_strided<T: Scalar>(...) -> CudaStorage<T> { ... }
    
    fn gemm<A: Algebra>(&self, a, m, k, b, n) -> CudaStorage<A::Scalar> {
        // Use cuTENSOR for Standard, cuBLAS if available
        // For complex, use CudaComplex wrapper internally
    }
}
```

### Step 3: Complex Type Handling

For `Tensor<Complex64, Cuda>`:
1. User creates tensor with `Complex64` data
2. Internally convert to `CudaComplex<f64>` for GPU operations
3. Convert back when returning to user

OR: Document that users should use `Tensor<CudaComplex<f64>, Cuda>` for CUDA.

## Alternative: Separate CUDA API

Keep the current design where:
- `Tensor<T, Cpu>` uses the `Backend` trait
- CUDA uses direct `Cuda::contract()` API

Pros:
- Simpler implementation
- cuTENSOR's API is inherently different (mode-based contraction)

Cons:
- No unified API
- Users need different code for CPU vs CUDA

## Recommendation

For v1, use the **Alternative** approach (separate API) because:
1. cuTENSOR's contraction API is fundamentally different from GEMM
2. The Backend trait is optimized for reshape-to-GEMM path
3. Implementing full Backend would require significant work

For v2, consider a higher-level unified API that abstracts over both backends.

## Files to Modify

If implementing unified design:
- `src/backend/cuda/mod.rs` - Add `Backend` impl
- `src/backend/cuda/storage.rs` - Add `Storage` impl
- `src/backend/traits.rs` - Consider `CudaScalar` trait
- `src/algebra/mod.rs` - Ensure `CudaComplex` is a `Scalar`
