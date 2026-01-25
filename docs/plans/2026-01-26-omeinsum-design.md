# OMEinsum-rs Design Document

**Date:** 2026-01-26
**Status:** Implementation Started
**Package:** `omeinsum`
**Related:** [tropical-gemm#21](https://github.com/TensorBFS/tropical-gemm/issues/21), [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)

## Summary

`omeinsum` is a Rust implementation of Einstein summation with support for both standard and tropical algebras. It provides:

- **Algebra-agnostic tensor operations** via the `Semiring`/`Algebra` traits
- **Generic data types** - works with `f32`, `f64`, `i32`, `i64`, etc.
- **Backpropagation support** for both tropical and standard operations
- **Contraction order optimization** via [omeco](https://github.com/GiggleLiu/omeco)
- **CPU and CUDA backends** (CUDA optional)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User API                                │
│   einsum::<MaxPlus<f32>>(&tensors, &ixs, &iy) → Tensor         │
│   einsum_with_grad::<Standard<f64>>(...) → (Tensor, Gradient)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Einsum Engine                              │
│   Einsum::new(ixs, iy, size_dict)                              │
│       .optimize_greedy() / .optimize_treesa()                  │
│       .execute::<Algebra, Scalar, Backend>(&tensors)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Tensor::contract_binary::<A>()                    │
│   1. Classify indices (batch, left, right, contracted)         │
│   2. Permute (zero-cost view)                                  │
│   3. Reshape (zero-cost if contiguous)                         │
│   4. GEMM (dispatches to algebra-specific kernel)              │
│   5. Reshape + permute result                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌───────────────────────┐     ┌───────────────────────┐
│    Algebra Dispatch   │     │   Backend Dispatch    │
│  Standard<T>: (+, ×)  │     │  Cpu: Vec<T> storage  │
│  MaxPlus<T>: (max, +) │     │  Cuda: CudaSlice<T>   │
│  MinPlus<T>: (min, +) │     │       (optional)      │
│  MaxMul<T>:  (max, ×) │     └───────────────────────┘
└───────────────────────┘
```

## Core Traits

### Semiring Trait

The fundamental algebraic abstraction:

```rust
pub trait Semiring: Copy + Clone + Send + Sync + 'static {
    type Scalar: Scalar;

    fn zero() -> Self;                    // Additive identity
    fn one() -> Self;                     // Multiplicative identity
    fn add(self, rhs: Self) -> Self;      // ⊕
    fn mul(self, rhs: Self) -> Self;      // ⊗
    fn from_scalar(s: Self::Scalar) -> Self;
    fn to_scalar(self) -> Self::Scalar;
}
```

### Algebra Trait (extends Semiring for autodiff)

```rust
pub trait Algebra: Semiring {
    type Index: Copy + Default;

    // For argmax tracking (tropical backprop)
    fn add_with_argmax(self, self_idx: Index, rhs: Self, rhs_idx: Index) -> (Self, Index);

    // Backward passes
    fn add_backward(self, rhs: Self, grad_out: Scalar, winner_idx: Option<Index>)
        -> (Scalar, Scalar);
    fn mul_backward(self, rhs: Self, grad_out: Scalar) -> (Scalar, Scalar);

    fn needs_argmax() -> bool;
}
```

### Algebra Implementations

| Type | ⊕ (add) | ⊗ (mul) | Zero | One | `needs_argmax` |
|------|---------|---------|------|-----|----------------|
| `Standard<T>` | + | × | 0 | 1 | false |
| `MaxPlus<T>` | max | + | -∞ | 0 | true |
| `MinPlus<T>` | min | + | +∞ | 0 | true |
| `MaxMul<T>` | max | × | 0 | 1 | true |

### Backend Trait

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Storage<T: Scalar>: Storage<T>;

    fn name() -> &'static str;
    fn synchronize(&self);
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>;
    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T>;
    fn copy_strided<T: Scalar>(...) -> Self::Storage<T>;

    // Core GEMM operations
    fn gemm<A: Algebra>(&self, a, m, k, b, n) -> Storage<A::Scalar>;
    fn gemm_with_argmax<A: Algebra>(...) -> (Storage<A::Scalar>, Storage<u32>);
    fn gemm_backward_a<A: Algebra>(...) -> Storage<A::Scalar>;
    fn gemm_backward_b<A: Algebra>(...) -> Storage<A::Scalar>;
}
```

## Tensor Type

Stride-based tensor with zero-copy views:

```rust
pub struct Tensor<T: Scalar, B: Backend> {
    storage: Arc<B::Storage<T>>,  // Reference-counted, shared across views
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    backend: B,
}
```

### Key Operations

| Operation | Cost | Notes |
|-----------|------|-------|
| `permute(&[usize])` | O(1) | Just reorders shape/strides |
| `reshape(&[usize])` | O(1) if contiguous | Triggers copy if non-contiguous |
| `contiguous()` | O(n) if non-contiguous | No-op if already contiguous |
| `gemm::<A>(&other)` | O(m×k×n) | Ensures contiguous, then GEMM |
| `contract_binary::<A>(...)` | O(GEMM) | Permute→reshape→GEMM→reshape→permute |

### GEMM with Algebra Dispatch

```rust
impl<T: Scalar, B: Backend> Tensor<T, B> {
    pub fn gemm<A: Algebra<Scalar = T>>(&self, other: &Self) -> Self {
        let a = self.contiguous();
        let b = other.contiguous();
        let c_storage = self.backend.gemm::<A>(&a.storage, m, k, &b.storage, n);
        Self::from_raw(c_storage, ...)
    }
}
```

## Einsum Engine

Integration with omeco for contraction order optimization:

```rust
pub struct Einsum<L: Label = usize> {
    ixs: Vec<Vec<L>>,           // Input indices
    iy: Vec<L>,                  // Output indices
    size_dict: HashMap<L, usize>,
    optimized: Option<NestedEinsum<L>>,
}

impl Einsum<usize> {
    pub fn optimize_greedy(&mut self) -> &mut Self;
    pub fn optimize_treesa(&mut self) -> &mut Self;
    pub fn execute<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>;
    pub fn execute_with_argmax<A, T, B>(...) -> (Tensor<T, B>, Vec<Tensor<u32, B>>);
}
```

## Backpropagation

### Standard Arithmetic

- **Add backward**: Both inputs get `grad_out`
- **Mul backward**: `grad_a = grad_out × b`, `grad_b = grad_out × a`

### Tropical Arithmetic

- **Add backward**: Only the "winner" (max/min) gets `grad_out`
- **Mul backward**: Both inputs get `grad_out` (since tropical mul is +)

Argmax tracking during forward pass enables efficient backward computation.

## Crate Structure

```
omeinsum-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── algebra/
│   │   ├── mod.rs
│   │   ├── semiring.rs      # Semiring + Algebra traits
│   │   ├── standard.rs      # Standard<T>
│   │   └── tropical.rs      # MaxPlus, MinPlus, MaxMul
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── traits.rs        # Backend + Storage traits
│   │   ├── cpu.rs           # Cpu backend
│   │   └── cuda.rs          # Cuda backend (optional)
│   ├── tensor/
│   │   ├── mod.rs           # Tensor<T, B>
│   │   ├── ops.rs           # gemm, contract_binary
│   │   └── view.rs          # TensorView
│   └── einsum/
│       ├── mod.rs           # einsum(), einsum_with_grad()
│       ├── engine.rs        # Einsum struct
│       └── builder.rs       # EinBuilder
├── docs/plans/
├── tests/
└── examples/
```

## Dependencies

```toml
[dependencies]
num-traits = "0.2"
bytemuck = { version = "1.14", features = ["derive"] }
omeco = { path = "../omeco" }

[dependencies.tropical-gemm]
path = "../tropical-gemm"
optional = true

[dependencies.tropical-gemm-cuda]
path = "../tropical-gemm/crates/tropical-gemm-cuda"
optional = true

[features]
default = ["parallel", "tropical-kernels"]
parallel = ["rayon"]
tropical-kernels = ["tropical-gemm"]
cuda = ["tropical-gemm-cuda", "cudarc", "tropical-kernels"]
```

## Example Usage

```rust
use omeinsum::{Tensor, Einsum, einsum};
use omeinsum::algebra::{Standard, MaxPlus, MinPlus};
use omeinsum::backend::Cpu;

// Standard matrix multiplication
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// C[i,k] = Σ_j A[i,j] × B[j,k]
let c_standard = a.gemm::<Standard<f32>>(&b);

// C[i,k] = max_j (A[i,j] + B[j,k])
let c_tropical = a.gemm::<MaxPlus<f32>>(&b);

// Einsum with optimization
let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2], vec![2, 3]],  // A[i,j], B[j,k], C[k,l]
    vec![0, 3],                                  // D[i,l]
    [(0, 10), (1, 20), (2, 30), (3, 40)].into(),
);
ein.optimize_greedy();
let result = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a, &b, &c]);

// With gradient computation
let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&tensors, &ixs, &iy);
let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &tensors);
```

## Implementation Status

### Completed
- [x] `Semiring` and `Algebra` traits
- [x] `Standard<T>`, `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>` implementations
- [x] `Backend` and `Storage` traits
- [x] `Cpu` backend with generic GEMM
- [x] `Tensor<T, B>` with permute, reshape, contiguous
- [x] `contract_binary` with reshape-to-GEMM
- [x] `Einsum` engine with omeco integration
- [x] `EinBuilder` for fluent construction

### Pending
- [ ] Integration with `tropical-gemm` optimized kernels
- [ ] CUDA backend
- [ ] Batched GEMM in contract_binary
- [ ] Full backward pass implementation
- [ ] Python bindings
- [ ] Comprehensive tests and benchmarks

## Open Questions

1. **BLAS dispatch for Standard<T>**: Should we use BLAS/cuBLAS for standard arithmetic instead of generic loops?

2. **Batched contraction**: Current implementation doesn't handle batch dimensions in contract_binary. Need to implement batched GEMM path.

3. **Memory pool**: Should we add a memory pool for temporary allocations during contraction?

4. **Sliced einsum**: omeco supports sliced contraction for memory-constrained execution. Priority for v1?

## References

- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) - Julia reference
- [omeco](https://github.com/GiggleLiu/omeco) - Contraction order optimization
- [tropical-gemm](https://github.com/TensorBFS/tropical-gemm) - Optimized tropical kernels
- [opt_einsum](https://github.com/dgasmith/opt_einsum) - Python einsum optimization
