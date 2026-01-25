# Architecture

This chapter describes the internal architecture of omeinsum-rs.

## Module Structure

```
omeinsum/
├── algebra/          # Semiring and algebra traits
│   ├── mod.rs
│   ├── semiring.rs   # Semiring, Algebra, Scalar traits
│   ├── standard.rs   # Standard<T> implementation
│   └── tropical.rs   # MaxPlus, MinPlus, MaxMul
├── backend/          # Execution backends
│   ├── mod.rs
│   ├── traits.rs     # Backend, Storage traits
│   └── cpu.rs        # CPU backend implementation
├── tensor/           # Tensor type
│   ├── mod.rs        # Tensor<T, B> definition
│   ├── view.rs       # View operations (permute, reshape)
│   └── ops.rs        # GEMM, contract_binary
├── einsum/           # Einsum engine
│   ├── mod.rs        # einsum(), einsum_with_grad()
│   ├── engine.rs     # Einsum struct, optimization
│   └── builder.rs    # EinBuilder (planned)
└── lib.rs            # Public API exports
```

## Core Abstractions

### Scalar Trait

Base trait for numeric types:

```rust
pub trait Scalar: Copy + Clone + Default + Send + Sync + 'static {
    fn neg_infinity() -> Self;
    fn infinity() -> Self;
}
```

### Semiring Trait

Defines the algebraic structure:

```rust
pub trait Semiring: Copy + Clone + Send + Sync + 'static {
    type Scalar: Scalar;
    fn zero() -> Self;
    fn one() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn from_scalar(s: Self::Scalar) -> Self;
    fn to_scalar(self) -> Self::Scalar;
}
```

### Algebra Trait

Extends Semiring with backpropagation support:

```rust
pub trait Algebra: Semiring {
    type Index: Copy + Clone + Default + Send + Sync;

    fn add_with_argmax(self, self_idx: Self::Index, rhs: Self, rhs_idx: Self::Index)
        -> (Self, Self::Index);

    fn add_backward(self, rhs: Self, grad_out: Self::Scalar, winner_idx: Option<Self::Index>)
        -> (Self::Scalar, Self::Scalar);

    fn mul_backward(self, rhs: Self, grad_out: Self::Scalar)
        -> (Self::Scalar, Self::Scalar);

    fn needs_argmax() -> bool;
}
```

### Backend Trait

Abstracts execution hardware:

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Storage<T: Scalar>: Storage<T>;

    fn gemm<A: Algebra>(&self, a: &Self::Storage<A::Scalar>, m: usize, k: usize,
                        b: &Self::Storage<A::Scalar>, n: usize) -> Self::Storage<A::Scalar>;

    fn gemm_with_argmax<A: Algebra<Index = u32>>(&self, ...)
        -> (Self::Storage<A::Scalar>, Self::Storage<u32>);
}
```

## Tensor Implementation

### Stride-Based Storage

Tensors use stride-based views for efficient transformations:

```rust
pub struct Tensor<T: Scalar, B: Backend> {
    storage: Arc<B::Storage<T>>,  // Shared storage
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    backend: B,
}
```

### Zero-Copy Operations

- **Permute**: Reorders strides, no data copy
- **Reshape**: Updates shape if contiguous, otherwise copies
- **Contiguous**: Copies if non-contiguous

## Contraction Strategy

Binary contraction uses reshape-to-GEMM:

1. Classify indices: batch, left-only, right-only, contracted
2. Permute tensors to [batch, left/right, contracted]
3. Reshape to 2D matrices
4. Execute GEMM
5. Reshape and permute to output

This leverages optimized GEMM implementations for all algebras.

## Optimization Integration

Uses omeco for contraction order:

```rust
use omeco::{EinCode, GreedyMethod, TreeSA, optimize_code};

let code = EinCode::new(ixs, iy);
let tree = optimize_code(&code, &size_dict, &GreedyMethod::new(0.0, 0.0));
```

The resulting `NestedEinsum` tree is executed recursively.
