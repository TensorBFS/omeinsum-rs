# omeinsum-rs

Rust implementation of OMEinsum.jl - Einstein summation for tropical and standard tensor networks.

## Project Structure

```
src/
├── lib.rs              # Public API exports
├── algebra/            # Algebra types (Standard, MaxPlus, MinPlus, MaxMul)
│   ├── mod.rs
│   ├── semiring.rs     # Semiring and Algebra traits
│   ├── standard.rs     # Standard (+, ×) algebra
│   └── tropical.rs     # Tropical algebras (feature-gated)
├── backend/            # Hardware backends
│   ├── mod.rs
│   ├── traits.rs       # Backend and Storage traits
│   ├── cpu/            # CPU backend (always enabled)
│   └── cuda/           # CUDA backend (optional feature)
├── tensor/             # Tensor type with stride-based views
│   ├── mod.rs
│   ├── ops.rs          # contract_binary methods
│   └── view.rs         # TensorView
└── einsum/             # Einsum engine
    ├── mod.rs          # einsum, einsum_with_grad, cost_and_gradient
    ├── engine.rs       # Einsum struct, optimization, execution
    ├── backward.rs     # Gradient computation, CacheTree
    └── builder.rs      # EinBuilder fluent API
```

## Key Conventions

### Column-Major Layout
All tensors use column-major (Fortran) ordering. Element `[i,j,k]` in shape `[m,n,p]` is at position `i + j*m + k*m*n`.

### Index Labels
Einsum indices are `usize` integers, not characters. Example: `A[0,1] @ B[1,2] -> C[0,2]` for matmul.

### Algebra System
- `Semiring` trait: zero, one, add (⊕), mul (⊗)
- `Algebra` trait: extends Semiring with gradient support (`needs_argmax`, `add_backward`, `mul_backward`)

### Gradient Computation
- Standard algebra: index-exchange trick (swap input/output indices)
- Tropical algebras: argmax routing (gradient flows through winner only)

## Testing

```bash
# Run all tests (requires tropical feature for full coverage)
cargo test --features tropical

# Run with coverage
cargo llvm-cov --features tropical
```

Test coverage target: >95%

## Features

- `tropical`: Enable MaxPlus, MinPlus, MaxMul algebras with optimized SIMD kernels
- `cuda`: CUDA backend (requires cuTENSOR 2.0+)
- `parallel`: Parallel execution via rayon

## Reference

Julia implementation: [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)

Key files for reference:
- `~/.julia/dev/OMEinsum/src/bp.jl` - cost_and_gradient, CacheTree, back_propagate
- `~/.julia/dev/OMEinsum/src/einsum.jl` - einsum_grad (backward rule)
