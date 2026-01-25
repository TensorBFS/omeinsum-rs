# Introduction

**omeinsum-rs** is a Rust library for efficient tensor network contractions supporting both standard and tropical (semiring) algebras. It provides a unified interface for einsum operations with automatic contraction order optimization.

## What is Einsum?

Einstein summation (einsum) is a compact notation for expressing tensor operations. Instead of writing explicit loops, you specify index labels:

```
C[i,k] = Σ_j A[i,j] × B[j,k]    # Matrix multiplication
```

In einsum notation: `ij,jk->ik`

## What are Tropical Algebras?

Tropical algebras replace standard arithmetic with alternative operations:

| Algebra | Addition (⊕) | Multiplication (⊗) | Use Case |
|---------|--------------|-------------------|----------|
| Standard | + | × | Normal arithmetic |
| MaxPlus | max | + | Longest path, Viterbi |
| MinPlus | min | + | Shortest path |
| MaxMul | max | × | Max probability |

## Key Features

- **Multiple Algebras**: Standard arithmetic, MaxPlus, MinPlus, MaxMul
- **Contraction Optimization**: Uses [omeco](https://github.com/GiggleLiu/omeco) for optimal contraction order
- **Backpropagation Support**: Argmax tracking for tropical gradient computation
- **Flexible Tensors**: Stride-based views with zero-copy permute/reshape
- **Backend Abstraction**: CPU now, GPU planned

## Example

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::MaxPlus;

// Create tensors
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Tropical matrix multiplication: C[i,k] = max_j (A[i,j] + B[j,k])
let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
```

## Relationship to OMEinsum.jl

This library is inspired by [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), bringing its powerful tensor contraction capabilities to Rust with support for tropical algebras from [tropical-gemm](https://github.com/TensorBFS/tropical-gemm).
