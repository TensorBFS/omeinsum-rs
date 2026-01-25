# omeinsum-rs

[![CI](https://github.com/TensorBFS/omeinsum-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorBFS/omeinsum-rs/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/TensorBFS/omeinsum-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/omeinsum-rs)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tensorbfs.github.io/omeinsum-rs/)

Einstein summation for tropical and standard tensor networks in Rust. Inspired by [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

## Features

- **Multiple Algebras**: Standard arithmetic, MaxPlus, MinPlus, MaxMul semirings
- **Contraction Optimization**: Uses [omeco](https://github.com/GiggleLiu/omeco) for optimal contraction order
- **Backpropagation Support**: Argmax tracking for tropical gradient computation
- **Flexible Tensors**: Stride-based views with zero-copy permute/reshape

## Installation

```toml
[dependencies]
omeinsum = "0.1"
```

## Quick Start

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::{Standard, MaxPlus};

// Create tensors
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Standard matrix multiplication: C[i,k] = Σ_j A[i,j] × B[j,k]
let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

// Tropical matrix multiplication: C[i,k] = max_j (A[i,j] + B[j,k])
let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
```

## Documentation

📖 **[User Guide](https://tensorbfs.github.io/omeinsum-rs/)** - Installation, tutorials, examples

📚 **[API Reference](https://tensorbfs.github.io/omeinsum-rs/api/omeinsum/)** - Rust API documentation

## Algebras

| Type | ⊕ | ⊗ | Use Case |
|------|---|---|----------|
| `Standard<T>` | + | × | Normal arithmetic |
| `MaxPlus<T>` | max | + | Longest path, Viterbi |
| `MinPlus<T>` | min | + | Shortest path |
| `MaxMul<T>` | max | × | Max probability |

## Contraction Optimization

```rust
use omeinsum::Einsum;
use std::collections::HashMap;

// A[i,j] × B[j,k] × C[k,l] → D[i,l]
let sizes: HashMap<usize, usize> = [(0, 10), (1, 20), (2, 30), (3, 40)].into();

let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2], vec![2, 3]],
    vec![0, 3],
    sizes,
);

// Optimize contraction order (critical for performance!)
ein.optimize_greedy();  // Fast O(n²) algorithm
// ein.optimize_treesa();  // Better for large networks

let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);
```

## Related Projects

- [tropical-gemm](https://github.com/TensorBFS/tropical-gemm) - High-performance tropical GEMM
- [omeco](https://github.com/GiggleLiu/omeco) - Contraction order optimization
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) - Julia einsum library

## License

MIT
