# Getting Started

This chapter covers installation and basic usage of omeinsum-rs.

## Prerequisites

- Rust 1.70 or later
- Cargo package manager

## Quick Example

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;

// Matrix multiplication: C = A × B
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

let c = einsum::<Standard<f32>, _, _>(
    &[&a, &b],
    &[&[0, 1], &[1, 2]],  // A[i,j], B[j,k]
    &[0, 2],               // -> C[i,k]
);

assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
```

Continue to [Installation](./installation.md) for detailed setup instructions.
