# Quick Start

## Basic Tensor Operations

### Creating Tensors

```rust
use omeinsum::{Tensor, Cpu};

// From data with shape
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Check properties
assert_eq!(a.shape(), &[2, 3]);
assert_eq!(a.ndim(), 2);
assert_eq!(a.numel(), 6);
```

### Tensor Views

```rust
// Zero-copy transpose
let a_t = a.permute(&[1, 0]);
assert_eq!(a_t.shape(), &[3, 2]);

// Reshape (zero-copy when contiguous)
let a_flat = a.reshape(&[6]);
assert_eq!(a_flat.shape(), &[6]);
```

## Einsum Operations

### Matrix Multiplication

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;

let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// C[i,k] = Σ_j A[i,j] × B[j,k]
let c = einsum::<Standard<f32>, _, _>(
    &[&a, &b],
    &[&[0, 1], &[1, 2]],
    &[0, 2],
);
```

### Tropical Operations

```rust
use omeinsum::algebra::MaxPlus;

// C[i,k] = max_j (A[i,j] + B[j,k])
let c = einsum::<MaxPlus<f32>, _, _>(
    &[&a, &b],
    &[&[0, 1], &[1, 2]],
    &[0, 2],
);
```

## Using the Einsum Builder

For more control over contraction:

```rust
use omeinsum::{Einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;
use std::collections::HashMap;

let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();

let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2]],  // ij, jk
    vec![0, 2],                     // -> ik
    sizes,
);

// Optimize contraction order
ein.optimize_greedy();

// Execute
let c = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
```

## Next Steps

- [Algebra Types](./algebras.md) - Learn about different semirings
- [Einsum API](./einsum-api.md) - Advanced einsum usage
- [Optimization](./optimization.md) - Contraction order optimization
