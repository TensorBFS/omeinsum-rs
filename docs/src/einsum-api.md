# Einsum API

The einsum API provides a high-level interface for tensor network contractions.

## Quick Einsum

The simplest way to perform einsum:

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;

let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Matrix multiplication: ij,jk->ik
let c = einsum::<Standard<f32>, _, _>(
    &[&a, &b],           // Input tensors
    &[&[0, 1], &[1, 2]], // Index labels: A[0,1], B[1,2]
    &[0, 2],             // Output labels: C[0,2]
);
```

## Index Labels

Indices are represented as `usize` values. Matching indices indicate contraction:

| Operation | Inputs | Labels | Output |
|-----------|--------|--------|--------|
| Matrix multiply | A[m,k], B[k,n] | `[[0,1], [1,2]]` | `[0,2]` → C[m,n] |
| Batch matmul | A[b,m,k], B[b,k,n] | `[[0,1,2], [0,2,3]]` | `[0,1,3]` → C[b,m,n] |
| Outer product | A[m], B[n] | `[[0], [1]]` | `[0,1]` → C[m,n] |
| Trace | A[n,n] | `[[0,0]]` | `[]` → scalar |
| Sum | A[m,n] | `[[0,1]]` | `[]` → scalar |

## Einsum Struct

For more control, use the `Einsum` struct directly:

```rust
use omeinsum::{Einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;
use std::collections::HashMap;

// Define size dictionary
let sizes: HashMap<usize, usize> = [
    (0, 10),  // i: 10
    (1, 20),  // j: 20
    (2, 30),  // k: 30
].into();

// Create einsum specification
let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2]],  // A[i,j], B[j,k]
    vec![0, 2],                     // -> C[i,k]
    sizes,
);

// Check the einsum code
let code = ein.code();
println!("Einsum: {:?}", code);
```

## Contraction Optimization

### Greedy Algorithm

Fast O(n²) algorithm, good for most cases:

```rust
let mut ein = Einsum::new(/* ... */);
ein.optimize_greedy();

assert!(ein.is_optimized());
```

### Simulated Annealing

Slower but finds better orderings for complex networks:

```rust
let mut ein = Einsum::new(/* ... */);
ein.optimize_treesa();
```

### Inspect Contraction Tree

```rust
if let Some(tree) = ein.contraction_tree() {
    println!("Contraction tree: {:?}", tree);
}
```

## Chain Contraction Example

Contracting a chain of matrices:

```rust
use omeinsum::{Einsum, Tensor, Cpu};
use omeinsum::algebra::Standard;
use std::collections::HashMap;

// A[i,j] × B[j,k] × C[k,l] → D[i,l]
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let c = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

let sizes: HashMap<usize, usize> = [
    (0, 2), (1, 2), (2, 2), (3, 2)
].into();

let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2], vec![2, 3]],
    vec![0, 3],
    sizes,
);

ein.optimize_greedy();
let d = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);

assert_eq!(d.shape(), &[2, 2]);
```

## Einsum with Gradients

For backpropagation support:

```rust
use omeinsum::einsum_with_grad;
use omeinsum::algebra::MaxPlus;

let (result, gradient) = einsum_with_grad::<MaxPlus<f32>, _, _>(
    &[&a, &b],
    &[&[0, 1], &[1, 2]],
    &[0, 2],
);

// gradient can be used for backpropagation
// (full backward pass implementation in progress)
```
