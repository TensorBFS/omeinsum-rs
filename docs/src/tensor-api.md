# Tensor API

The `Tensor<T, B>` type provides a flexible, stride-based tensor implementation.

## Creating Tensors

### From Data

```rust
use omeinsum::{Tensor, Cpu};

// Create from slice with shape
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Row-major layout:
// [[1, 2, 3],
//  [4, 5, 6]]
```

### Zeros and Ones

```rust
let zeros = Tensor::<f32, Cpu>::zeros(&[3, 4]);
let ones = Tensor::<f32, Cpu>::ones(&[3, 4]);
```

## Properties

```rust
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

t.shape()    // &[2, 3]
t.strides()  // &[3, 1] for row-major
t.ndim()     // 2
t.numel()    // 6
```

## Views and Transformations

### Permute (Transpose)

Zero-copy axis reordering:

```rust
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Transpose: swap axes 0 and 1
let t_t = t.permute(&[1, 0]);
assert_eq!(t_t.shape(), &[3, 2]);

// 3D example: (batch, height, width) -> (batch, width, height)
let img = Tensor::<f32, Cpu>::zeros(&[10, 28, 28]);
let img_t = img.permute(&[0, 2, 1]);
assert_eq!(img_t.shape(), &[10, 28, 28]);
```

### Reshape

```rust
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Flatten
let flat = t.reshape(&[6]);

// Reshape to different dimensions
let reshaped = t.reshape(&[3, 2]);
```

### Contiguous

Convert non-contiguous views to contiguous storage:

```rust
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
let t_t = t.permute(&[1, 0]);  // Non-contiguous after transpose

let t_contig = t_t.contiguous();  // Copy to contiguous memory
assert!(t_contig.is_contiguous());
```

## Matrix Operations

### GEMM (General Matrix Multiplication)

```rust
use omeinsum::algebra::{Standard, MaxPlus};

let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Standard matrix multiplication
let c = a.gemm::<Standard<f32>>(&b);

// Tropical matrix multiplication
let c_trop = a.gemm::<MaxPlus<f32>>(&b);
```

### Binary Contraction

General tensor contraction:

```rust
// A[i,j,k] × B[j,k,l] -> C[i,l]
let a = Tensor::<f32, Cpu>::zeros(&[2, 3, 4]);
let b = Tensor::<f32, Cpu>::zeros(&[3, 4, 5]);

let c = a.contract_binary::<Standard<f32>>(
    &b,
    &[0, 1, 2],  // A's indices: i, j, k
    &[1, 2, 3],  // B's indices: j, k, l
    &[0, 3],     // Output: i, l
);

assert_eq!(c.shape(), &[2, 5]);
```

## Data Access

```rust
let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Convert to Vec
let data = t.to_vec();  // [1.0, 2.0, 3.0, 4.0]
```
