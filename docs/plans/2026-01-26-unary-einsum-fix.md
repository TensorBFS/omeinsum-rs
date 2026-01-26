# Unary Einsum Operations Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix single-tensor einsum operations (trace, sum, diagonal, transpose, partial trace) to work through the convenience `einsum()` function.

**Architecture:** Replace specialized unary handlers with a unified naive loop that handles all unary operations uniformly by iterating over output indices and accumulating over contracted indices.

**Tech Stack:** Rust, existing Tensor/Algebra abstractions

---

## Background

### Problem

The convenience `einsum()` function always calls `optimize_greedy()`, which creates a contraction tree. For single-tensor operations, this creates a `Leaf` node that just clones the tensor without applying any transformation.

### Root Cause

In `execute_tree()`:
```rust
NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone()
```

This bypasses the `execute_unary` logic that handles trace/diagonal/sum.

### Issues Addressed

- #7 - Fix optimizer to handle single-tensor einsum operations
- #2 - Support unary transpose/permutation in einsum
- #3 - Support higher-dimensional partial trace
- #5 - Support gradients for outer product operations (fixed automatically)

---

## Design

### Approach: Naive Loop

Instead of separate code paths for trace, diagonal, sum, and permutation, use a unified loop:

1. **Classify indices**: Separate output (outer) indices from contracted (inner) indices
2. **Loop over outer**: Iterate over all output positions
3. **Loop over inner**: For each output position, accumulate over inner indices

This handles all cases uniformly:

| Operation | Input | Output | Outer | Inner |
|-----------|-------|--------|-------|-------|
| Identity | `ij` | `ij` | i,j | - |
| Transpose | `ij` | `ji` | j,i | - |
| Sum axis | `ij` | `i` | i | j |
| Sum all | `ij` | `` | - | i,j |
| Trace | `ii` | `` | - | i |
| Diagonal | `ii` | `i` | i | - |
| Partial trace 4D | `ijjk` | `ik` | i,k | j |

### Key Insight: Repeated Indices

For `ix = [0, 1, 1, 2]` (ijjk), positions 1 and 2 both map to index label `1`. When we compute the input position, both use the same `idx_values[&1]`. This automatically extracts the diagonal - only elements where both j positions match are accessed.

---

## Implementation

### Change 1: Leaf Handler in `execute_tree`

**File:** `src/einsum/engine.rs`

```rust
// Before (line ~264)
NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone(),

// After
NestedEinsum::Leaf { tensor_index } => {
    execute_unary_naive::<A, T, B>(
        tensors[*tensor_index],
        &self.ixs[*tensor_index],
        &self.iy,
        &self.size_dict,
    )
}
```

### Change 2: New Free Function `execute_unary_naive`

**File:** `src/einsum/engine.rs`

```rust
/// Execute unary einsum operation using naive loop.
/// Handles trace, diagonal, sum, permutation uniformly.
fn execute_unary_naive<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],                     // input indices
    iy: &[usize],                     // output indices
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T>,
    T: Scalar,
    B: Backend + Default,
{
    // 1. Classify indices
    let outer: &[usize] = iy;
    let outer_set: HashSet<usize> = outer.iter().copied().collect();
    // Collect inner indices deterministically, preserving the order from `ix`
    let mut inner_vec: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for i in ix.iter().copied().filter(|i| !outer_set.contains(i)) {
        if seen.insert(i) {
            inner_vec.push(i);
        }
    }

    // 2. Build output shape
    let out_shape: Vec<usize> = outer.iter()
        .map(|&idx| size_dict[&idx])
        .collect();
    let out_size = out_shape.iter().product::<usize>().max(1);

    // 3. Build inner ranges
    let inner_ranges: Vec<usize> = inner_vec.iter()
        .map(|&idx| size_dict[&idx])
        .collect();
    let inner_size = inner_ranges.iter().product::<usize>().max(1);

    // 4. Allocate output
    let mut out_data = vec![A::zero().to_scalar(); out_size];

    // 5. Loop over output positions
    for out_linear in 0..out_size {
        let out_multi = linear_to_multi(out_linear, &out_shape);

        // Map: outer index label -> value
        let mut idx_values: HashMap<usize, usize> = outer.iter()
            .zip(out_multi.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();

        // 6. Accumulate over inner indices
        let mut acc = A::zero();
        for inner_linear in 0..inner_size {
            let inner_multi = linear_to_multi(inner_linear, &inner_ranges);
            for (&idx, &val) in inner_vec.iter().zip(inner_multi.iter()) {
                idx_values.insert(idx, val);
            }

            // 7. Compute input position and accumulate
            let in_pos = compute_input_position(ix, &idx_values, tensor.shape());
            acc = acc.add(A::from_scalar(tensor.get(in_pos)));
        }

        out_data[out_linear] = acc.to_scalar();
    }

    if out_shape.is_empty() {
        Tensor::from_data(&out_data, &[])
    } else {
        Tensor::from_data(&out_data, &out_shape)
    }
}
```

### Change 3: Helper Functions

**File:** `src/einsum/engine.rs`

```rust
/// Convert linear index to multi-dimensional index (column-major)
fn linear_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut multi = vec![0; shape.len()];
    for i in 0..shape.len() {
        multi[i] = linear % shape[i];
        linear /= shape[i];
    }
    multi
}

/// Compute input tensor position from index values (column-major)
fn compute_input_position(
    ix: &[usize],
    idx_values: &HashMap<usize, usize>,
    shape: &[usize],
) -> usize {
    let mut pos = 0;
    let mut stride = 1;
    for (dim, &idx) in ix.iter().enumerate() {
        pos += idx_values[&idx] * stride;
        stride *= shape[dim];
    }
    pos
}
```

### Change 4: Add `Tensor::get` Method (if not exists)

```rust
impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Get element at linear index (column-major)
    pub fn get(&self, index: usize) -> T {
        self.to_vec()[index]
    }
}
```

### Change 5: Remove Old `execute_unary`

Delete the old `execute_unary` method from `impl Einsum<usize>` as it's replaced by `execute_unary_naive`.

---

## Testing

### Tests to Un-ignore

In `tests/omeinsum_compat.rs`:

1. `test_transpose` - Remove `#[ignore]`, update to use convenience function
2. `test_partial_trace_4d` - Remove `#[ignore]`, verify shape reduction works
3. `test_outer_product_gradient` - Remove `#[ignore]`, should work now

### New Unit Tests

Add to `src/einsum/engine.rs` tests module:

```rust
#[test]
fn test_unary_transpose() {
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum::<Standard<f32>, _, _>(&[&a], &[&[0, 1]], &[1, 0]);
    assert_eq!(c.shape(), &[3, 2]);
}

#[test]
fn test_unary_partial_trace_4d() {
    let a = Tensor::<f32, Cpu>::from_data(&vec![1.0; 16], &[2, 2, 2, 2]);
    let c = einsum::<Standard<f32>, _, _>(&[&a], &[&[0, 1, 1, 2]], &[0, 2]);
    assert_eq!(c.shape(), &[2, 2]);
}
```

---

## Deferred Work

- **Issue #9**: Tropical backprop for unary operations - `execute_tree_with_argmax` needs separate design discussion
- **Issue #4**: Unary gradients for Standard algebra - requires extending gradient infrastructure

---

## Summary

| Component | Change |
|-----------|--------|
| `execute_tree` Leaf | Call `execute_unary_naive` instead of clone |
| `execute_unary_naive` | New ~50 line free function with naive loop |
| `linear_to_multi` | New helper function |
| `compute_input_position` | New helper function |
| `Tensor::get` | New method (if needed) |
| Old `execute_unary` | Remove |

**Estimated LOC:** ~80 lines added, ~60 lines removed
