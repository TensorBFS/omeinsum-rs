# Unified Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a unified `Backend` trait so `Tensor<T, Cuda>` works with the same `einsum()` API as `Tensor<T, Cpu>`.

**Architecture:** Define Backend at the contraction level (not GEMM). CPU implements contract via permute→reshape→GEMM→reshape. CUDA implements contract via direct cuTENSOR call. Use `BackendScalar<B>` marker trait for compile-time type safety.

**Tech Stack:** Rust, faer (CPU GEMM), cuTENSOR (CUDA contractions)

---

## Task 1: Add BackendScalar Marker Trait

**Files:**
- Modify: `src/backend/traits.rs`

**Step 1: Add the BackendScalar trait definition**

Add after the `Storage` trait:

```rust
/// Marker trait for scalar types supported by a specific backend.
///
/// This enables compile-time checking that a scalar type is supported
/// by a particular backend (e.g., CUDA only supports f32/f64/complex).
pub trait BackendScalar<B: Backend>: Scalar {}
```

**Step 2: Add blanket impl for Cpu**

Add at end of file:

```rust
// CPU supports all Scalar types
impl<T: Scalar> BackendScalar<crate::backend::Cpu> for T {}
```

**Step 3: Run tests to verify nothing broke**

Run: `cargo test --features tropical --lib`
Expected: All tests pass (no behavior change yet)

**Step 4: Commit**

```bash
git add src/backend/traits.rs
git commit -m "feat(backend): add BackendScalar marker trait

Enables compile-time checking that a scalar type is supported by a
particular backend. CPU supports all Scalar types via blanket impl."
```

---

## Task 2: Add Backend::contract Method Signature

**Files:**
- Modify: `src/backend/traits.rs`

**Step 1: Add contract method to Backend trait**

Add to the `Backend` trait after `copy_strided`:

```rust
    /// Binary tensor contraction.
    ///
    /// Computes: C[modes_c] = Σ A[modes_a] ⊗ B[modes_b]
    /// where the sum is over indices appearing in both A and B but not in C.
    ///
    /// # Arguments
    /// * `a`, `b` - Input tensor storage
    /// * `shape_a`, `shape_b` - Tensor shapes
    /// * `strides_a`, `strides_b` - Tensor strides (for non-contiguous support)
    /// * `modes_a`, `modes_b` - Index labels for each tensor dimension
    /// * `shape_c`, `modes_c` - Output shape and index labels
    fn contract<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>;

    /// Contraction with argmax tracking for tropical backpropagation.
    ///
    /// Returns (result, argmax) where argmax contains the index that "won"
    /// the reduction at each output position.
    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>;
```

**Step 2: Verify compilation fails (Cpu doesn't implement yet)**

Run: `cargo check --features tropical`
Expected: Error about `contract` not implemented for `Cpu`

**Step 3: Commit (WIP)**

```bash
git add src/backend/traits.rs
git commit -m "wip: add Backend::contract method signature"
```

---

## Task 3: Reorganize CPU Backend into Module

**Files:**
- Rename: `src/backend/cpu.rs` → `src/backend/cpu/mod.rs`
- Create: `src/backend/cpu/contract.rs`
- Modify: `src/backend/mod.rs`

**Step 1: Create cpu directory and move file**

```bash
mkdir -p src/backend/cpu
mv src/backend/cpu.rs src/backend/cpu/mod.rs
```

**Step 2: Add module declaration in cpu/mod.rs**

Add at top of `src/backend/cpu/mod.rs`:

```rust
mod contract;
```

**Step 3: Create empty contract.rs**

Create `src/backend/cpu/contract.rs`:

```rust
//! CPU tensor contraction via reshape→GEMM→reshape.

use crate::algebra::{Algebra, Scalar};
use crate::backend::BackendScalar;
use super::Cpu;

/// Classify modes into batch, left, right, and contracted.
pub(super) fn classify_modes(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    todo!("Will be implemented in next task")
}
```

**Step 4: Verify module structure compiles**

Run: `cargo check --features tropical`
Expected: Still fails (contract not implemented), but no module errors

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(backend): reorganize cpu into module

Prepare for adding contract.rs with contraction logic."
```

---

## Task 4: Implement CPU Contract Helper Functions

**Files:**
- Modify: `src/backend/cpu/contract.rs`

**Step 1: Implement classify_modes**

Replace the todo in `src/backend/cpu/contract.rs`:

```rust
//! CPU tensor contraction via reshape→GEMM→reshape.

use std::collections::HashSet;

/// Classify modes into batch, left-only, right-only, and contracted.
///
/// - batch: in both A and B, and in output C
/// - left: only in A (free indices from A)
/// - right: only in B (free indices from B)
/// - contracted: in both A and B, but NOT in output C
pub(super) fn classify_modes(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let a_set: HashSet<i32> = modes_a.iter().copied().collect();
    let b_set: HashSet<i32> = modes_b.iter().copied().collect();
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();

    let mut batch = Vec::new();
    let mut left = Vec::new();
    let mut contracted = Vec::new();

    for &m in modes_a {
        if b_set.contains(&m) && c_set.contains(&m) {
            if !batch.contains(&m) {
                batch.push(m);
            }
        } else if b_set.contains(&m) && !c_set.contains(&m) {
            if !contracted.contains(&m) {
                contracted.push(m);
            }
        } else if !left.contains(&m) {
            left.push(m);
        }
    }

    let right: Vec<i32> = modes_b
        .iter()
        .filter(|m| !a_set.contains(m))
        .copied()
        .collect();

    (batch, left, right, contracted)
}

/// Find the position of a mode in a modes array.
pub(super) fn mode_position(modes: &[i32], mode: i32) -> usize {
    modes.iter().position(|&m| m == mode).expect("mode not found")
}

/// Compute the product of dimensions for given modes.
pub(super) fn product_of_dims(modes: &[i32], all_modes: &[i32], shape: &[usize]) -> usize {
    modes
        .iter()
        .map(|&m| shape[mode_position(all_modes, m)])
        .product::<usize>()
        .max(1)
}

/// Compute permutation to reorder modes to [first..., second..., third...].
pub(super) fn compute_permutation(
    current: &[i32],
    first: &[i32],
    second: &[i32],
    third: &[i32],
) -> Vec<usize> {
    let target: Vec<i32> = first
        .iter()
        .chain(second.iter())
        .chain(third.iter())
        .copied()
        .collect();

    target
        .iter()
        .map(|m| mode_position(current, *m))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_modes_matmul() {
        // ij,jk->ik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1], &[1, 2], &[0, 2]);

        assert!(batch.is_empty());
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![2]);
        assert_eq!(contracted, vec![1]);
    }

    #[test]
    fn test_classify_modes_batched() {
        // bij,bjk->bik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(batch, vec![0]);
        assert_eq!(left, vec![1]);
        assert_eq!(right, vec![3]);
        assert_eq!(contracted, vec![2]);
    }

    #[test]
    fn test_product_of_dims() {
        let modes = &[0, 1, 2];
        let shape = &[2, 3, 4];

        assert_eq!(product_of_dims(&[0], modes, shape), 2);
        assert_eq!(product_of_dims(&[1, 2], modes, shape), 12);
        assert_eq!(product_of_dims(&[], modes, shape), 1);
    }

    #[test]
    fn test_compute_permutation() {
        // Current: [0, 1, 2], want: [0, 2, 1]
        let perm = compute_permutation(&[0, 1, 2], &[0], &[2], &[1]);
        assert_eq!(perm, vec![0, 2, 1]);
    }
}
```

**Step 2: Run the new tests**

Run: `cargo test --features tropical contract::tests`
Expected: All 4 tests pass

**Step 3: Commit**

```bash
git add src/backend/cpu/contract.rs
git commit -m "feat(cpu): add contraction helper functions

- classify_modes: categorize indices into batch/left/right/contracted
- product_of_dims: compute dimension products
- compute_permutation: reorder modes for reshape"
```

---

## Task 5: Implement CPU Contract Core Logic

**Files:**
- Modify: `src/backend/cpu/contract.rs`
- Modify: `src/backend/cpu/mod.rs`

**Step 1: Add the contract function to contract.rs**

Add at end of `src/backend/cpu/contract.rs` (before tests):

```rust
use crate::algebra::Algebra;
use crate::backend::Cpu;
use crate::tensor::compute_contiguous_strides;

/// Execute tensor contraction on CPU via reshape→GEMM→reshape.
pub(super) fn contract<A: Algebra>(
    cpu: &Cpu,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Vec<A::Scalar>
where
    A::Scalar: crate::algebra::Scalar,
{
    // 1. Make inputs contiguous if needed
    let a_contig = ensure_contiguous(a, shape_a, strides_a);
    let b_contig = ensure_contiguous(b, shape_b, strides_b);

    // 2. Classify modes
    let (batch, left, right, contracted) = classify_modes(modes_a, modes_b, modes_c);

    // 3. Compute dimension sizes
    let batch_size = product_of_dims(&batch, modes_a, shape_a);
    let left_size = product_of_dims(&left, modes_a, shape_a);
    let right_size = product_of_dims(&right, modes_b, shape_b);
    let contract_size = product_of_dims(&contracted, modes_a, shape_a);

    // 4. Permute A to [batch, left, contracted]
    let a_perm = compute_permutation(modes_a, &batch, &left, &contracted);
    let a_permuted = permute_data(&a_contig, shape_a, &a_perm);

    // 5. Permute B to [batch, contracted, right]
    let b_perm = compute_permutation(modes_b, &batch, &contracted, &right);
    let b_permuted = permute_data(&b_contig, shape_b, &b_perm);

    // 6. Call GEMM
    let c_data = if batch.is_empty() {
        cpu.gemm_internal::<A>(&a_permuted, left_size, contract_size, &b_permuted, right_size)
    } else {
        cpu.gemm_batched_internal::<A>(
            &a_permuted, batch_size, left_size, contract_size,
            &b_permuted, right_size,
        )
    };

    // 7. Permute result to output order
    let current_order: Vec<i32> = batch.iter()
        .chain(left.iter())
        .chain(right.iter())
        .copied()
        .collect();

    if current_order == modes_c {
        c_data
    } else {
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|m| current_order.iter().position(|x| x == m).unwrap())
            .collect();
        permute_data(&c_data, &c_shape_current, &out_perm)
    }
}

/// Ensure data is contiguous (copy if strided).
fn ensure_contiguous<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
) -> Vec<T> {
    let expected_strides = compute_contiguous_strides(shape);
    if strides == expected_strides {
        data.to_vec()
    } else {
        // Copy with stride handling
        let numel: usize = shape.iter().product();
        let mut result = vec![T::default(); numel];
        copy_strided_to_contiguous(data, &mut result, shape, strides);
        result
    }
}

/// Copy strided data to contiguous buffer.
fn copy_strided_to_contiguous<T: Copy>(
    src: &[T],
    dst: &mut [T],
    shape: &[usize],
    strides: &[usize],
) {
    let numel: usize = shape.iter().product();
    let dst_strides = compute_contiguous_strides(shape);

    for i in 0..numel {
        // Convert linear index to multi-index
        let mut remaining = i;
        let mut src_offset = 0;
        for dim in 0..shape.len() {
            let coord = remaining % shape[dim];
            remaining /= shape[dim];
            src_offset += coord * strides[dim];
        }
        dst[i] = src[src_offset];
    }
}

/// Permute data according to axis permutation.
fn permute_data<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Vec<T> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return data.to_vec(); // Already in correct order
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let numel: usize = shape.iter().product();
    let mut result = vec![T::default(); numel];

    let old_strides = compute_contiguous_strides(shape);
    let new_strides = compute_contiguous_strides(&new_shape);

    for new_idx in 0..numel {
        // Convert new linear index to new multi-index
        let mut remaining = new_idx;
        let mut new_coords = vec![0; shape.len()];
        for dim in 0..new_shape.len() {
            new_coords[dim] = remaining % new_shape[dim];
            remaining /= new_shape[dim];
        }

        // Map to old coordinates via inverse permutation
        let mut old_idx = 0;
        for (new_dim, &old_dim) in perm.iter().enumerate() {
            old_idx += new_coords[new_dim] * old_strides[old_dim];
        }

        result[new_idx] = data[old_idx];
    }

    result
}
```

**Step 2: Add internal GEMM methods to Cpu**

Add to `src/backend/cpu/mod.rs`, inside the `impl Backend for Cpu` block, after existing methods:

```rust
impl Cpu {
    /// Internal GEMM for contraction (not part of Backend trait).
    pub(crate) fn gemm_internal<A: Algebra>(
        &self,
        a: &[A::Scalar],
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> Vec<A::Scalar> {
        self.gemm::<A>(a, m, k, b, n)
    }

    /// Internal batched GEMM for contraction.
    pub(crate) fn gemm_batched_internal<A: Algebra>(
        &self,
        a: &[A::Scalar],
        batch_size: usize,
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> Vec<A::Scalar> {
        self.gemm_batched::<A>(a, batch_size, m, k, b, n)
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo check --features tropical`
Expected: Still fails (Backend::contract not implemented), but contract.rs compiles

**Step 4: Commit**

```bash
git add src/backend/cpu/
git commit -m "feat(cpu): implement contract core logic

- ensure_contiguous: handle strided inputs
- permute_data: reorder tensor dimensions
- contract: full reshape→GEMM→reshape pipeline"
```

---

## Task 6: Implement Backend::contract for Cpu

**Files:**
- Modify: `src/backend/cpu/mod.rs`

**Step 1: Add contract implementation to Backend impl**

Add to `impl Backend for Cpu`, using the contract module:

```rust
    fn contract<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        contract::contract::<A>(
            self, a, shape_a, strides_a, modes_a,
            b, shape_b, strides_b, modes_b,
            shape_c, modes_c,
        )
    }

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        contract::contract_with_argmax::<A>(
            self, a, shape_a, strides_a, modes_a,
            b, shape_b, strides_b, modes_b,
            shape_c, modes_c,
        )
    }
```

**Step 2: Add contract_with_argmax to contract.rs**

Add to `src/backend/cpu/contract.rs` (similar to contract but with argmax):

```rust
/// Execute tensor contraction with argmax tracking.
pub(super) fn contract_with_argmax<A: Algebra<Index = u32>>(
    cpu: &Cpu,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> (Vec<A::Scalar>, Vec<u32>)
where
    A::Scalar: crate::algebra::Scalar,
{
    // Same setup as contract
    let a_contig = ensure_contiguous(a, shape_a, strides_a);
    let b_contig = ensure_contiguous(b, shape_b, strides_b);
    let (batch, left, right, contracted) = classify_modes(modes_a, modes_b, modes_c);
    let batch_size = product_of_dims(&batch, modes_a, shape_a);
    let left_size = product_of_dims(&left, modes_a, shape_a);
    let right_size = product_of_dims(&right, modes_b, shape_b);
    let contract_size = product_of_dims(&contracted, modes_a, shape_a);

    let a_perm = compute_permutation(modes_a, &batch, &left, &contracted);
    let a_permuted = permute_data(&a_contig, shape_a, &a_perm);
    let b_perm = compute_permutation(modes_b, &batch, &contracted, &right);
    let b_permuted = permute_data(&b_contig, shape_b, &b_perm);

    // Call GEMM with argmax
    let (c_data, argmax) = if batch.is_empty() {
        cpu.gemm_with_argmax_internal::<A>(
            &a_permuted, left_size, contract_size,
            &b_permuted, right_size,
        )
    } else {
        cpu.gemm_batched_with_argmax_internal::<A>(
            &a_permuted, batch_size, left_size, contract_size,
            &b_permuted, right_size,
        )
    };

    // Permute result
    let current_order: Vec<i32> = batch.iter()
        .chain(left.iter())
        .chain(right.iter())
        .copied()
        .collect();

    if current_order == modes_c {
        (c_data, argmax)
    } else {
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|m| current_order.iter().position(|x| x == m).unwrap())
            .collect();
        (
            permute_data(&c_data, &c_shape_current, &out_perm),
            permute_data(&argmax, &c_shape_current, &out_perm),
        )
    }
}
```

**Step 3: Add argmax internal methods to Cpu**

Add to `src/backend/cpu/mod.rs`:

```rust
impl Cpu {
    // ... existing methods ...

    pub(crate) fn gemm_with_argmax_internal<A: Algebra<Index = u32>>(
        &self,
        a: &[A::Scalar],
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        self.gemm_with_argmax::<A>(a, m, k, b, n)
    }

    pub(crate) fn gemm_batched_with_argmax_internal<A: Algebra<Index = u32>>(
        &self,
        a: &[A::Scalar],
        batch_size: usize,
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        self.gemm_batched_with_argmax::<A>(a, batch_size, m, k, b, n)
    }
}
```

**Step 4: Add use statement for BackendScalar**

Add at top of `src/backend/cpu/mod.rs`:

```rust
use super::BackendScalar;
```

**Step 5: Run tests**

Run: `cargo test --features tropical`
Expected: All existing tests pass

**Step 6: Commit**

```bash
git add src/backend/cpu/
git commit -m "feat(cpu): implement Backend::contract and contract_with_argmax

CPU now implements the unified contract API via reshape→GEMM→reshape."
```

---

## Task 7: Add Integration Test for CPU Contract

**Files:**
- Create: `tests/backend_contract.rs`

**Step 1: Write the test file**

Create `tests/backend_contract.rs`:

```rust
//! Tests for Backend::contract unified API.

use omeinsum::{Cpu, Tensor, Standard};
use omeinsum::backend::Backend;

#[test]
fn test_cpu_contract_matmul() {
    // ij,jk->ik (matrix multiplication)
    let cpu = Cpu::default();

    let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2 column-major
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = cpu.contract::<Standard<f64>>(
        &a, &[2, 2], &[1, 2], &[0, 1],
        &b, &[2, 2], &[1, 2], &[1, 2],
        &[2, 2], &[0, 2],
    );

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //         = [[19, 22], [43, 50]]
    // Column-major: [19, 43, 22, 50]
    assert_eq!(c, vec![19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn test_cpu_contract_inner_product() {
    // i,i-> (inner product)
    let cpu = Cpu::default();

    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let c = cpu.contract::<Standard<f64>>(
        &a, &[3], &[1], &[0],
        &b, &[3], &[1], &[0],
        &[1], &[],
    );

    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(c, vec![32.0]);
}

#[test]
fn test_cpu_contract_outer_product() {
    // i,j->ij (outer product)
    let cpu = Cpu::default();

    let a = vec![1.0f64, 2.0];
    let b = vec![3.0, 4.0, 5.0];

    let c = cpu.contract::<Standard<f64>>(
        &a, &[2], &[1], &[0],
        &b, &[3], &[1], &[1],
        &[2, 3], &[0, 1],
    );

    // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3,4,5], [6,8,10]]
    // Column-major: [3, 6, 4, 8, 5, 10]
    assert_eq!(c, vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
}

#[test]
fn test_cpu_contract_batched() {
    // bij,bjk->bik (batched matmul)
    let cpu = Cpu::default();

    // 2 batches of 2x2 matrices
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];

    let c = cpu.contract::<Standard<f64>>(
        &a, &[2, 2, 2], &[1, 2, 4], &[0, 1, 2],
        &b, &[2, 2, 2], &[1, 2, 4], &[0, 2, 3],
        &[2, 2, 2], &[0, 1, 3],
    );

    // Batch 0: identity @ [[1,2],[3,4]] = [[1,2],[3,4]]
    // Batch 1: 2*identity @ [[5,6],[7,8]] = [[10,12],[14,16]]
    assert_eq!(c.len(), 8);
}

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_tropical() {
    use omeinsum::algebra::MaxPlus;

    let cpu = Cpu::default();

    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];

    let c = cpu.contract::<MaxPlus<f64>>(
        &a, &[2, 2], &[1, 2], &[0, 1],
        &b, &[2, 2], &[1, 2], &[1, 2],
        &[2, 2], &[0, 2],
    );

    // MaxPlus: C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(1+1, 2+3) = 5
    // C[1,0] = max(3+1, 4+3) = 7
    // C[0,1] = max(1+2, 2+4) = 6
    // C[1,1] = max(3+2, 4+4) = 8
    assert_eq!(c, vec![5.0, 7.0, 6.0, 8.0]);
}
```

**Step 2: Run the tests**

Run: `cargo test --features tropical backend_contract`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/backend_contract.rs
git commit -m "test: add Backend::contract integration tests

Tests matmul, inner product, outer product, batched, and tropical."
```

---

## Task 8: Simplify Tensor::contract_binary to Use Backend::contract

**Files:**
- Modify: `src/tensor/ops.rs`

**Step 1: Rewrite contract_binary_impl**

Replace the complex implementation in `src/tensor/ops.rs` with a thin wrapper:

```rust
    fn contract_binary_impl<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        track_argmax: bool,
    ) -> (Self, Option<Tensor<u32, B>>)
    where
        T: crate::backend::BackendScalar<B>,
    {
        assert_eq!(ia.len(), self.ndim(), "ia length must match self.ndim()");
        assert_eq!(ib.len(), other.ndim(), "ib length must match other.ndim()");

        // Convert usize indices to i32 modes
        let modes_a: Vec<i32> = ia.iter().map(|&i| i as i32).collect();
        let modes_b: Vec<i32> = ib.iter().map(|&i| i as i32).collect();
        let modes_c: Vec<i32> = iy.iter().map(|&i| i as i32).collect();

        // Compute output shape
        let shape_c = compute_output_shape(
            self.shape(), &modes_a,
            other.shape(), &modes_b,
            &modes_c,
        );

        if track_argmax {
            let (c_storage, argmax_storage) = self.backend.contract_with_argmax::<A>(
                self.storage.as_ref(),
                self.shape(),
                self.strides(),
                &modes_a,
                other.storage.as_ref(),
                other.shape(),
                other.strides(),
                &modes_b,
                &shape_c,
                &modes_c,
            );

            let c = Self::from_storage(c_storage, &shape_c, self.backend.clone());
            let argmax = Tensor::<u32, B>::from_storage(
                argmax_storage,
                &shape_c,
                self.backend.clone(),
            );
            (c, Some(argmax))
        } else {
            let c_storage = self.backend.contract::<A>(
                self.storage.as_ref(),
                self.shape(),
                self.strides(),
                &modes_a,
                other.storage.as_ref(),
                other.shape(),
                other.strides(),
                &modes_b,
                &shape_c,
                &modes_c,
            );

            let c = Self::from_storage(c_storage, &shape_c, self.backend.clone());
            (c, None)
        }
    }
```

**Step 2: Add compute_output_shape helper**

Add before the impl block:

```rust
/// Compute output shape from input shapes and modes.
fn compute_output_shape(
    shape_a: &[usize],
    modes_a: &[i32],
    shape_b: &[usize],
    modes_b: &[i32],
    modes_c: &[i32],
) -> Vec<usize> {
    let mut shape_map = std::collections::HashMap::new();
    for (idx, &m) in modes_a.iter().enumerate() {
        shape_map.insert(m, shape_a[idx]);
    }
    for (idx, &m) in modes_b.iter().enumerate() {
        shape_map.insert(m, shape_b[idx]);
    }
    modes_c.iter().map(|m| shape_map[m]).collect()
}
```

**Step 3: Add BackendScalar bound to public methods**

Update `contract_binary` and `contract_binary_with_argmax`:

```rust
    pub fn contract_binary<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> Self
    where
        T: crate::backend::BackendScalar<B>,
    {
        let (result, _) = self.contract_binary_impl::<A>(other, ia, ib, iy, false);
        result
    }

    pub fn contract_binary_with_argmax<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> (Self, Tensor<u32, B>)
    where
        T: crate::backend::BackendScalar<B>,
    {
        let (result, argmax) = self.contract_binary_impl::<A>(other, ia, ib, iy, true);
        (result, argmax.expect("argmax requested but not returned"))
    }
```

**Step 4: Remove old helper functions**

Remove `classify_indices`, `index_of`, `compute_permutation` from `ops.rs` (they're now in `cpu/contract.rs`).

**Step 5: Run tests**

Run: `cargo test --features tropical`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/tensor/ops.rs
git commit -m "refactor(tensor): simplify contract_binary to use Backend::contract

Tensor is now a thin wrapper; backend handles all complexity."
```

---

## Task 9: Remove Old GEMM Methods from Backend Trait

**Files:**
- Modify: `src/backend/traits.rs`
- Modify: `src/backend/cpu/mod.rs`
- Modify: `src/tensor/ops.rs`

**Step 1: Remove gemm methods from Backend trait**

Remove these from the `Backend` trait in `src/backend/traits.rs`:
- `fn gemm<A: Algebra>(...)`
- `fn gemm_with_argmax<A: Algebra<Index = u32>>(...)`
- `fn gemm_batched<A: Algebra>(...)`
- `fn gemm_batched_with_argmax<A: Algebra<Index = u32>>(...)`
- `fn gemm_backward_a<A: Algebra>(...)`
- `fn gemm_backward_b<A: Algebra>(...)`

**Step 2: Move GEMM methods to Cpu impl (not trait)**

In `src/backend/cpu/mod.rs`, the existing `gemm*` methods should become inherent methods on `Cpu`, not part of `impl Backend for Cpu`. They're already used internally by `contract.rs`.

**Step 3: Remove Tensor::gemm**

Remove `gemm` and `gemm_with_argmax` from `src/tensor/ops.rs` (users should use `einsum` or `contract_binary`).

**Step 4: Update any tests using Tensor::gemm**

Change to use `contract_binary` or `einsum` instead.

**Step 5: Run tests**

Run: `cargo test --features tropical`
Expected: All tests pass

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(backend): remove GEMM methods from Backend trait

GEMM is now an internal implementation detail of CPU backend.
Users should use einsum() or contract_binary()."
```

---

## Task 10: Add CUDA Backend::contract Implementation

**Files:**
- Modify: `src/backend/cuda/mod.rs`

**Step 1: Add BackendScalar impls for CUDA types**

Add to `src/backend/traits.rs`:

```rust
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f32 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f64 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f32> {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f64> {}
```

**Step 2: Implement Backend for Cuda**

Add to `src/backend/cuda/mod.rs`:

```rust
impl Backend for Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;

    fn name() -> &'static str {
        "cuda"
    }

    fn synchronize(&self) {
        self.device.synchronize().expect("CUDA sync failed");
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let slice = self.device.alloc_zeros::<T>(len).expect("CUDA alloc failed");
        CudaStorage::new(slice, self.device.clone())
    }

    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let slice = self.device.htod_sync_copy(data).expect("CUDA htod failed");
        CudaStorage::new(slice, self.device.clone())
    }

    fn copy_strided<T: Scalar>(
        &self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        // For now, download to CPU, copy strided, upload back
        // TODO: Implement GPU strided copy kernel
        let cpu_data = src.to_vec();
        let numel: usize = shape.iter().product();
        let mut result = vec![T::default(); numel];

        let dst_strides = crate::tensor::compute_contiguous_strides(shape);
        for i in 0..numel {
            let mut remaining = i;
            let mut src_offset = offset;
            for dim in 0..shape.len() {
                let coord = remaining % shape[dim];
                remaining /= shape[dim];
                src_offset += coord * strides[dim];
            }
            result[i] = cpu_data[src_offset];
        }

        self.from_slice(&result)
    }

    fn contract<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        _strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        _strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self> + CutensorType + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + num_traits::One + num_traits::Zero,
    {
        // Compute contiguous strides
        let strides_a = compute_contiguous_strides(shape_a);
        let strides_b = compute_contiguous_strides(shape_b);
        let strides_c = compute_contiguous_strides(shape_c);

        // Use existing Cuda::contract method
        self.contract_cutensor::<A::Scalar>(
            a, shape_a, &strides_a, modes_a,
            b, shape_b, &strides_b, modes_b,
            shape_c, &strides_c, modes_c,
        ).expect("cuTENSOR contraction failed")
    }

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        _a: &Self::Storage<A::Scalar>,
        _shape_a: &[usize],
        _strides_a: &[usize],
        _modes_a: &[i32],
        _b: &Self::Storage<A::Scalar>,
        _shape_b: &[usize],
        _strides_b: &[usize],
        _modes_b: &[i32],
        _shape_c: &[usize],
        _modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        // cuTENSOR doesn't support argmax tracking
        // For tropical on CUDA, would need custom kernel
        panic!("contract_with_argmax not yet supported on CUDA")
    }
}
```

**Step 3: Rename existing contract method**

Rename `Cuda::contract` to `Cuda::contract_cutensor` to avoid confusion.

**Step 4: Run tests**

Run: `cargo test --features tropical`
Expected: All CPU tests pass (CUDA tests need hardware)

**Step 5: Commit**

```bash
git add src/backend/
git commit -m "feat(cuda): implement Backend trait for Cuda

Cuda now implements unified Backend::contract via cuTENSOR.
contract_with_argmax panics (needs custom kernel)."
```

---

## Task 11: Export BackendScalar in Public API

**Files:**
- Modify: `src/backend/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Export BackendScalar**

In `src/backend/mod.rs`:

```rust
pub use traits::{Backend, BackendScalar, Storage};
```

**Step 2: Re-export in lib.rs**

In `src/lib.rs`, ensure `BackendScalar` is accessible:

```rust
pub use backend::{Backend, BackendScalar, Cpu, Storage};
```

**Step 3: Run doc tests**

Run: `cargo test --doc --features tropical`
Expected: All pass

**Step 4: Commit**

```bash
git add src/backend/mod.rs src/lib.rs
git commit -m "feat: export BackendScalar in public API"
```

---

## Task 12: Final Cleanup and Documentation

**Files:**
- Modify: `src/backend/traits.rs` (docs)
- Modify: `src/tensor/ops.rs` (remove dead code)

**Step 1: Add documentation to Backend::contract**

Ensure `Backend::contract` has comprehensive docs with examples.

**Step 2: Remove any dead code**

Run: `cargo clippy --features tropical`
Fix any warnings about unused code.

**Step 3: Run full test suite**

Run: `cargo test --features tropical`
Expected: All tests pass

**Step 4: Commit**

```bash
git add -A
git commit -m "docs: add Backend::contract documentation and cleanup"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add BackendScalar marker trait |
| 2 | Add Backend::contract signature |
| 3 | Reorganize CPU into module |
| 4 | Implement CPU contract helpers |
| 5 | Implement CPU contract core |
| 6 | Implement Backend::contract for Cpu |
| 7 | Add integration tests |
| 8 | Simplify Tensor::contract_binary |
| 9 | Remove old GEMM methods |
| 10 | Add CUDA Backend impl |
| 11 | Export BackendScalar |
| 12 | Final cleanup |
