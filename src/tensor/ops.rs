//! Tensor operations including GEMM and contraction.

use super::{compute_contiguous_strides, Tensor};
use crate::algebra::{Algebra, Scalar};
use crate::backend::Backend;

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// General matrix multiplication using the specified algebra.
    ///
    /// Computes C = A ⊗ B where:
    /// - ⊗ is element-wise semiring multiplication
    /// - Reduction uses semiring addition
    ///
    /// # Type Parameters
    ///
    /// * `A` - The algebra (e.g., `Standard<f32>`, `MaxPlus<f32>`)
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    /// use omeinsum::algebra::{Standard, MaxPlus};
    ///
    /// let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    ///
    /// // Standard: C[i,j] = Σ_k A[i,k] × B[k,j]
    /// let c_std = a.gemm::<Standard<f32>>(&b);
    ///
    /// // Tropical: C[i,j] = max_k (A[i,k] + B[k,j])
    /// let c_trop = a.gemm::<MaxPlus<f32>>(&b);
    /// ```
    pub fn gemm<A: Algebra<Scalar = T>>(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "gemm requires 2D tensors");
        assert_eq!(other.ndim(), 2, "gemm requires 2D tensors");
        assert_eq!(
            self.shape[1], other.shape[0],
            "gemm dimension mismatch: [{}, {}] × [{}, {}]",
            self.shape[0], self.shape[1], other.shape[0], other.shape[1]
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        // Ensure inputs are contiguous
        let a = self.contiguous();
        let b = other.contiguous();

        // Call backend GEMM
        let c_storage = self.backend.gemm::<A>(&a.storage, m, k, &b.storage, n);

        Self::from_raw(
            c_storage,
            vec![m, n],
            compute_contiguous_strides(&[m, n]),
            0,
            self.backend.clone(),
        )
    }

    /// GEMM with argmax tracking for backpropagation.
    ///
    /// Returns `(result, argmax)` where `argmax[i, j]` is the index `k`
    /// that "won" the reduction for element `[i, j]`.
    pub fn gemm_with_argmax<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
    ) -> (Self, Tensor<u32, B>) {
        assert_eq!(self.ndim(), 2, "gemm requires 2D tensors");
        assert_eq!(other.ndim(), 2, "gemm requires 2D tensors");
        assert_eq!(
            self.shape[1], other.shape[0],
            "gemm dimension mismatch: [{}, {}] × [{}, {}]",
            self.shape[0], self.shape[1], other.shape[0], other.shape[1]
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let a = self.contiguous();
        let b = other.contiguous();

        let (c_storage, argmax_storage) = self
            .backend
            .gemm_with_argmax::<A>(&a.storage, m, k, &b.storage, n);

        let c = Self::from_raw(
            c_storage,
            vec![m, n],
            compute_contiguous_strides(&[m, n]),
            0,
            self.backend.clone(),
        );

        let argmax = Tensor::<u32, B>::from_raw(
            argmax_storage,
            vec![m, n],
            compute_contiguous_strides(&[m, n]),
            0,
            self.backend.clone(),
        );

        (c, argmax)
    }

    /// Binary tensor contraction using reshape-to-GEMM strategy.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to contract with
    /// * `ia` - Index labels for self
    /// * `ib` - Index labels for other
    /// * `iy` - Output index labels
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    /// use omeinsum::algebra::MaxPlus;
    ///
    /// // A[i,j,k] × B[j,k,l] → C[i,l]
    /// let a = Tensor::<f32, Cpu>::from_data(&(0..24).map(|x| x as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    /// let b = Tensor::<f32, Cpu>::from_data(&(0..60).map(|x| x as f32).collect::<Vec<_>>(), &[3, 4, 5]);
    /// let c = a.contract_binary::<MaxPlus<f32>>(&b, &[0, 1, 2], &[1, 2, 3], &[0, 3]);
    /// assert_eq!(c.shape(), &[2, 5]);
    /// ```
    pub fn contract_binary<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> Self {
        let (result, _) = self.contract_binary_impl::<A>(other, ia, ib, iy, false);
        result
    }

    /// Binary contraction with argmax tracking.
    pub fn contract_binary_with_argmax<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> (Self, Tensor<u32, B>) {
        let (result, argmax) = self.contract_binary_impl::<A>(other, ia, ib, iy, true);
        (result, argmax.expect("argmax requested but not returned"))
    }

    fn contract_binary_impl<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        track_argmax: bool,
    ) -> (Self, Option<Tensor<u32, B>>) {
        assert_eq!(ia.len(), self.ndim(), "ia length must match self.ndim()");
        assert_eq!(ib.len(), other.ndim(), "ib length must match other.ndim()");

        // Classify indices
        let (batch, left, right, contracted) = classify_indices(ia, ib, iy);

        // Compute sizes
        let batch_size: usize = batch
            .iter()
            .map(|&i| self.shape[index_of(ia, i)])
            .product::<usize>()
            .max(1);
        let left_size: usize = left
            .iter()
            .map(|&i| self.shape[index_of(ia, i)])
            .product::<usize>()
            .max(1);
        let right_size: usize = right
            .iter()
            .map(|&i| other.shape[index_of(ib, i)])
            .product::<usize>()
            .max(1);
        let contract_size: usize = contracted
            .iter()
            .map(|&i| self.shape[index_of(ia, i)])
            .product::<usize>()
            .max(1);

        // Permute A to [batch, left, contracted]
        let a_perm_order = compute_permutation(ia, &batch, &left, &contracted);
        let a_permuted = self.permute(&a_perm_order);
        let a_matrix = a_permuted.reshape(&[batch_size * left_size, contract_size]);

        // Permute B to [batch, contracted, right]
        let b_perm_order = compute_permutation(ib, &batch, &contracted, &right);
        let b_permuted = other.permute(&b_perm_order);
        let b_matrix = b_permuted.reshape(&[batch_size * contract_size, right_size]);

        // Handle batch dimension
        // For simplicity, we reshape B so that GEMM handles batching implicitly
        // A: [B*L, K], B: [K, B*R] (need to rearrange B for this)

        // Actually for batched case, we need to be more careful
        // For now, handle the simple non-batched case
        let (c_matrix, argmax) = if batch.is_empty() {
            // Simple case: no batch dimensions
            // A: [L, K], B needs to be [K, R]
            let b_for_gemm = b_matrix.reshape(&[contract_size, right_size]);

            if track_argmax {
                let (c, arg) = a_matrix.gemm_with_argmax::<A>(&b_for_gemm);
                (c, Some(arg))
            } else {
                (a_matrix.gemm::<A>(&b_for_gemm), None)
            }
        } else {
            // Batched case: use batched GEMM
            // A is [batch_size * left_size, contract_size], need [batch_size, left_size, contract_size]
            // B is [batch_size * contract_size, right_size], need [batch_size, contract_size, right_size]
            let a_batched = a_permuted
                .reshape(&[batch_size, left_size, contract_size])
                .contiguous();
            let b_batched = b_permuted
                .reshape(&[batch_size, contract_size, right_size])
                .contiguous();

            if track_argmax {
                let (c_storage, argmax_storage) = self.backend.gemm_batched_with_argmax::<A>(
                    &a_batched.storage,
                    batch_size,
                    left_size,
                    contract_size,
                    &b_batched.storage,
                    right_size,
                );

                let c = Self::from_raw(
                    c_storage,
                    vec![batch_size, left_size, right_size],
                    compute_contiguous_strides(&[batch_size, left_size, right_size]),
                    0,
                    self.backend.clone(),
                );

                let argmax = Tensor::<u32, B>::from_raw(
                    argmax_storage,
                    vec![batch_size, left_size, right_size],
                    compute_contiguous_strides(&[batch_size, left_size, right_size]),
                    0,
                    self.backend.clone(),
                );

                (c, Some(argmax))
            } else {
                let c_storage = self.backend.gemm_batched::<A>(
                    &a_batched.storage,
                    batch_size,
                    left_size,
                    contract_size,
                    &b_batched.storage,
                    right_size,
                );

                let c = Self::from_raw(
                    c_storage,
                    vec![batch_size, left_size, right_size],
                    compute_contiguous_strides(&[batch_size, left_size, right_size]),
                    0,
                    self.backend.clone(),
                );

                (c, None)
            }
        };

        // Compute output shape
        let mut out_shape = Vec::new();
        let mut shape_map = std::collections::HashMap::new();
        for (idx, &i) in ia.iter().enumerate() {
            shape_map.insert(i, self.shape[idx]);
        }
        for (idx, &i) in ib.iter().enumerate() {
            shape_map.insert(i, other.shape[idx]);
        }
        for &i in iy {
            out_shape.push(*shape_map.get(&i).expect("Output index not found"));
        }

        // Reshape and permute to output
        let c_shaped = c_matrix.reshape(&out_shape);

        // Compute output permutation if needed
        // Current order is [batch..., left..., right...]
        // Need to permute to iy order
        let current_order: Vec<usize> = batch
            .iter()
            .chain(left.iter())
            .chain(right.iter())
            .copied()
            .collect();

        if current_order == iy {
            (c_shaped, argmax)
        } else {
            let out_perm: Vec<usize> = iy
                .iter()
                .map(|i| current_order.iter().position(|x| x == i).unwrap())
                .collect();
            (c_shaped.permute(&out_perm).contiguous(), argmax)
        }
    }
}

/// Classify indices into batch, left-only, right-only, and contracted.
fn classify_indices(
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let ia_set: std::collections::HashSet<_> = ia.iter().copied().collect();
    let ib_set: std::collections::HashSet<_> = ib.iter().copied().collect();
    let iy_set: std::collections::HashSet<_> = iy.iter().copied().collect();

    let mut batch = Vec::new();
    let mut left = Vec::new();
    let mut contracted = Vec::new();

    for &i in ia {
        if ib_set.contains(&i) && iy_set.contains(&i) {
            batch.push(i);
        } else if ib_set.contains(&i) && !iy_set.contains(&i) {
            contracted.push(i);
        } else {
            left.push(i);
        }
    }

    let right: Vec<usize> = ib.iter().filter(|i| !ia_set.contains(i)).copied().collect();

    (batch, left, right, contracted)
}

/// Find index of value in slice.
fn index_of(slice: &[usize], value: usize) -> usize {
    slice
        .iter()
        .position(|&x| x == value)
        .expect("Index not found")
}

/// Compute permutation to reorder indices.
fn compute_permutation(
    current: &[usize],
    first: &[usize],
    second: &[usize],
    third: &[usize],
) -> Vec<usize> {
    let target: Vec<usize> = first
        .iter()
        .chain(second.iter())
        .chain(third.iter())
        .copied()
        .collect();

    target
        .iter()
        .map(|i| {
            current
                .iter()
                .position(|x| x == i)
                .expect("Index not found")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::backend::Cpu;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_gemm_standard() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.gemm::<Standard<f32>>(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_gemm_maxplus() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.gemm::<MaxPlus<f32>>(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_contract_binary() {
        // A[i,j] × B[j,k] → C[i,k]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn test_classify_indices() {
        // A[i,j,k] × B[j,k,l] → C[i,l]
        let (batch, left, right, contracted) = classify_indices(&[0, 1, 2], &[1, 2, 3], &[0, 3]);

        assert!(batch.is_empty());
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![3]);
        assert_eq!(contracted, vec![1, 2]);
    }

    #[test]
    fn test_contract_binary_batched() {
        // A[b,i,j] × B[b,j,k] → C[b,i,k]
        // 2 batches, 2x2 matrices
        let a =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);

        let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(c.shape(), &[2, 2, 2]);
        // Batch 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
        // Batch 1: [[5,6],[7,8]] @ [[1,0],[0,1]] = [[5,6],[7,8]]
        assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
