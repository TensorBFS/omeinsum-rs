//! Tensor operations including GEMM and contraction.

use super::{compute_contiguous_strides, Tensor};
use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar};

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
    ) -> Self
    where
        T: BackendScalar<B>,
    {
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
    ) -> (Self, Tensor<u32, B>)
    where
        T: BackendScalar<B>,
    {
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
    ) -> (Self, Option<Tensor<u32, B>>)
    where
        T: BackendScalar<B>,
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
