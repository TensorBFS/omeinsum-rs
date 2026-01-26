//! Backward pass implementation for einsum gradients.
//!
//! This module provides gradient computation for tensor contractions,
//! supporting both standard arithmetic and tropical algebras.
//!
//! The key insight for standard algebra is the **index-exchange trick**:
//! - Forward: `y = einsum(ix -> iy, x)`
//! - Backward: `grad_x = einsum(iy -> ix, grad_y)`

use crate::algebra::{Algebra, Scalar};
use crate::backend::Backend;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::engine::execute_unary_naive;

/// Compute gradient for a unary einsum operation.
///
/// Uses the index-exchange trick: backward(ix -> iy) = forward(iy -> ix).
///
/// # Arguments
///
/// * `grad_y` - Gradient of the output tensor
/// * `ix` - Input index labels
/// * `iy` - Output index labels
/// * `size_dict` - Mapping from index labels to sizes
///
/// # Returns
///
/// Gradient tensor with the same shape as the original input.
pub fn contract_unary_backward<A, T, B>(
    grad_y: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend + Default,
{
    // The elegant insight: gradient is just einsum with swapped indices!
    // Forward: y = einsum(ix -> iy, x)
    // Backward: grad_x = einsum(iy -> ix, grad_y)
    execute_unary_naive::<A, T, B>(grad_y, iy, ix, size_dict)
}

/// Compute gradients for a binary contraction.
///
/// Given the gradient of the output (grad_c), compute gradients for both inputs (a, b).
///
/// # Arguments
///
/// * `grad_c` - Gradient of the output tensor
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `argmax` - Optional argmax tensor for tropical algebras
/// * `ia` - Index labels for first input
/// * `ib` - Index labels for second input
/// * `iy` - Output index labels
///
/// # Returns
///
/// Tuple of (grad_a, grad_b) gradients for the two inputs.
pub fn contract_binary_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    argmax: Option<&Tensor<u32, B>>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    if A::needs_argmax() {
        // Tropical backward: route gradients through argmax
        let argmax = argmax.expect("Tropical backward requires argmax");
        tropical_backward::<A, T, B>(grad_c, a, b, argmax, ia, ib, iy)
    } else {
        // Standard backward: grad_a = grad_c @ b.T, grad_b = a.T @ grad_c
        standard_backward::<A, T, B>(grad_c, a, b, ia, ib, iy)
    }
}

/// Backward pass for standard (non-tropical) algebra.
///
/// For a contraction C = A @ B (with proper index handling):
/// - grad_a = grad_c @ b.T  (contracted with b's transpose)
/// - grad_b = a.T @ grad_c  (a's transpose contracted with grad_c)
fn standard_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    // For C[iy] = A[ia] @ B[ib] (contraction over shared indices in ia and ib not in iy):
    //
    // grad_A[ia] = grad_C[iy] @ B[ib] contracted appropriately
    // grad_B[ib] = A[ia] @ grad_C[iy] contracted appropriately
    //
    // The key insight is:
    // - To get grad_A, we contract grad_C with B, but now the contracted indices
    //   are the "right" indices of iy that came from ib, and the output should be ia
    // - To get grad_B, we contract A with grad_C, and the output should be ib

    // Find contracted indices (in both ia and ib, but not in iy)
    let contracted: Vec<usize> = ia
        .iter()
        .filter(|&i| ib.contains(i) && !iy.contains(i))
        .copied()
        .collect();

    // For grad_a: contract grad_c with b to get shape of a
    // grad_c has indices iy, b has indices ib
    // We want result with indices ia
    // The contraction should be over indices that are in both iy and ib (right indices)
    let grad_a = grad_c.contract_binary::<A>(b, iy, ib, ia);

    // For grad_b: contract a with grad_c to get shape of b
    // a has indices ia, grad_c has indices iy
    // We want result with indices ib
    // The contraction should be over indices that are in both ia and iy (left indices)
    let grad_b = a.contract_binary::<A>(grad_c, ia, iy, ib);

    let _ = contracted; // Mark as used (for documentation clarity)

    (grad_a, grad_b)
}

/// Backward pass for tropical (max/min-plus) algebra.
///
/// For tropical algebras, gradients are routed through the argmax:
/// only the "winning" element gets the gradient.
fn tropical_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    argmax: &Tensor<u32, B>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    // For tropical backward, we need to use the argmax to route gradients.
    // The argmax tells us which k index "won" for each output element.
    //
    // For a simple matmul C[i,j] = max_k (A[i,k] + B[k,j]):
    // - grad_A[i, argmax[i,j]] += grad_C[i,j] for each (i,j)
    // - grad_B[argmax[i,j], j] += grad_C[i,j] for each (i,j)

    // Get the shapes we need
    let a_shape = a.shape();
    let b_shape = b.shape();

    // For the simple 2D matmul case: C[i,j] = max_k (A[i,k] + B[k,j])
    // We need to scatter gradients using the argmax.
    //
    // This requires accessing the backend's gemm_backward functions.
    // For now, implement for 2D case (matmul).

    if a.ndim() == 2 && b.ndim() == 2 && grad_c.ndim() == 2 {
        // Matmul case: C[m,n] = A[m,k] * B[k,n]
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // Make tensors contiguous for backend operations
        let grad_c_contig = grad_c.contiguous();
        let a_contig = a.contiguous();
        let b_contig = b.contiguous();
        let argmax_contig = argmax.contiguous();

        // Get storage references
        let grad_c_storage = grad_c_contig
            .storage()
            .expect("contiguous tensor should have storage");
        let b_storage = b_contig
            .storage()
            .expect("contiguous tensor should have storage");
        let a_storage = a_contig
            .storage()
            .expect("contiguous tensor should have storage");
        let argmax_storage = argmax_contig
            .storage()
            .expect("contiguous tensor should have storage");

        // Use backend functions
        let grad_a_storage =
            a.backend()
                .gemm_backward_a::<A>(grad_c_storage, argmax_storage, b_storage, m, k, n);

        let grad_b_storage =
            a.backend()
                .gemm_backward_b::<A>(grad_c_storage, argmax_storage, a_storage, m, k, n);

        let grad_a = Tensor::from_storage(grad_a_storage, a_shape, a.backend().clone());
        let grad_b = Tensor::from_storage(grad_b_storage, b_shape, b.backend().clone());

        (grad_a, grad_b)
    } else {
        // For higher-dimensional cases, we would need more complex logic
        // For now, this handles the common matmul case
        let _ = (ia, ib, iy); // Mark as used
        unimplemented!("Tropical backward only implemented for 2D matmul currently");
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
    fn test_standard_backward_matmul() {
        // Test backward pass for standard matmul
        // C[i,k] = sum_j A[i,j] * B[j,k]
        //
        // For forward: A[2,3] @ B[3,2] -> C[2,2]
        // grad_A = grad_C @ B.T  -> [2,2] @ [2,3] -> [2,3]
        // grad_B = A.T @ grad_C  -> [3,2] @ [2,2] -> [3,2]

        // Column-major: data [1,2,3,4,5,6] for shape [2,3] represents:
        // A = [[1, 3, 5],
        //      [2, 4, 6]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Column-major: data [1,2,3,4,5,6] for shape [3,2] represents:
        // B = [[1, 4],
        //      [2, 5],
        //      [3, 6]]
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        // grad_c is all ones [2, 2] -> [[1, 1], [1, 1]]
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1]; // i, j
        let ib = &[1, 2]; // j, k
        let iy = &[0, 2]; // i, k

        let (grad_a, grad_b) =
            contract_binary_backward::<Standard<f32>, _, _>(&grad_c, &a, &b, None, ia, ib, iy);

        // grad_A = grad_C @ B.T
        // B.T = [[1, 2, 3], [4, 5, 6]]
        // grad_A = [[1,1],[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9],[5,7,9]]
        // In column-major: [5, 5, 7, 7, 9, 9]
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_a.to_vec(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);

        // grad_B = A.T @ grad_C
        // A.T = [[1, 2], [3, 4], [5, 6]]
        // grad_B = [[1,2],[3,4],[5,6]] @ [[1,1],[1,1]] = [[3,3],[7,7],[11,11]]
        // In column-major: [3, 7, 11, 3, 7, 11]
        assert_eq!(grad_b.shape(), &[3, 2]);
        assert_eq!(grad_b.to_vec(), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
    }

    #[test]
    fn test_standard_backward_square_matmul() {
        // Simpler case: 2x2 matrices
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // grad_c is identity matrix
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) =
            contract_binary_backward::<Standard<f32>, _, _>(&grad_c, &a, &b, None, ia, ib, iy);

        // grad_A = grad_C @ B.T = [[1,0],[0,1]] @ [[5,7],[6,8]] = [[5,7],[6,8]]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![5.0, 7.0, 6.0, 8.0]);

        // grad_B = A.T @ grad_C = [[1,3],[2,4]] @ [[1,0],[0,1]] = [[1,3],[2,4]]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_tropical_backward_matmul() {
        // Test backward pass for MaxPlus matmul
        // C[i,j] = max_k (A[i,k] + B[k,j])
        //
        // Column-major: [1,2,3,4] for shape [2,2] → [[1,3],[2,4]]
        // A = [[1, 3], [2, 4]], B = [[1, 3], [2, 4]]
        // C[0,0] = max(1+1, 3+2) = max(2, 5) = 5, argmax = 1
        // C[1,0] = max(2+1, 4+2) = max(3, 6) = 6, argmax = 1
        // C[0,1] = max(1+3, 3+4) = max(4, 7) = 7, argmax = 1
        // C[1,1] = max(2+3, 4+4) = max(5, 8) = 8, argmax = 1

        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // Compute forward to get argmax
        let (c, argmax) = a.gemm_with_argmax::<MaxPlus<f32>>(&b);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(argmax.to_vec(), vec![1, 1, 1, 1]);

        // grad_c is all ones
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) = contract_binary_backward::<MaxPlus<f32>, _, _>(
            &grad_c,
            &a,
            &b,
            Some(&argmax),
            ia,
            ib,
            iy,
        );

        // For tropical backward with argmax all = 1:
        // grad_A[i, argmax[i,j]] += grad_C[i,j]
        // grad_A[0,1] = grad_C[0,0] + grad_C[0,1] = 2
        // grad_A[1,1] = grad_C[1,0] + grad_C[1,1] = 2
        // grad_A = [[0, 2], [0, 2]] in column-major: [0, 0, 2, 2]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![0.0, 0.0, 2.0, 2.0]);

        // grad_B[argmax[i,j], j] += grad_C[i,j]
        // grad_B[1,0] = grad_C[0,0] + grad_C[1,0] = 2
        // grad_B[1,1] = grad_C[0,1] + grad_C[1,1] = 2
        // grad_B = [[0, 0], [2, 2]] in column-major: [0, 2, 0, 2]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![0.0, 2.0, 0.0, 2.0]);
    }

    // Test only with tropical feature, not tropical-kernels, because the optimized
    // tropical-gemm kernels may have different iteration order for small matrices
    #[cfg(all(feature = "tropical", not(feature = "tropical-kernels")))]
    #[test]
    fn test_tropical_backward_different_winners() {
        // Test case where different elements have different winners
        // Column-major: [5,1,1,5] for shape [2,2] → [[5,1],[1,5]]
        // Column-major: [1,5,5,1] for shape [2,2] → [[1,5],[5,1]]
        // A = [[5, 1], [1, 5]], B = [[1, 5], [5, 1]]
        // C[0,0] = max(5+1, 1+5) = max(6, 6) = 6, argmax = 0 (first wins on tie)
        // C[1,0] = max(1+1, 5+5) = max(2, 10) = 10, argmax = 1
        // C[0,1] = max(5+5, 1+1) = max(10, 2) = 10, argmax = 0
        // C[1,1] = max(1+5, 5+1) = max(6, 6) = 6, argmax = 0

        let a = Tensor::<f32, Cpu>::from_data(&[5.0, 1.0, 1.0, 5.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 5.0, 5.0, 1.0], &[2, 2]);

        let (c, argmax) = a.gemm_with_argmax::<MaxPlus<f32>>(&b);
        // Column-major: [6, 10, 10, 6]
        assert_eq!(c.to_vec(), vec![6.0, 10.0, 10.0, 6.0]);
        // Column-major argmax: [0, 1, 0, 0]
        // argmax[0,0]=0, argmax[1,0]=1, argmax[0,1]=0, argmax[1,1]=0
        assert_eq!(argmax.to_vec(), vec![0, 1, 0, 0]);

        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) = contract_binary_backward::<MaxPlus<f32>, _, _>(
            &grad_c,
            &a,
            &b,
            Some(&argmax),
            ia,
            ib,
            iy,
        );

        // argmax (column-major) = [0, 1, 0, 0]
        // grad_A[i, argmax[i,k]] += grad_C[i,k]
        // idx=0 (i=0,k=0): argmax=0, grad_A[0,0] += 1
        // idx=1 (i=1,k=0): argmax=1, grad_A[1,1] += 1
        // idx=2 (i=0,k=1): argmax=0, grad_A[0,0] += 1
        // idx=3 (i=1,k=1): argmax=0, grad_A[1,0] += 1
        // grad_A = [[2, 0], [1, 1]] in column-major: [2, 1, 0, 1]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![2.0, 1.0, 0.0, 1.0]);

        // grad_B[argmax[i,k], k] += grad_C[i,k]
        // idx=0: argmax=0, grad_B[0,0] += 1
        // idx=1: argmax=1, grad_B[1,0] += 1
        // idx=2: argmax=0, grad_B[0,1] += 1
        // idx=3: argmax=0, grad_B[0,1] += 1
        // grad_B = [[1, 2], [1, 0]] in column-major: [1, 1, 2, 0]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![1.0, 1.0, 2.0, 0.0]);
    }
}
