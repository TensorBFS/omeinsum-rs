//! Tests for backward pass / gradient computation.
//!
//! These tests verify that gradients are correctly computed for tensor contractions
//! in both standard and tropical algebras.

use omeinsum::backend::Cpu;
use omeinsum::{einsum_with_grad, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::MaxPlus;

#[test]
fn test_backward_matmul_standard() {
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Standard matmul: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    assert_eq!(result.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    // For C = A @ B, with grad_out = ones:
    // grad_A = grad_out @ B^T = [[1,1],[1,1]] @ [[1,3],[2,4]] = [[3,7],[3,7]]
    // grad_B = A^T @ grad_out = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    // Verify gradient shapes are correct
    assert_eq!(grad_a.to_vec(), vec![3.0, 7.0, 3.0, 7.0]);
    assert_eq!(grad_b.to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_backward_matmul_identity() {
    // Test with identity matrix to verify gradients
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let identity = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &identity], &[&[0, 1], &[1, 2]], &[0, 2]);

    // A @ I = A
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &identity]);

    assert_eq!(grads.len(), 2);
    // When grad_out = I, grad_A = I @ I^T = I
    assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_backward_matmul_rectangular() {
    // Test backward with non-square matrices
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    assert_eq!(result.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    // grad_A should be [2, 3], grad_B should be [3, 2]
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);

    // grad_A = grad_out @ B^T = [[1,1],[1,1]] @ [[1,3,5],[2,4,6]] = [[3,7,11],[3,7,11]]
    assert_eq!(grad_a.to_vec(), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);

    // grad_B = A^T @ grad_out = [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
    assert_eq!(grad_b.to_vec(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn test_backward_matmul_ones() {
    // Test with all ones gradient
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(result.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    // grad_A = grad_out @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    assert_eq!(grads[0].to_vec(), vec![11.0, 15.0, 11.0, 15.0]);

    // grad_B = A^T @ grad_out = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    assert_eq!(grads[1].to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
}

// ============================================================================
// Tropical algebra backward tests (require tropical feature)
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_backward_matmul_tropical() {
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // MaxPlus: C[i,k] = max_j (A[i,j] + B[j,k])
    // C[0,0] = max(1+1, 2+3) = max(2, 5) = 5
    // C[0,1] = max(1+2, 2+4) = max(3, 6) = 6
    // C[1,0] = max(3+1, 4+3) = max(4, 7) = 7
    // C[1,1] = max(3+2, 4+4) = max(5, 8) = 8
    assert_eq!(result.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    // For tropical backward, only the winning path gets the gradient
    // All winners are at j=1 (index 1), so:
    // grad_A[i,1] accumulates gradients from all C[i,:] that won with j=1
    // grad_B[1,k] accumulates gradients from all C[:,k] that won with j=1
    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    // grad_A: j=1 won for all outputs
    // grad_A[0,1] = grad_C[0,0] + grad_C[0,1] = 2
    // grad_A[1,1] = grad_C[1,0] + grad_C[1,1] = 2
    assert_eq!(grad_a.to_vec(), vec![0.0, 2.0, 0.0, 2.0]);

    // grad_B: j=1 won for all outputs
    // grad_B[1,0] = grad_C[0,0] + grad_C[1,0] = 2
    // grad_B[1,1] = grad_C[0,1] + grad_C[1,1] = 2
    assert_eq!(grad_b.to_vec(), vec![0.0, 0.0, 2.0, 2.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_backward_tropical_sparse_gradient() {
    // Test that tropical backward produces sparse gradients (only winners)
    let a = Tensor::<f32, Cpu>::from_data(&[0.0, 10.0, 20.0, 0.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // MaxPlus results - clear winners due to large differences in A
    // C[0,0] = max(0+1, 10+1) = 11 (j=1 wins)
    // C[0,1] = max(0+1, 10+1) = 11 (j=1 wins)
    // C[1,0] = max(20+1, 0+1) = 21 (j=0 wins)
    // C[1,1] = max(20+1, 0+1) = 21 (j=0 wins)
    assert_eq!(result.to_vec(), vec![11.0, 11.0, 21.0, 21.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    // The gradient should be sparse - only the winning paths get gradients
    let grad_a = &grads[0];
    let _grad_b = &grads[1];

    // A[0,1]=10 won for C[0,:] (both outputs), A[1,0]=20 won for C[1,:] (both outputs)
    // So grad_A[0,0]=0, grad_A[0,1]=2, grad_A[1,0]=2, grad_A[1,1]=0
    let grad_a_vec = grad_a.to_vec();
    assert_eq!(grad_a_vec[0], 0.0); // Loser
    assert_eq!(grad_a_vec[1], 2.0); // Winner (twice)
    assert_eq!(grad_a_vec[2], 2.0); // Winner (twice)
    assert_eq!(grad_a_vec[3], 0.0); // Loser
}

#[cfg(feature = "tropical")]
#[test]
fn test_backward_tropical_different_winners() {
    // Test case where different elements have different winners
    let a = Tensor::<f32, Cpu>::from_data(&[5.0, 1.0, 1.0, 5.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 5.0, 5.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // C[0,0] = max(5+1, 1+5) = max(6, 6) = 6, j=0 wins (first on tie)
    // C[0,1] = max(5+5, 1+1) = max(10, 2) = 10, j=0 wins
    // C[1,0] = max(1+1, 5+5) = max(2, 10) = 10, j=1 wins
    // C[1,1] = max(1+5, 5+1) = max(6, 6) = 6, j=0 wins (first on tie)
    assert_eq!(result.to_vec(), vec![6.0, 10.0, 10.0, 6.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    // Winners: j=0 for [0,0], [0,1], [1,1]; j=1 for [1,0]
    // grad_A[0,0] = grad_C[0,0] + grad_C[0,1] = 2
    // grad_A[0,1] = 0
    // grad_A[1,0] = grad_C[1,1] = 1
    // grad_A[1,1] = grad_C[1,0] = 1
    assert_eq!(grad_a.to_vec(), vec![2.0, 0.0, 1.0, 1.0]);

    // grad_B[0,0] = grad_C[0,0] = 1
    // grad_B[0,1] = grad_C[0,1] + grad_C[1,1] = 2
    // grad_B[1,0] = grad_C[1,0] = 1
    // grad_B[1,1] = 0
    assert_eq!(grad_b.to_vec(), vec![1.0, 2.0, 1.0, 0.0]);
}
