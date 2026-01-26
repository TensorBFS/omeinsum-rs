//! Additional tests for code coverage.
//!
//! These tests target specific code paths that may not be covered by other tests.

use omeinsum::backend::{Backend, Cpu};
use omeinsum::tensor::TensorView;
use omeinsum::{einsum, einsum_with_grad, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::{MaxMul, MaxPlus, MinPlus};

// ============================================================================
// TensorView tests
// ============================================================================

#[test]
fn test_tensor_view_basic() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let view = TensorView::new(&t);

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.numel(), 6);
    assert!(view.is_contiguous());
    assert_eq!(view.as_tensor().shape(), &[2, 3]);
}

#[test]
fn test_tensor_view_from_trait() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let view: TensorView<f32, Cpu> = (&t).into();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.numel(), 4);
}

#[test]
fn test_tensor_view_non_contiguous() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t_permuted = t.permute(&[1, 0]); // Now [3, 2] with non-contiguous strides
    let view = TensorView::new(&t_permuted);

    assert_eq!(view.shape(), &[3, 2]);
    assert!(!view.is_contiguous());
}

// ============================================================================
// Backend tests
// ============================================================================

#[test]
fn test_cpu_backend_name() {
    assert_eq!(Cpu::name(), "cpu");
}

#[test]
fn test_cpu_backend_synchronize() {
    let cpu = Cpu;
    cpu.synchronize(); // Should be no-op
}

#[test]
fn test_cpu_backend_alloc() {
    let cpu = Cpu;
    let storage: Vec<f64> = cpu.alloc(10);
    assert_eq!(storage.len(), 10);
    assert!(storage.iter().all(|&x| x == 0.0));
}

#[test]
fn test_cpu_backend_from_slice() {
    let cpu = Cpu;
    let data = [1.0f32, 2.0, 3.0];
    let storage = cpu.from_slice(&data);
    assert_eq!(storage, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_storage_is_empty() {
    let empty: Vec<f32> = vec![];
    let non_empty: Vec<f32> = vec![1.0, 2.0];

    assert!(empty.is_empty());
    assert!(!non_empty.is_empty());
}

#[test]
fn test_storage_get_set() {
    use omeinsum::backend::Storage;

    let mut storage: Vec<f64> = vec![1.0, 2.0, 3.0];

    assert_eq!(storage.get(0), 1.0);
    assert_eq!(storage.get(1), 2.0);

    storage.set(1, 5.0);
    assert_eq!(storage.get(1), 5.0);
}

#[test]
fn test_storage_zeros() {
    use omeinsum::backend::Storage;

    let zeros: Vec<f32> = Vec::zeros(5);
    assert_eq!(zeros.len(), 5);
    assert!(zeros.iter().all(|&x| x == 0.0));
}

// ============================================================================
// Tensor operations tests
// ============================================================================

#[test]
fn test_tensor_clone() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = t.clone();
    assert_eq!(t.to_vec(), t2.to_vec());
    assert_eq!(t.shape(), t2.shape());
}

#[test]
fn test_tensor_debug() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let debug_str = format!("{:?}", t);
    assert!(debug_str.contains("Tensor"));
    assert!(debug_str.contains("shape"));
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::<f64, Cpu>::zeros(&[3, 4]);
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(t.numel(), 12);
    assert!(t.to_vec().iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_from_storage() {
    let storage = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<f32, Cpu>::from_storage(storage, &[2, 2], Cpu);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_backend() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let _backend = t.backend();
}

#[test]
fn test_tensor_storage() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let storage = t.storage();
    assert!(storage.is_some());
    assert_eq!(storage.unwrap().len(), 3);
}

#[test]
fn test_tensor_strides() {
    // Test that strides are computed correctly for column-major layout
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: stride for first dim = 1, stride for second dim = 2
    assert_eq!(t.strides(), &[1, 2]);
}

#[test]
fn test_tensor_reshape_various() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // Reshape to 1D
    let t1 = t.reshape(&[6]);
    assert_eq!(t1.shape(), &[6]);

    // Reshape to 3D
    let t3 = t.reshape(&[1, 2, 3]);
    assert_eq!(t3.shape(), &[1, 2, 3]);
}

#[test]
fn test_tensor_contiguous_already() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(t.is_contiguous());

    let t2 = t.contiguous();
    assert!(t2.is_contiguous());
    assert_eq!(t.to_vec(), t2.to_vec());
}

#[test]
fn test_tensor_sum_various_shapes() {
    // 1D tensor
    let t1 = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(t1.sum::<Standard<f64>>(), 6.0);

    // 3D tensor
    let t3 = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(t3.sum::<Standard<f64>>(), 36.0);
}

#[test]
fn test_tensor_sum_axis_various() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // Sum along axis 0
    let s0 = t.sum_axis::<Standard<f64>>(0);
    assert_eq!(s0.shape(), &[3]);

    // Sum along axis 1
    let s1 = t.sum_axis::<Standard<f64>>(1);
    assert_eq!(s1.shape(), &[2]);
}

#[test]
fn test_tensor_diagonal_3x3() {
    // 3x3 square matrix diagonal
    // Column-major: [1,4,7,2,5,8,3,6,9] for:
    //   [[1,2,3],
    //    [4,5,6],
    //    [7,8,9]]
    let t = Tensor::<f64, Cpu>::from_data(
        &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        &[3, 3]
    );
    let diag = t.diagonal();
    assert_eq!(diag.shape(), &[3]);
    // Diagonal elements: [1,5,9]
    assert_eq!(diag.to_vec(), vec![1.0, 5.0, 9.0]);
}

// ============================================================================
// Einsum tests for coverage
// ============================================================================

#[test]
fn test_einsum_scalar_output() {
    // Contract to scalar: ij,ij->
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[]);
    assert_eq!(c.shape(), &[]);
    assert_eq!(c.to_vec(), vec![10.0]); // 1+2+3+4 = 10
}

#[test]
fn test_einsum_with_grad_single_tensor() {
    // Single tensor should return unchanged gradient
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[0, 1]);

    assert_eq!(result.to_vec(), a.to_vec());

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);
    assert_eq!(grads.len(), 1);
}

#[test]
fn test_einsum_batch_contraction() {
    // Batch matrix multiply: bij,bjk->bik
    let a = Tensor::<f64, Cpu>::from_data(
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], // 2 batches of 2x2 identity-ish
        &[2, 2, 2],
    );
    let b = Tensor::<f64, Cpu>::from_data(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    );

    let c = einsum::<Standard<f64>, _, _>(
        &[&a, &b],
        &[&[0, 1, 2], &[0, 2, 3]], // b=0, i=1, j=2, k=3
        &[0, 1, 3],                 // output: b, i, k
    );

    assert_eq!(c.shape(), &[2, 2, 2]);
}

// ============================================================================
// Tropical algebra tests for coverage
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_maxplus_is_zero() {
    use omeinsum::algebra::Semiring;

    let zero = MaxPlus::<f32>::zero();
    let one = MaxPlus::<f32>::one();
    let val = MaxPlus(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_is_zero() {
    use omeinsum::algebra::Semiring;

    let zero = MinPlus::<f32>::zero();
    let one = MinPlus::<f32>::one();
    let val = MinPlus(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_is_zero() {
    use omeinsum::algebra::Semiring;

    let zero = MaxMul::<f32>::zero();
    let one = MaxMul::<f32>::one();
    let val = MaxMul(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_add_backward() {
    use omeinsum::algebra::Algebra;

    // MaxPlus add_backward
    let a = MaxPlus(3.0f32);
    let b = MaxPlus(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(1)); // b wins (index 1)
    assert_eq!(ga, 0.0);
    assert_eq!(gb, 1.0);

    let (ga, gb) = a.add_backward(b, 1.0, Some(0)); // a wins (index 0)
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 0.0);

    // MinPlus add_backward (same logic)
    let a = MinPlus(3.0f32);
    let b = MinPlus(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(0)); // a wins (smaller)
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 0.0);

    // MaxMul add_backward
    let a = MaxMul(3.0f32);
    let b = MaxMul(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(1)); // b wins
    assert_eq!(ga, 0.0);
    assert_eq!(gb, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_mul_backward() {
    use omeinsum::algebra::Algebra;

    // MaxPlus mul_backward (multiplication is addition, so both get gradient)
    let a = MaxPlus(3.0f32);
    let b = MaxPlus(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 1.0);

    // MinPlus mul_backward
    let a = MinPlus(3.0f32);
    let b = MinPlus(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 1.0);

    // MaxMul mul_backward (multiplication is real multiplication)
    let a = MaxMul(3.0f32);
    let b = MaxMul(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 5.0); // grad_a = grad_out * b
    assert_eq!(gb, 3.0); // grad_b = grad_out * a
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_needs_argmax() {
    use omeinsum::algebra::Algebra;

    assert!(MaxPlus::<f32>::needs_argmax());
    assert!(MinPlus::<f32>::needs_argmax());
    assert!(MaxMul::<f32>::needs_argmax());
    assert!(!Standard::<f32>::needs_argmax());
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_f64_operations() {
    use omeinsum::algebra::Semiring;

    // MaxPlus f64
    let a = MaxPlus(2.0f64);
    let b = MaxPlus(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 3.0);
    assert_eq!(a.mul(b).to_scalar(), 5.0);

    // MinPlus f64
    let a = MinPlus(2.0f64);
    let b = MinPlus(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 2.0);
    assert_eq!(a.mul(b).to_scalar(), 5.0);

    // MaxMul f64
    let a = MaxMul(2.0f64);
    let b = MaxMul(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 3.0);
    assert_eq!(a.mul(b).to_scalar(), 6.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_gemm_f64() {
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // MaxPlus f64
    let c = a.gemm::<MaxPlus<f64>>(&b);
    assert_eq!(c.shape(), &[2, 2]);

    // MinPlus f64
    let c = a.gemm::<MinPlus<f64>>(&b);
    assert_eq!(c.shape(), &[2, 2]);

    // MaxMul f64
    let c = a.gemm::<MaxMul<f64>>(&b);
    assert_eq!(c.shape(), &[2, 2]);
}

// ============================================================================
// Standard algebra tests for coverage
// ============================================================================

#[test]
fn test_standard_is_zero() {
    use omeinsum::algebra::Semiring;

    let zero = Standard::<f32>::zero();
    let one = Standard::<f32>::one();
    let val = Standard(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[test]
fn test_standard_f64() {
    use omeinsum::algebra::Semiring;

    let a = Standard(2.0f64);
    let b = Standard(3.0f64);

    assert_eq!(a.add(b).to_scalar(), 5.0);
    assert_eq!(a.mul(b).to_scalar(), 6.0);
    assert_eq!(Standard::<f64>::zero().to_scalar(), 0.0);
    assert_eq!(Standard::<f64>::one().to_scalar(), 1.0);
}

// ============================================================================
// Complex number tests
// ============================================================================

#[test]
fn test_complex_tensor_basic() {
    use num_complex::Complex64 as C64;

    let t = Tensor::<C64, Cpu>::from_data(
        &[C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(1.0, 1.0), C64::new(2.0, 0.0)],
        &[2, 2],
    );

    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.numel(), 4);
}

#[test]
fn test_complex_gemm() {
    use num_complex::Complex64 as C64;

    let a = Tensor::<C64, Cpu>::from_data(
        &[C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        &[2, 2],
    );
    let b = Tensor::<C64, Cpu>::from_data(
        &[C64::new(1.0, 1.0), C64::new(2.0, 0.0), C64::new(0.0, 1.0), C64::new(3.0, 0.0)],
        &[2, 2],
    );

    // Identity * B = B
    let c = a.gemm::<Standard<C64>>(&b);
    assert_eq!(c.shape(), &[2, 2]);
}

// ============================================================================
// Batched GEMM tests
// ============================================================================

#[test]
fn test_gemm_batched_standard() {
    let cpu = Cpu;

    // 2 batches of 2x2 matrices
    let a = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0,    // batch 1
    ];
    let b = vec![
        1.0f32, 0.0, 0.0, 1.0, // batch 0: identity
        1.0, 0.0, 0.0, 1.0,    // batch 1: identity
    ];

    let c = cpu.gemm_batched::<Standard<f32>>(&a, 2, 2, 2, &b, 2);
    assert_eq!(c.len(), 8);

    // Each batch should equal original (multiplied by identity)
    assert_eq!(&c[0..4], &a[0..4]);
    assert_eq!(&c[4..8], &a[4..8]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_gemm_batched_tropical() {
    let cpu = Cpu;

    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // zeros

    let c = cpu.gemm_batched::<MaxPlus<f32>>(&a, 2, 2, 2, &b, 2);
    assert_eq!(c.len(), 8);
}

#[cfg(feature = "tropical")]
#[test]
fn test_gemm_batched_with_argmax() {
    let cpu = Cpu;

    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];

    let (c, argmax) = cpu.gemm_batched_with_argmax::<MaxPlus<f32>>(&a, 2, 2, 2, &b, 2);
    assert_eq!(c.len(), 8);
    assert_eq!(argmax.len(), 8);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_tensor_1x1() {
    let t = Tensor::<f64, Cpu>::from_data(&[42.0], &[1, 1]);
    assert_eq!(t.shape(), &[1, 1]);
    assert_eq!(t.get(0), 42.0);
    assert_eq!(t.sum::<Standard<f64>>(), 42.0);
    assert_eq!(t.diagonal().to_vec(), vec![42.0]);
}

#[test]
fn test_einsum_identity() {
    // i->i (identity)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0]], &[0]);
    assert_eq!(c.to_vec(), a.to_vec());
}

#[test]
fn test_einsum_transpose() {
    // ij->ji (transpose)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[1, 0]);
    assert_eq!(c.shape(), &[3, 2]);
}

#[test]
fn test_einsum_outer_product() {
    // i,j->ij (outer product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[1]], &[0, 1]);
    assert_eq!(c.shape(), &[2, 3]);
}
