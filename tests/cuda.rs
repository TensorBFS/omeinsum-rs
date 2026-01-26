//! CUDA backend tests for GPU tensor operations.
//!
//! # Requirements
//!
//! These tests require:
//! - An NVIDIA GPU
//! - CUDA Toolkit installed (nvcc accessible)
//! - cuTENSOR library installed
//!
//! # Running
//!
//! ```bash
//! cargo test --features cuda
//! ```
//!
//! If CUDA is not available, these tests will not be compiled.

#![cfg(feature = "cuda")]

use omeinsum::backend::{Cuda, CudaStorage};

/// Test that CUDA device initialization works.
#[test]
fn test_cuda_init() {
    let cuda = Cuda::new();
    assert!(cuda.is_ok(), "Failed to initialize CUDA: {:?}", cuda.err());
}

/// Test host-to-device and device-to-host memory transfers (f32).
#[test]
fn test_storage_roundtrip_f32() {
    let cuda = Cuda::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let slice = cuda.device().htod_sync_copy(&data).unwrap();
    let storage = CudaStorage::new(slice, cuda.device().clone());
    assert_eq!(storage.to_vec().unwrap(), data);
}

/// Test host-to-device and device-to-host memory transfers (f64).
#[test]
fn test_storage_roundtrip_f64() {
    let cuda = Cuda::new().unwrap();
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let slice = cuda.device().htod_sync_copy(&data).unwrap();
    let storage = CudaStorage::new(slice, cuda.device().clone());
    assert_eq!(storage.to_vec().unwrap(), data);
}

/// Test matrix multiplication with f32.
///
/// Computes C[i,k] = sum_j A[i,j] * B[j,k]
/// where A is 2x3 and B is 3x2, resulting in C being 2x2.
#[test]
fn test_matmul_f32() {
    let cuda = Cuda::new().unwrap();

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3, row-major)
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B = [[1, 2], [3, 4], [5, 6]]  (3x2, row-major)
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = CudaStorage::new(
        cuda.device().htod_sync_copy(&a_data).unwrap(),
        cuda.device().clone(),
    );
    let b = CudaStorage::new(
        cuda.device().htod_sync_copy(&b_data).unwrap(),
        cuda.device().clone(),
    );

    // C[i,k] = sum_j A[i,j] * B[j,k]
    // Row-major: A is 2x3 with strides [3,1], B is 3x2 with strides [2,1]
    let c = cuda
        .contract::<f32>(
            &a,
            &[2, 3],
            &[3, 1],
            &[0, 1],
            &b,
            &[3, 2],
            &[2, 1],
            &[1, 2],
            &[2, 2],
            &[2, 1],
            &[0, 2],
        )
        .unwrap();

    let result = c.to_vec().unwrap();
    // Expected: [[22, 28], [49, 64]]
    assert!((result[0] - 22.0).abs() < 1e-5, "result[0] = {}", result[0]);
    assert!((result[1] - 28.0).abs() < 1e-5, "result[1] = {}", result[1]);
    assert!((result[2] - 49.0).abs() < 1e-5, "result[2] = {}", result[2]);
    assert!((result[3] - 64.0).abs() < 1e-5, "result[3] = {}", result[3]);
}

/// Test matrix multiplication with f64 (double precision).
#[test]
fn test_matmul_f64() {
    let cuda = Cuda::new().unwrap();

    // 2x2 matrix multiplication
    // A = [[1, 2], [3, 4]]
    let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
    // B = [[5, 6], [7, 8]]
    let b_data = vec![5.0f64, 6.0, 7.0, 8.0];

    let a = CudaStorage::new(
        cuda.device().htod_sync_copy(&a_data).unwrap(),
        cuda.device().clone(),
    );
    let b = CudaStorage::new(
        cuda.device().htod_sync_copy(&b_data).unwrap(),
        cuda.device().clone(),
    );

    // C[i,k] = sum_j A[i,j] * B[j,k]
    let c = cuda
        .contract::<f64>(
            &a,
            &[2, 2],
            &[2, 1],
            &[0, 1],
            &b,
            &[2, 2],
            &[2, 1],
            &[1, 2],
            &[2, 2],
            &[2, 1],
            &[0, 2],
        )
        .unwrap();

    let result = c.to_vec().unwrap();
    // Expected: [[19, 22], [43, 50]]
    // [1,2] dot [5,7] = 5+14 = 19
    // [1,2] dot [6,8] = 6+16 = 22
    // [3,4] dot [5,7] = 15+28 = 43
    // [3,4] dot [6,8] = 18+32 = 50
    assert!(
        (result[0] - 19.0).abs() < 1e-10,
        "result[0] = {}",
        result[0]
    );
    assert!(
        (result[1] - 22.0).abs() < 1e-10,
        "result[1] = {}",
        result[1]
    );
    assert!(
        (result[2] - 43.0).abs() < 1e-10,
        "result[2] = {}",
        result[2]
    );
    assert!(
        (result[3] - 50.0).abs() < 1e-10,
        "result[3] = {}",
        result[3]
    );
}

/// Test vector inner product (dot product).
///
/// Computes c = sum_i A[i] * B[i]
#[test]
fn test_inner_product() {
    let cuda = Cuda::new().unwrap();

    // A = [1, 2, 3, 4]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    // B = [2, 3, 4, 5]
    let b_data = vec![2.0f32, 3.0, 4.0, 5.0];

    let a = CudaStorage::new(
        cuda.device().htod_sync_copy(&a_data).unwrap(),
        cuda.device().clone(),
    );
    let b = CudaStorage::new(
        cuda.device().htod_sync_copy(&b_data).unwrap(),
        cuda.device().clone(),
    );

    // c = sum_i A[i] * B[i]
    // shapes: A[4], B[4], C[1] (scalar as 1-element tensor)
    let c = cuda
        .contract::<f32>(
            &a,
            &[4],
            &[1],
            &[0],
            &b,
            &[4],
            &[1],
            &[0],
            &[1],
            &[1],
            &[], // scalar output (no free indices)
        )
        .unwrap();

    let result = c.to_vec().unwrap();
    // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 40.0).abs() < 1e-5,
        "inner product = {}",
        result[0]
    );
}

/// Test vector outer product.
///
/// Computes C[i,j] = A[i] * B[j]
#[test]
fn test_outer_product() {
    let cuda = Cuda::new().unwrap();

    // A = [1, 2, 3]
    let a_data = vec![1.0f32, 2.0, 3.0];
    // B = [4, 5]
    let b_data = vec![4.0f32, 5.0];

    let a = CudaStorage::new(
        cuda.device().htod_sync_copy(&a_data).unwrap(),
        cuda.device().clone(),
    );
    let b = CudaStorage::new(
        cuda.device().htod_sync_copy(&b_data).unwrap(),
        cuda.device().clone(),
    );

    // C[i,j] = A[i] * B[j] (no contraction, just outer product)
    let c = cuda
        .contract::<f32>(
            &a,
            &[3],
            &[1],
            &[0],
            &b,
            &[2],
            &[1],
            &[1],
            &[3, 2],
            &[2, 1],
            &[0, 1],
        )
        .unwrap();

    let result = c.to_vec().unwrap();
    // Expected: [[4, 5], [8, 10], [12, 15]]
    // Row-major: [4, 5, 8, 10, 12, 15]
    assert_eq!(result.len(), 6);
    assert!((result[0] - 4.0).abs() < 1e-5);
    assert!((result[1] - 5.0).abs() < 1e-5);
    assert!((result[2] - 8.0).abs() < 1e-5);
    assert!((result[3] - 10.0).abs() < 1e-5);
    assert!((result[4] - 12.0).abs() < 1e-5);
    assert!((result[5] - 15.0).abs() < 1e-5);
}

/// Test batch matrix multiplication.
///
/// Computes C[b,i,k] = sum_j A[b,i,j] * B[b,j,k]
/// where b is the batch dimension.
#[test]
fn test_batch_matmul() {
    let cuda = Cuda::new().unwrap();

    // Batch of 2, each 2x2 matrix
    // A[0] = [[1, 2], [3, 4]], A[1] = [[5, 6], [7, 8]]
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    // B[0] = [[1, 0], [0, 1]], B[1] = [[2, 0], [0, 2]]  (identity and 2*identity)
    let b_data = vec![
        1.0f32, 0.0, 0.0, 1.0, // batch 0: identity
        2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
    ];

    let a = CudaStorage::new(
        cuda.device().htod_sync_copy(&a_data).unwrap(),
        cuda.device().clone(),
    );
    let b = CudaStorage::new(
        cuda.device().htod_sync_copy(&b_data).unwrap(),
        cuda.device().clone(),
    );

    // C[b,i,k] = sum_j A[b,i,j] * B[b,j,k]
    // Shapes: A[2,2,2], B[2,2,2], C[2,2,2]
    // Row-major strides: A[4,2,1], B[4,2,1], C[4,2,1]
    let c = cuda
        .contract::<f32>(
            &a,
            &[2, 2, 2],
            &[4, 2, 1],
            &[0, 1, 2], // b, i, j
            &b,
            &[2, 2, 2],
            &[4, 2, 1],
            &[0, 2, 3], // b, j, k
            &[2, 2, 2],
            &[4, 2, 1],
            &[0, 1, 3], // b, i, k
        )
        .unwrap();

    let result = c.to_vec().unwrap();
    // Expected:
    // C[0] = A[0] @ I = [[1, 2], [3, 4]]
    // C[1] = A[1] @ 2I = [[10, 12], [14, 16]]
    assert_eq!(result.len(), 8);
    // Batch 0
    assert!((result[0] - 1.0).abs() < 1e-5, "C[0,0,0] = {}", result[0]);
    assert!((result[1] - 2.0).abs() < 1e-5, "C[0,0,1] = {}", result[1]);
    assert!((result[2] - 3.0).abs() < 1e-5, "C[0,1,0] = {}", result[2]);
    assert!((result[3] - 4.0).abs() < 1e-5, "C[0,1,1] = {}", result[3]);
    // Batch 1
    assert!((result[4] - 10.0).abs() < 1e-5, "C[1,0,0] = {}", result[4]);
    assert!((result[5] - 12.0).abs() < 1e-5, "C[1,0,1] = {}", result[5]);
    assert!((result[6] - 14.0).abs() < 1e-5, "C[1,1,0] = {}", result[6]);
    assert!((result[7] - 16.0).abs() < 1e-5, "C[1,1,1] = {}", result[7]);
}

/// Test that CudaStorage correctly reports length and emptiness.
#[test]
fn test_storage_len() {
    let cuda = Cuda::new().unwrap();

    let data = vec![1.0f32, 2.0, 3.0];
    let slice = cuda.device().htod_sync_copy(&data).unwrap();
    let storage = CudaStorage::new(slice, cuda.device().clone());

    assert_eq!(storage.len(), 3);
    assert!(!storage.is_empty());
}
