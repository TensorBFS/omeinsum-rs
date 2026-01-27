//! Tests for Backend::contract unified API.

use omeinsum::backend::Backend;
use omeinsum::{Cpu, Standard};

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

    // Column-major: [1,2,3,4] for shape [2,2] -> A = [[1,3],[2,4]] (rows)
    // Column-major: [5,6,7,8] for shape [2,2] -> B = [[5,7],[6,8]]
    // A @ B = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]]
    //       = [[23, 31], [34, 46]]
    // Column-major: [23, 34, 31, 46]
    assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
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
    use omeinsum::MaxPlus;

    let cpu = Cpu::default();

    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];

    let c = cpu.contract::<MaxPlus<f64>>(
        &a, &[2, 2], &[1, 2], &[0, 1],
        &b, &[2, 2], &[1, 2], &[1, 2],
        &[2, 2], &[0, 2],
    );

    // Column-major: [1,2,3,4] -> A = [[1,3],[2,4]]
    // MaxPlus: C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(A[0,0]+B[0,0], A[0,1]+B[1,0]) = max(1+1, 3+2) = 5
    // C[1,0] = max(A[1,0]+B[0,0], A[1,1]+B[1,0]) = max(2+1, 4+2) = 6
    // C[0,1] = max(A[0,0]+B[0,1], A[0,1]+B[1,1]) = max(1+3, 3+4) = 7
    // C[1,1] = max(A[1,0]+B[0,1], A[1,1]+B[1,1]) = max(2+3, 4+4) = 8
    // Column-major: [5, 6, 7, 8]
    assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
}
