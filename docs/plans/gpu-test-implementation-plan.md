# GPU Test Implementation Plan

This plan details the work needed to bring GPU test coverage to parity with CPU tests.

## Executive Summary

| Category | CPU Tests | GPU Tests (Existing) | To Implement | Priority |
|----------|-----------|----------------------|--------------|----------|
| Unary Operations | 27 | 0 | 27 | HIGH |
| Binary Rules | 12 | 3 | 9 | MEDIUM |
| Backward/Gradient | 25 | 4 | 12 | HIGH |
| OMEinsum Compat | 40 | 0 | 20 | MEDIUM |
| Einsum Core | 35 | 10 | 15 | HIGH |
| Coverage Tests | 45 | 2 | 10 | LOW |
| CPU-GPU Consistency | 0 | 0 | 15 | HIGH |
| Edge Cases | 8 | 2 | 6 | HIGH |

**Total: ~114 new GPU tests to implement**

---

## Phase 1: High Priority - Core Functionality (Week 1)

### 1.1 Unary Operations on GPU

These are currently **completely missing** from GPU tests. Port from `tests/unary_ops.rs`:

```rust
// File: tests/cuda.rs (add to existing file)

// ============================================================================
// GPU Unary Operation Tests
// ============================================================================

#[test]
fn test_cuda_unary_trace_2x2() {
    // ii -> (trace of 2x2 matrix)
}

#[test]
fn test_cuda_unary_trace_3x3() {
    // ii -> (trace of 3x3 matrix)
}

#[test]
fn test_cuda_unary_diagonal_2x2() {
    // ii -> i (extract diagonal)
}

#[test]
fn test_cuda_unary_diagonal_3x3() {
    // ii -> i
}

#[test]
fn test_cuda_unary_sum_all_2d() {
    // ij -> (sum all elements)
}

#[test]
fn test_cuda_unary_sum_axis_0() {
    // ij -> j (sum over first axis)
}

#[test]
fn test_cuda_unary_sum_axis_1() {
    // ij -> i (sum over second axis)
}

#[test]
fn test_cuda_unary_sum_3d_to_1d() {
    // ijk -> i (sum over j and k)
}

#[test]
fn test_cuda_unary_transpose_2x2() {
    // ij -> ji
}

#[test]
fn test_cuda_unary_transpose_2x3() {
    // ij -> ji (non-square)
}

#[test]
fn test_cuda_unary_permute_3d() {
    // ijk -> kji
}

#[test]
fn test_cuda_unary_permute_3d_partial() {
    // ijk -> jik
}

#[test]
fn test_cuda_unary_identity_2d() {
    // ij -> ij (no-op)
}

#[test]
fn test_cuda_unary_partial_trace_4d() {
    // ijjk -> ik
}

#[test]
fn test_cuda_unary_diag_extract_and_embed() {
    // ii -> ii (project to diagonal matrix)
}

#[test]
fn test_cuda_unary_duplicate_vector_to_diagonal() {
    // i -> ii
}

#[test]
fn test_cuda_unary_repeat_add_dimension() {
    // i -> ij (broadcast)
}

#[test]
fn test_cuda_unary_repeat_prepend_dimension() {
    // i -> ji
}
```

### 1.2 CPU-GPU Consistency Tests

**Completely missing** - critical for verifying correctness:

```rust
// File: tests/cuda.rs (new section)

// ============================================================================
// CPU-GPU Consistency Tests
// ============================================================================

use omeinsum::backend::Cpu;

/// Helper to compare GPU and CPU results
fn assert_gpu_cpu_equal_f32(gpu: &[f32], cpu: &[f32], tol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "Length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!((g - c).abs() < tol,
            "Mismatch at {}: GPU={}, CPU={}", i, g, c);
    }
}

fn assert_gpu_cpu_equal_f64(gpu: &[f64], cpu: &[f64], tol: f64) {
    assert_eq!(gpu.len(), cpu.len(), "Length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!((g - c).abs() < tol,
            "Mismatch at {}: GPU={}, CPU={}", i, g, c);
    }
}

#[test]
fn test_consistency_matmul_f32() {
    // Compare GPU and CPU matrix multiplication
    let cuda = Cuda::new().unwrap();

    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

    // CPU
    let a_cpu = Tensor::<f32, Cpu>::from_data(&data_a, &[2, 2]);
    let b_cpu = Tensor::<f32, Cpu>::from_data(&data_b, &[2, 2]);
    let c_cpu = einsum::<Standard<f32>, _, _>(
        &[&a_cpu, &b_cpu], &[&[0, 1], &[1, 2]], &[0, 2]
    );

    // GPU
    let a_gpu = Tensor::<f32, Cuda>::from_data_with_backend(&data_a, &[2, 2], cuda.clone());
    let b_gpu = Tensor::<f32, Cuda>::from_data_with_backend(&data_b, &[2, 2], cuda);
    let c_gpu = einsum::<Standard<f32>, _, _>(
        &[&a_gpu, &b_gpu], &[&[0, 1], &[1, 2]], &[0, 2]
    );

    assert_gpu_cpu_equal_f32(&c_gpu.to_vec(), &c_cpu.to_vec(), 1e-5);
}

#[test]
fn test_consistency_matmul_f64() {
    // Same as above but f64
}

#[test]
fn test_consistency_batch_matmul() {
    // Compare batched matmul: bij,bjk->bik
}

#[test]
fn test_consistency_tensor_contraction_3d() {
    // Compare ijk,kl->ijl
}

#[test]
fn test_consistency_inner_product() {
    // Compare i,i->
}

#[test]
fn test_consistency_outer_product() {
    // Compare i,j->ij
}

#[test]
fn test_consistency_trace() {
    // Compare ii->
}

#[test]
fn test_consistency_transpose() {
    // Compare ij->ji
}

#[test]
fn test_consistency_chain_3_tensors() {
    // Compare ij,jk,kl->il
}

#[test]
fn test_consistency_einsum_rectangular() {
    // Compare [2,3] @ [3,4] -> [2,4]
}

#[test]
fn test_consistency_large_tensors() {
    // Compare with larger tensors (100x100)
}

#[test]
fn test_consistency_random_data() {
    // Use seeded random data for more thorough testing
}

#[test]
fn test_consistency_gradient_matmul() {
    // Compare gradient computation on CPU vs GPU
}

#[test]
fn test_consistency_gradient_rectangular() {
    // Compare gradient for non-square matrices
}

#[test]
fn test_consistency_complex_matmul() {
    // Compare complex number matrix multiplication
}
```

### 1.3 Additional Gradient Tests on GPU

Port missing gradient tests from `tests/backward.rs`:

```rust
#[test]
fn test_cuda_backward_matmul_identity() {
    // Test with identity matrix
}

#[test]
fn test_cuda_backward_matmul_ones() {
    // Test with all ones gradient
}

#[test]
fn test_cuda_backward_large_values_f64() {
    // Test with large values where precision matters
}

#[test]
fn test_cuda_backward_all_matmul_transposes() {
    // Test ij,jk->ik, ij,kj->ik, ji,jk->ik, ji,kj->ik
}

#[test]
fn test_cuda_backward_3tensor_chain() {
    // Test gradient through A @ B @ C
}
```

---

## Phase 2: Medium Priority - Extended Patterns (Week 2)

### 2.1 OMEinsum Compatibility Tests on GPU

Port key tests from `tests/omeinsum_compat.rs`:

```rust
#[test]
fn test_cuda_compat_matrix_multiplication() {
    // ein"ij,jk -> ik"
}

#[test]
fn test_cuda_compat_matrix_multiplication_transposed_output() {
    // ein"ij,jk -> ki"
}

#[test]
fn test_cuda_compat_hadamard_product() {
    // ein"ij,ij -> ij"
}

#[test]
fn test_cuda_compat_vector_matrix_contraction() {
    // ein"j,jk -> k"
}

#[test]
fn test_cuda_compat_matrix_vector_contraction() {
    // ein"ij,j -> i"
}

#[test]
fn test_cuda_compat_batch_matrix_multiplication() {
    // ein"bij,bjk -> bik"
}

#[test]
fn test_cuda_compat_three_matrix_chain() {
    // ein"ij,jk,kl -> il"
}

#[test]
fn test_cuda_compat_star_contraction() {
    // ein"ai,bi,ci -> abc"
}

#[test]
fn test_cuda_compat_tensor_network_contraction() {
    // ein"ij,jk,ki -> " (trace of product)
}

#[test]
fn test_cuda_compat_partial_trace_4d() {
    // ein"ijjk -> ik"
}

#[test]
fn test_cuda_compat_higher_dimensional() {
    // ein"ijk,jkl -> il"
}

#[test]
fn test_cuda_compat_4d_batch_contraction() {
    // ein"abij,abjk -> abik"
}
```

### 2.2 Einsum Core Pattern Tests on GPU

Port from `tests/einsum_core.rs`:

```rust
#[test]
fn test_cuda_einsum_identity_4d() {
    // ijkl -> ijkl
}

#[test]
fn test_cuda_einsum_matrix_vector() {
    // ij,j -> i
}

#[test]
fn test_cuda_einsum_contract_to_scalar() {
    // ij,ij -> (Frobenius inner product)
}

#[test]
fn test_cuda_einsum_trace_4d() {
    // ijji -> (double trace)
}

#[test]
fn test_cuda_einsum_partial_trace() {
    // ijjk -> ik
}

#[test]
fn test_cuda_einsum_diag_extract() {
    // ijjk -> ijk
}

#[test]
fn test_cuda_einsum_permute_2d() {
    // ij -> ji
}

#[test]
fn test_cuda_einsum_permute_4d() {
    // ijkl -> jkil
}

#[test]
fn test_cuda_einsum_tensor_contraction_4d_2d() {
    // ijkl,jk -> il
}

#[test]
fn test_cuda_einsum_star_contraction() {
    // ai,ai,ai -> a
}

#[test]
fn test_cuda_einsum_index_sum() {
    // ijk -> ij (sum over k)
}

#[test]
fn test_cuda_einsum_hadamard() {
    // ij,ij -> ij
}

#[test]
fn test_cuda_einsum_outer_product_4d() {
    // ij,kl -> ijkl
}

#[test]
fn test_cuda_einsum_project_to_diag() {
    // ii -> ii
}

#[test]
fn test_cuda_einsum_large_contraction() {
    // Stress test with larger tensors
}
```

---

## Phase 3: Edge Cases and Error Handling (Week 2-3)

### 3.1 Edge Case Tests

```rust
#[test]
fn test_cuda_edge_scalar_contraction() {
    // 0-dimensional output tensor
}

#[test]
fn test_cuda_edge_size_one_dimension() {
    // Tensors with size-1 dimensions
}

#[test]
fn test_cuda_edge_single_element_tensors() {
    // 1x1 matrix operations
}

#[test]
fn test_cuda_edge_large_mode_count() {
    // Tensors with many modes (>10)
}

#[test]
fn test_cuda_edge_repeated_index_in_input() {
    // einsum("ii,j->j") diagonal extraction
}

#[test]
fn test_cuda_edge_all_contracted() {
    // All indices contracted (full trace)
}
```

### 3.2 Error Handling Tests

```rust
#[test]
#[should_panic]
fn test_cuda_error_unsupported_tropical() {
    // Tropical algebra should error on GPU
}

#[test]
#[should_panic]
fn test_cuda_error_contract_with_argmax() {
    // Argmax tracking not supported
}

#[test]
fn test_cuda_error_invalid_device() {
    // Error handling for invalid device ordinal
}
```

---

## Phase 4: Low Priority - Performance and Advanced (Week 3-4)

### 4.1 Plan Cache Tests

```rust
#[test]
fn test_cuda_cache_hit() {
    // Same contraction reuses cached plan
}

#[test]
fn test_cuda_cache_different_shapes() {
    // Different shapes create new plans
}

#[test]
fn test_cuda_cache_different_dtypes() {
    // Different dtypes create new plans
}
```

### 4.2 Coverage Tests on GPU

Port select tests from `tests/coverage.rs`:

```rust
#[test]
fn test_cuda_tensor_clone() {
    // CudaStorage clone
}

#[test]
fn test_cuda_tensor_zeros() {
    // Tensor::zeros on GPU
}

#[test]
fn test_cuda_tensor_from_storage() {
    // Tensor from CudaStorage
}

#[test]
fn test_cuda_storage_get_set() {
    // CudaStorage get/set operations
}

#[test]
fn test_cuda_einsum_builder_methods() {
    // EinBuilder with GPU backend
}
```

---

## Implementation Notes

### Data Types to Test
- `f32` - single precision (primary)
- `f64` - double precision (primary)
- `CudaComplex<f32>` - complex single
- `CudaComplex<f64>` - complex double

### Numerical Tolerance Standards
```rust
const F32_ABS_TOL: f32 = 1e-5;
const F64_ABS_TOL: f64 = 1e-10;
```

### Known Limitations (Document in Tests)
1. **Tropical algebra NOT supported on GPU** - cuTENSOR doesn't support custom semirings
2. **Argmax tracking NOT supported on GPU** - would require custom CUDA kernels
3. **Strided inputs** - current implementation downloads to CPU, modifies, re-uploads

### Test Helper Functions to Create
```rust
/// Create random tensor with seed for reproducibility
fn random_gpu_tensor<T>(shape: &[usize], seed: u64, cuda: &Cuda) -> Tensor<T, Cuda>;

/// Compare two tensors with tolerance
fn assert_tensors_close<T: Float>(a: &[T], b: &[T], tol: T);

/// Run same operation on CPU and GPU and compare
fn compare_cpu_gpu<F, T>(cpu_op: F, gpu_op: F, shape: &[usize])
where
    F: FnOnce(&Tensor<T, _>) -> Tensor<T, _>;
```

---

## Test Execution Commands

```bash
# Run all CUDA tests
cargo test --features cuda

# Run specific category
cargo test --features cuda test_cuda_unary
cargo test --features cuda test_consistency
cargo test --features cuda test_cuda_backward

# Run with release optimizations
cargo test --features cuda --release

# Run single test with output
cargo test --features cuda test_cuda_einsum_matmul_standard -- --nocapture

# Skip slow tests
cargo test --features cuda -- --skip large --skip bench
```

---

## Progress Tracking

### Phase 1 Checklist
- [ ] Implement 17 unary operation tests
- [ ] Implement 15 CPU-GPU consistency tests
- [ ] Implement 5 additional gradient tests
- [ ] Add helper functions for testing

### Phase 2 Checklist
- [ ] Implement 12 OMEinsum compat tests
- [ ] Implement 15 einsum core pattern tests

### Phase 3 Checklist
- [ ] Implement 6 edge case tests
- [ ] Implement 3 error handling tests

### Phase 4 Checklist
- [ ] Implement 3 plan cache tests
- [ ] Implement 5 coverage tests

---

## Estimated Effort

| Phase | Tests | Lines of Code | Estimated Time |
|-------|-------|---------------|----------------|
| Phase 1 | 37 | ~1,500 | 2-3 days |
| Phase 2 | 27 | ~1,200 | 2 days |
| Phase 3 | 9 | ~300 | 1 day |
| Phase 4 | 8 | ~400 | 1 day |
| **Total** | **81** | **~3,400** | **~7 days** |

---

## Success Criteria

1. All new GPU tests pass on NVIDIA GPU with cuTENSOR 2.0+
2. CPU-GPU consistency tests show < 1e-5 (f32) / < 1e-10 (f64) error
3. No regressions in existing CPU tests
4. GPU test coverage reaches 90%+ of CPU test patterns
5. Clear documentation for unsupported features (tropical, argmax)
