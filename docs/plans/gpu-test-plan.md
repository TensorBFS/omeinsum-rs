# GPU (CUDA) Test Plan for omeinsum-rs

## Overview

This document provides a detailed test plan for GPU testing of the omeinsum-rs CUDA backend. The tests ensure correctness, performance, and compatibility of GPU tensor operations using cuTENSOR.

---

## 1. Prerequisites and Environment Setup

### 1.1 Hardware Requirements
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, or newer)
- Minimum 4GB GPU memory for standard tests
- Minimum 8GB GPU memory for large tensor tests

### 1.2 Software Requirements
- CUDA Toolkit 11.0 or later (12.x recommended)
- cuTENSOR 2.0 or later (**Version 1.x is NOT compatible**)
- NVIDIA driver compatible with CUDA version

### 1.3 Environment Configuration
```bash
# Set cuTENSOR library path
export CUTENSOR_PATH=/path/to/cutensor/lib
export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH

# Alternative: Use conda
conda install -c nvidia cutensor-cu12

# Verify installation
ls $CUTENSOR_PATH/libcutensor.so*
```

### 1.4 Build Commands
```bash
# Build with CUDA feature
cargo build --features cuda

# Run all CUDA tests
cargo test --features cuda

# Run specific CUDA test module
cargo test --features cuda cuda::

# Run with release optimizations (for performance tests)
cargo test --features cuda --release
```

---

## 2. Test Categories

### 2.1 Device Initialization Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-INIT-001 | `test_cuda_init` | Verify CUDA device initialization on default device | HIGH |
| GPU-INIT-002 | `test_cuda_multi_device` | Test initialization on multiple GPU devices | MEDIUM |
| GPU-INIT-003 | `test_cuda_init_invalid_device` | Error handling for invalid device ordinal | MEDIUM |
| GPU-INIT-004 | `test_cuda_handle_creation` | Verify cuTENSOR handle creation | HIGH |
| GPU-INIT-005 | `test_cuda_clone_backend` | Test cloning Cuda backend shares device | MEDIUM |

### 2.2 Memory Transfer Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-MEM-001 | `test_storage_roundtrip_f32` | Host-to-device-to-host copy for f32 | HIGH |
| GPU-MEM-002 | `test_storage_roundtrip_f64` | Host-to-device-to-host copy for f64 | HIGH |
| GPU-MEM-003 | `test_storage_roundtrip_complex32` | Roundtrip for CudaComplex<f32> | HIGH |
| GPU-MEM-004 | `test_storage_roundtrip_complex64` | Roundtrip for CudaComplex<f64> | HIGH |
| GPU-MEM-005 | `test_storage_len` | Verify length and is_empty methods | MEDIUM |
| GPU-MEM-006 | `test_large_tensor_transfer` | Transfer tensors > 1GB | LOW |
| GPU-MEM-007 | `test_zero_length_storage` | Handle empty tensor allocation | MEDIUM |

### 2.3 Basic Contraction Tests (Low-Level cuTENSOR API)

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-CONT-001 | `test_matmul_f32` | 2x3 @ 3x2 matrix multiplication (f32) | HIGH |
| GPU-CONT-002 | `test_matmul_f64` | 2x2 @ 2x2 matrix multiplication (f64) | HIGH |
| GPU-CONT-003 | `test_inner_product` | Vector dot product (scalar output) | HIGH |
| GPU-CONT-004 | `test_outer_product` | Vector outer product | HIGH |
| GPU-CONT-005 | `test_batch_matmul` | Batched matrix multiplication [b,i,j] @ [b,j,k] | HIGH |
| GPU-CONT-006 | `test_tensor3_contraction_f64` | 3D tensor contraction | HIGH |
| GPU-CONT-007 | `test_trace_f64` | Diagonal sum via contraction | MEDIUM |

### 2.4 High-Level Einsum API Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-EIN-001 | `test_cuda_einsum_matmul_standard` | einsum("ij,jk->ik") with Standard algebra | HIGH |
| GPU-EIN-002 | `test_cuda_einsum_matmul_identity` | Multiply by identity matrix | HIGH |
| GPU-EIN-003 | `test_cuda_einsum_matmul_rectangular` | Non-square matrix multiplication | HIGH |
| GPU-EIN-004 | `test_cuda_einsum_tensor_contraction_3d` | 3D @ 2D tensor contraction | HIGH |
| GPU-EIN-005 | `test_cuda_einsum_batch_matmul` | Batched matmul via einsum | HIGH |
| GPU-EIN-006 | `test_cuda_einsum_contract_two_axes` | Contract over multiple axes | MEDIUM |
| GPU-EIN-007 | `test_cuda_einsum_matmul_f64` | f64 precision einsum | HIGH |
| GPU-EIN-008 | `test_cuda_einsum_matmul_chain` | A @ B @ C chain contraction | MEDIUM |
| GPU-EIN-009 | `test_cuda_einsum_matmul_four_tensors` | A @ B @ C @ D chain | MEDIUM |
| GPU-EIN-010 | `test_cuda_einsum_inner_product` | Scalar output einsum | HIGH |
| GPU-EIN-011 | `test_cuda_einsum_outer_product` | Outer product via einsum | HIGH |
| GPU-EIN-012 | `test_cuda_einsum_transpose` | Transpose via einsum("ij->ji") | HIGH |
| GPU-EIN-013 | `test_cuda_einsum_trace` | Trace via einsum("ii->") | HIGH |

### 2.5 Complex Number Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-CPX-001 | `test_matmul_complex64` | Complex matrix multiplication | HIGH |
| GPU-CPX-002 | `test_inner_product_complex64` | Complex dot product | HIGH |
| GPU-CPX-003 | `test_outer_product_complex64` | Complex outer product | MEDIUM |
| GPU-CPX-004 | `test_cuda_complex32_matmul` | CudaComplex<f32> operations | HIGH |
| GPU-CPX-005 | `test_complex_hermitian` | Hermitian matrix operations | LOW |

### 2.6 Manual Gradient/Backward Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-GRAD-001 | `test_cuda_manual_backward_matmul_f64` | grad_A = grad_C @ B^T, grad_B = A^T @ grad_C | HIGH |
| GPU-GRAD-002 | `test_cuda_manual_backward_rectangular_f64` | Gradient for non-square matrices | HIGH |
| GPU-GRAD-003 | `test_cuda_manual_backward_outer_product_f64` | Gradient for outer product | MEDIUM |
| GPU-GRAD-004 | `test_cuda_manual_backward_matmul_complex64` | Complex number gradients | MEDIUM |
| GPU-GRAD-005 | `test_cuda_backward_batch_matmul` | Gradient for batched matmul | MEDIUM |
| GPU-GRAD-006 | `test_cuda_backward_chain` | Gradient for A @ B @ C chain | MEDIUM |

### 2.7 CPU-GPU Consistency Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-CONS-001 | `test_cpu_gpu_matmul_consistency` | Compare CPU and GPU matmul results | HIGH |
| GPU-CONS-002 | `test_cpu_gpu_batch_matmul_consistency` | Compare batched operations | HIGH |
| GPU-CONS-003 | `test_cpu_gpu_chain_consistency` | Compare multi-tensor chains | MEDIUM |
| GPU-CONS-004 | `test_cpu_gpu_einsum_consistency` | Compare high-level einsum results | HIGH |
| GPU-CONS-005 | `test_cpu_gpu_gradient_consistency` | Compare gradient computations | HIGH |

### 2.8 Edge Cases and Error Handling

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-EDGE-001 | `test_scalar_contraction` | 0-dimensional output tensor | HIGH |
| GPU-EDGE-002 | `test_empty_contraction_axis` | No contracted indices (outer product) | MEDIUM |
| GPU-EDGE-003 | `test_all_contracted` | All indices contracted (full trace) | MEDIUM |
| GPU-EDGE-004 | `test_size_one_dimension` | Tensors with size-1 dimensions | MEDIUM |
| GPU-EDGE-005 | `test_large_mode_count` | Tensors with many modes (>10) | LOW |
| GPU-EDGE-006 | `test_repeated_index_in_input` | einsum("ii,j->j") diagonal extraction | HIGH |
| GPU-EDGE-007 | `test_unsupported_tropical` | Verify tropical algebra errors on GPU | HIGH |
| GPU-EDGE-008 | `test_contract_with_argmax_error` | Verify argmax tracking not supported | HIGH |

### 2.9 Plan Cache Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-CACHE-001 | `test_plan_cache_hit` | Same contraction reuses cached plan | MEDIUM |
| GPU-CACHE-002 | `test_plan_cache_eviction` | LRU eviction when cache is full | LOW |
| GPU-CACHE-003 | `test_plan_cache_different_shapes` | Different shapes create new plans | MEDIUM |
| GPU-CACHE-004 | `test_plan_cache_different_dtypes` | Different dtypes create new plans | MEDIUM |

### 2.10 Performance Tests

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| GPU-PERF-001 | `bench_matmul_1024x1024` | 1K x 1K matmul performance | LOW |
| GPU-PERF-002 | `bench_matmul_4096x4096` | 4K x 4K matmul performance | LOW |
| GPU-PERF-003 | `bench_batch_matmul_large` | Large batch (1024) matmul | LOW |
| GPU-PERF-004 | `bench_tensor_contraction_8d` | High-dimensional contraction | LOW |
| GPU-PERF-005 | `bench_gpu_vs_cpu` | Compare GPU/CPU speedup | LOW |

---

## 3. Test Implementation Details

### 3.1 Numerical Tolerance Standards

```rust
// Single precision (f32)
const F32_ABS_TOL: f32 = 1e-5;
const F32_REL_TOL: f32 = 1e-4;

// Double precision (f64)
const F64_ABS_TOL: f64 = 1e-10;
const F64_REL_TOL: f64 = 1e-9;

// Complex numbers: apply same tolerances to real and imaginary parts
```

### 3.2 Test Helper Functions

```rust
/// Compare GPU and CPU results with tolerance
fn assert_gpu_cpu_equal<T: Float>(gpu: &[T], cpu: &[T], tol: T) {
    assert_eq!(gpu.len(), cpu.len());
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        assert!((g - c).abs() < tol, "GPU: {}, CPU: {}", g, c);
    }
}

/// Helper to create random test data
fn random_tensor<T: Float>(shape: &[usize], seed: u64) -> Vec<T>;

/// Helper to run same operation on CPU and GPU
fn run_cpu_gpu_comparison<F, R>(cpu_op: F, gpu_op: F) -> (R, R);
```

### 3.3 Test Data Patterns

1. **Identity matrices**: Verify no-op behavior
2. **Sequential values**: `[1, 2, 3, ...]` for predictable results
3. **Random data**: With fixed seeds for reproducibility
4. **Edge values**: `0`, `1`, `-1`, large values, small values
5. **Special patterns**: Diagonal, symmetric, skew-symmetric

---

## 4. Missing Tests to Implement

Based on Julia OMEinsum's `test/cueinsum.jl` and current gaps:

### 4.1 High Priority Missing Tests

| Test | Julia Reference | Description |
|------|-----------------|-------------|
| `test_cuda_einsum_complex_standard` | cueinsum.jl:56-89 | Complex einsum via high-level API |
| `test_cuda_einsum_gradient` | cueinsum.jl:200-250 | Automatic differentiation on GPU |
| `test_cuda_einsum_nested` | cueinsum.jl:150-180 | Nested einsum expressions |
| `test_cuda_einsum_sum_reduction` | cueinsum.jl:100-120 | Sum over axes via einsum |
| `test_cuda_permute` | cueinsum.jl:90-100 | Axis permutation on GPU |

### 4.2 Medium Priority Missing Tests

| Test | Description |
|------|-------------|
| `test_cuda_strided_input` | Non-contiguous tensor input |
| `test_cuda_einsum_optimizer` | Verify optimizer works with GPU backend |
| `test_cuda_multi_stream` | Concurrent operations on multiple streams |
| `test_cuda_memory_pool` | Memory reuse behavior |

### 4.3 Low Priority/Future Tests

| Test | Description |
|------|-------------|
| `test_cuda_half_precision` | f16 support (when cuTENSOR supports) |
| `test_cuda_tensor_cores` | Verify Tensor Core utilization |
| `test_cuda_multi_gpu` | Multi-GPU distribution |

---

## 5. Test Execution Order

### Phase 1: Environment Validation
1. GPU-INIT-001 to GPU-INIT-005
2. GPU-MEM-001 to GPU-MEM-005

### Phase 2: Core Functionality
1. GPU-CONT-001 to GPU-CONT-007
2. GPU-EIN-001 to GPU-EIN-013

### Phase 3: Advanced Features
1. GPU-CPX-001 to GPU-CPX-005
2. GPU-GRAD-001 to GPU-GRAD-006

### Phase 4: Validation & Edge Cases
1. GPU-CONS-001 to GPU-CONS-005
2. GPU-EDGE-001 to GPU-EDGE-008

### Phase 5: Performance (Optional)
1. GPU-PERF-001 to GPU-PERF-005

---

## 6. CI/CD Integration

### 6.1 GitHub Actions Configuration

```yaml
# .github/workflows/cuda-tests.yml
name: CUDA Tests

on: [push, pull_request]

jobs:
  cuda-tests:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v4
      - name: Setup CUDA
        run: |
          export CUDA_PATH=/usr/local/cuda
          export CUTENSOR_PATH=/opt/cutensor/lib
          export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH
      - name: Run CUDA tests
        run: cargo test --features cuda --release
```

### 6.2 Test Filtering

```bash
# Run only high-priority tests
cargo test --features cuda -- --test-threads=1 test_cuda_

# Skip slow performance tests
cargo test --features cuda -- --skip bench_

# Run tests matching pattern
cargo test --features cuda -- matmul
```

---

## 7. Debugging Guide

### 7.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| cuTENSOR 1.x installed | Linker errors: `undefined symbol: cutensorContract` | Install cuTENSOR 2.0+ |
| Missing library | `libcutensor.so not found` | Set `CUTENSOR_PATH` correctly |
| CUDA not available | `Failed to initialize CUDA` | Check GPU and drivers |
| Numerical mismatch | Results differ from CPU | Check for precision loss, use f64 |

### 7.2 Debug Commands

```bash
# Check CUDA installation
nvidia-smi

# Check cuTENSOR version
ls -la $CUTENSOR_PATH/libcutensor*

# Run single test with debug output
RUST_BACKTRACE=1 cargo test --features cuda test_matmul_f32 -- --nocapture

# Run with address sanitizer (if supported)
RUSTFLAGS="-Z sanitizer=address" cargo test --features cuda
```

---

## 8. Test Coverage Summary

| Category | Total Tests | Implemented | Missing |
|----------|-------------|-------------|---------|
| Initialization | 5 | 2 | 3 |
| Memory Transfer | 7 | 4 | 3 |
| Basic Contraction | 7 | 7 | 0 |
| High-Level Einsum | 13 | 13 | 0 |
| Complex Numbers | 5 | 5 | 0 |
| Gradients | 6 | 4 | 2 |
| CPU-GPU Consistency | 5 | 0 | 5 |
| Edge Cases | 8 | 2 | 6 |
| Plan Cache | 4 | 0 | 4 |
| Performance | 5 | 0 | 5 |
| **Total** | **65** | **37** | **28** |

**Current Coverage: 57%**

---

## 9. Next Steps

1. **Immediate (High Priority)**:
   - Implement CPU-GPU consistency tests (GPU-CONS-001 to GPU-CONS-005)
   - Add edge case tests for scalar output and diagonal extraction
   - Verify error handling for unsupported operations

2. **Short-term (Medium Priority)**:
   - Implement plan cache tests
   - Add remaining gradient tests
   - Create strided input tests

3. **Long-term (Low Priority)**:
   - Performance benchmarks
   - Multi-GPU support tests
   - Half-precision tests (when available)

---

## Appendix A: Running Tests Locally

```bash
# Full GPU test suite
cd /home/jinguoliu/rcode/omeinsum-rs
export CUTENSOR_PATH=/path/to/cutensor/lib
export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH
cargo test --features cuda --release

# With verbose output
cargo test --features cuda -- --nocapture

# Generate test coverage report
cargo tarpaulin --features cuda --out Html
```

## Appendix B: Test Data for Verification

### Matrix Multiplication Reference

```
A = [[1, 2], [3, 4]]
B = [[1, 2], [3, 4]]
C = A @ B = [[7, 10], [15, 22]]

A = [[1, 2, 3], [4, 5, 6]]  (2x3)
B = [[1, 2], [3, 4], [5, 6]]  (3x2)
C = A @ B = [[22, 28], [49, 64]]  (2x2)
```

### Gradient Reference

```
Forward: C = A @ B
Backward: grad_A = grad_C @ B^T
          grad_B = A^T @ grad_C

Example with grad_C = ones:
A = [[1, 2], [3, 4]], B = [[1, 2], [3, 4]]
grad_A = [[1,1],[1,1]] @ [[1,3],[2,4]] = [[3, 7], [3, 7]]
grad_B = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4, 4], [6, 6]]
```
