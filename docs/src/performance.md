# Performance Guide

Tips for getting the best performance from omeinsum-rs.

## Contraction Order

The most important optimization is contraction order:

```rust
// Always optimize for networks with 3+ tensors
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();  // or optimize_treesa() for large networks
```

Bad contraction order can be exponentially slower.

## Memory Layout

### Keep Tensors Contiguous

Non-contiguous tensors require copies before GEMM:

```rust
// After permute, tensor may be non-contiguous
let t_permuted = t.permute(&[1, 0]);

// Make contiguous if you'll use it multiple times
let t_contig = t_permuted.contiguous();
```

### Avoid Unnecessary Copies

```rust
// Good: zero-copy view
let view = t.permute(&[1, 0]);

// Avoid: unnecessary explicit copy
let bad = t.permute(&[1, 0]).contiguous();  // Only if needed
```

## Parallelization

Enable the `parallel` feature (default):

```toml
[dependencies]
omeinsum = "0.1"  # parallel enabled by default
```

Disable for single-threaded workloads:

```toml
[dependencies]
omeinsum = { version = "0.1", default-features = false }
```

## Data Types

### Use f32 When Possible

`f32` is typically faster than `f64` due to:
- Smaller memory bandwidth
- Better SIMD utilization

```rust
// Prefer f32
let t = Tensor::<f32, Cpu>::from_data(&data, &shape);

// Use f64 only when precision is critical
let t = Tensor::<f64, Cpu>::from_data(&data, &shape);
```

## Benchmarking

Use release mode for benchmarks:

```bash
cargo run --release --example basic_einsum
```

Profile with:

```bash
cargo build --release
perf record ./target/release/examples/basic_einsum
perf report
```

## Common Pitfalls

### 1. Forgetting to Optimize

```rust
// Bad: no optimization
let ein = Einsum::new(ixs, iy, sizes);
let result = ein.execute::<A, T, B>(&tensors);

// Good: with optimization
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();
let result = ein.execute::<A, T, B>(&tensors);
```

### 2. Redundant Contiguous Calls

```rust
// Bad: unnecessary copy
let c = a.contiguous().gemm::<Standard<f32>>(&b.contiguous());

// Good: gemm handles this internally
let c = a.gemm::<Standard<f32>>(&b);
```

### 3. Debug Mode

Debug builds are ~10-50x slower:

```bash
# Bad: debug mode
cargo run --example benchmark

# Good: release mode
cargo run --release --example benchmark
```

## Future Optimizations

Planned performance improvements:
- CUDA backend for GPU acceleration
- Optimized tropical-gemm kernel integration
- Batched GEMM support
- Cache-aware blocking
