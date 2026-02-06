# Benchmark binary einsum with high-dimensional tensors (rank ~25, dim size 2)
#
# Run with: julia --project=~/.julia/dev/OMEinsum examples/bench_binary.jl

using OMEinsum
using BenchmarkTools

println("=== Binary Einsum Benchmark (dim size = 2) ===\n")

# Test cases: (name, rank_a, rank_b, num_contracted, num_batch)
test_cases = [
    # Simple cases
    ("matmul 10x10", 10, 10, 5, 0),
    ("batched matmul 8x8 batch=4", 8, 8, 4, 4),

    # High-dimensional cases (like tensor network contractions)
    ("high-D 12x12 contract=6", 12, 12, 6, 0),
    ("high-D 15x15 contract=7", 15, 15, 7, 0),
    ("high-D 18x18 contract=8", 18, 18, 8, 0),
    ("high-D 20x20 contract=9", 20, 20, 9, 0),

    # With batch dimensions
    ("high-D 12x12 contract=4 batch=4", 12, 12, 4, 4),
    ("high-D 15x15 contract=5 batch=5", 15, 15, 5, 5),
]

function bench_binary_einsum(name, rank_a, rank_b, num_contracted, num_batch)
    # Build index labels
    # A has: [left_a..., contracted..., batch...]
    # B has: [contracted..., right_b..., batch...]
    # C has: [left_a..., right_b..., batch...]

    num_left_a = rank_a - num_contracted - num_batch
    num_right_b = rank_b - num_contracted - num_batch

    # Assign index labels (use integers)
    next_idx = 1

    # Left indices (only in A)
    left_indices = collect(next_idx:next_idx + num_left_a - 1)
    next_idx += num_left_a

    # Contracted indices (in A and B, not in C)
    contracted_indices = collect(next_idx:next_idx + num_contracted - 1)
    next_idx += num_contracted

    # Right indices (only in B)
    right_indices = collect(next_idx:next_idx + num_right_b - 1)
    next_idx += num_right_b

    # Batch indices (in A, B, and C)
    batch_indices = collect(next_idx:next_idx + num_batch - 1)

    # Build index arrays
    ixs_a = tuple(vcat(left_indices, contracted_indices, batch_indices)...)
    ixs_b = tuple(vcat(contracted_indices, right_indices, batch_indices)...)
    ixs_c = tuple(vcat(left_indices, right_indices, batch_indices)...)

    # Create tensors with shape [2, 2, 2, ...]
    shape_a = ntuple(_ -> 2, rank_a)
    shape_b = ntuple(_ -> 2, rank_b)

    numel_a = 2^rank_a
    numel_b = 2^rank_b
    numel_c = 2^(num_left_a + num_right_b + num_batch)

    A = rand(Float32, shape_a...)
    B = rand(Float32, shape_b...)

    # Build einsum code
    code = EinCode((ixs_a, ixs_b), ixs_c)

    # Warm up
    for _ in 1:3
        einsum(code, (A, B))
    end

    # Benchmark
    t = @belapsed einsum($code, ($A, $B))
    avg_ms = t * 1000

    # Compute theoretical sizes
    left_size = 2^num_left_a
    right_size = 2^num_right_b
    contract_size = 2^num_contracted
    batch_size = 2^num_batch

    # Memory bandwidth (read A + read B + write C)
    bytes_moved = (numel_a + numel_b + numel_c) * 4
    bandwidth_gbs = bytes_moved / t / 1e9

    println(name)
    println("  A: rank=$rank_a, B: rank=$rank_b")
    println("  left=$num_left_a, contract=$num_contracted, right=$num_right_b, batch=$num_batch")
    println("  GEMM: [$(left_size)x$(contract_size)] @ [$(contract_size)x$(right_size)] x $batch_size batches")
    println("  Time: $(round(avg_ms, digits=3)) ms, Bandwidth: $(round(bandwidth_gbs, digits=2)) GB/s")
    println()
end

for (name, rank_a, rank_b, num_contracted, num_batch) in test_cases
    bench_binary_einsum(name, rank_a, rank_b, num_contracted, num_batch)
end
