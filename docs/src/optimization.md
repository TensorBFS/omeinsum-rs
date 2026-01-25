# Contraction Optimization

Finding the optimal contraction order is critical for tensor network performance.

## The Problem

Consider contracting tensors A, B, C, D with different contraction orders:

```
((A × B) × C) × D  vs  (A × B) × (C × D)  vs  A × ((B × C) × D)
```

Different orders have vastly different computational costs. For large networks, the difference can be exponential.

## Optimization Algorithms

omeinsum-rs uses [omeco](https://github.com/GiggleLiu/omeco) for contraction order optimization.

### Greedy Method

The greedy algorithm iteratively contracts the pair with minimum cost:

```rust
use omeinsum::Einsum;

let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();
```

- **Complexity**: O(n²) where n is number of tensors
- **Quality**: Good for most practical cases
- **Speed**: Fast

### Tree Simulated Annealing (TreeSA)

TreeSA uses simulated annealing to search for better contraction trees:

```rust
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_treesa();
```

- **Complexity**: O(iterations × n)
- **Quality**: Often finds optimal or near-optimal solutions
- **Speed**: Slower, but worthwhile for large networks

## When to Optimize

| Network Size | Recommendation |
|--------------|----------------|
| 2-3 tensors | No optimization needed |
| 4-10 tensors | Greedy is usually sufficient |
| 10+ tensors | Consider TreeSA |
| Performance-critical | Always optimize, benchmark both |

## Inspecting Results

```rust
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();

// Check if optimized
if ein.is_optimized() {
    // Get the contraction tree
    if let Some(tree) = ein.contraction_tree() {
        println!("Optimized tree: {:?}", tree);
    }
}
```

## Cost Model

The optimization minimizes total FLOP count, considering:

- Tensor dimensions from size dictionary
- Intermediate tensor sizes
- Number of operations per contraction

## No Optimization

For simple cases, you can skip optimization:

```rust
// Without optimization: contracts left-to-right
let ein = Einsum::new(ixs, iy, sizes);
let result = ein.execute::<Standard<f32>, f32, Cpu>(&tensors);
```

This uses simple pairwise contraction from left to right, which may be suboptimal for complex networks.

## Further Reading

- [omeco documentation](https://github.com/GiggleLiu/omeco)
- Gray & Kourtis, "Hyper-optimized tensor network contraction" (2021)
