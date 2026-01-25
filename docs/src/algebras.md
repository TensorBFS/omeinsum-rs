# Algebra Types

omeinsum-rs supports multiple algebraic structures (semirings) for tensor operations.

## Semiring Abstraction

A semiring has two operations:
- **Addition** (⊕): Associative, commutative, with identity (zero)
- **Multiplication** (⊗): Associative, with identity (one)

The `Algebra` trait extends `Semiring` with backpropagation support.

## Available Algebras

### Standard Arithmetic

```rust
use omeinsum::algebra::Standard;

// Standard: ⊕ = +, ⊗ = ×
// C[i,j] = Σ_k A[i,k] × B[k,j]
let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
```

### MaxPlus (Tropical)

```rust
use omeinsum::algebra::MaxPlus;

// MaxPlus: ⊕ = max, ⊗ = +
// C[i,j] = max_k (A[i,k] + B[k,j])
// Use case: Longest path, Viterbi algorithm
let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
```

### MinPlus

```rust
use omeinsum::algebra::MinPlus;

// MinPlus: ⊕ = min, ⊗ = +
// C[i,j] = min_k (A[i,k] + B[k,j])
// Use case: Shortest path (Dijkstra, Floyd-Warshall)
let c = einsum::<MinPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
```

### MaxMul

```rust
use omeinsum::algebra::MaxMul;

// MaxMul: ⊕ = max, ⊗ = ×
// C[i,j] = max_k (A[i,k] × B[k,j])
// Use case: Maximum probability paths
let c = einsum::<MaxMul<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
```

## Summary Table

| Algebra | ⊕ | ⊗ | Zero | One | Use Case |
|---------|---|---|------|-----|----------|
| `Standard<T>` | + | × | 0 | 1 | Normal arithmetic |
| `MaxPlus<T>` | max | + | -∞ | 0 | Longest path |
| `MinPlus<T>` | min | + | +∞ | 0 | Shortest path |
| `MaxMul<T>` | max | × | 0 | 1 | Max probability |

## Implementing Custom Algebras

You can implement the `Semiring` and `Algebra` traits for custom algebras:

```rust
use omeinsum::algebra::{Semiring, Algebra, Scalar};

#[derive(Copy, Clone)]
pub struct MyAlgebra<T>(T);

impl<T: Scalar> Semiring for MyAlgebra<T> {
    type Scalar = T;

    fn zero() -> Self { /* ... */ }
    fn one() -> Self { /* ... */ }
    fn add(self, rhs: Self) -> Self { /* ... */ }
    fn mul(self, rhs: Self) -> Self { /* ... */ }
    fn from_scalar(s: T) -> Self { /* ... */ }
    fn to_scalar(self) -> T { /* ... */ }
    fn is_zero(&self) -> bool { /* ... */ }
}
```
