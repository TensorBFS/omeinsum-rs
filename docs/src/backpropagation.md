# Backpropagation

omeinsum-rs supports gradient computation for both standard and tropical algebras.

## Standard Backpropagation

For standard arithmetic, gradients follow the chain rule:

```
C = A × B
∂L/∂A = ∂L/∂C × B^T
∂L/∂B = A^T × ∂L/∂C
```

## Tropical Backpropagation

Tropical algebras use argmax tracking for gradient routing.

### The Challenge

In tropical algebra:
```
C[i,j] = max_k (A[i,k] + B[k,j])
```

The gradient only flows through the winning path (the k that achieved the max).

### Argmax Tracking

During forward pass, we track which index "won":

```rust
let (c, argmax) = a.gemm_with_argmax::<MaxPlus<f32>>(&b);
// argmax[i,j] = the k that maximized A[i,k] + B[k,j]
```

### Backward Pass

Gradients are routed using the argmax:

```rust
// For each output element [i,j]:
// k* = argmax[i,j]
// ∂L/∂A[i,k*] += ∂L/∂C[i,j]
// ∂L/∂B[k*,j] += ∂L/∂C[i,j]
```

## API Usage

### With Argmax Tracking

```rust
use omeinsum::algebra::MaxPlus;

// GEMM with argmax
let (c, argmax) = a.gemm_with_argmax::<MaxPlus<f32>>(&b);

// Contract with argmax
let (c, argmax) = a.contract_binary_with_argmax::<MaxPlus<f32>>(
    &b, &[0, 1], &[1, 2], &[0, 2]
);
```

### Einsum with Gradients

```rust
use omeinsum::einsum_with_grad;

let (result, gradient) = einsum_with_grad::<MaxPlus<f32>, _, _>(
    &[&a, &b],
    &[&[0, 1], &[1, 2]],
    &[0, 2],
);

// Use gradient.backward() for gradient computation
// (Implementation in progress)
```

## Implementation Status

| Feature | Status |
|---------|--------|
| Forward pass | Complete |
| Argmax tracking | Complete |
| GEMM backward | Implemented |
| Full einsum backward | In progress |

## Tie-Breaking

When multiple indices achieve the same maximum, the implementation uses a deterministic tie-breaking rule (first winning index). This ensures reproducible gradients.

## References

- Zhang et al., "Tropical Geometry of Deep Neural Networks" (2018)
- tropical-gemm gradient implementation
