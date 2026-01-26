# Showcase Examples

This chapter demonstrates practical applications of einsum with gradients across three different algebras: real numbers, complex numbers, and tropical numbers.

Each example shows a real-world use case where **differentiation through tensor networks** provides meaningful results.

| Example | Algebra | Application | Gradient Meaning |
|---------|---------|-------------|------------------|
| [Bayesian Network](#bayesian-network-marginals) | `Standard<f64>` | Probabilistic inference | Marginal probability |
| [Tensor Train](#tensor-train-quantum-states) | `Standard<Complex64>` | Quantum simulation | Energy optimization direction |
| [Independent Set](#maximum-weight-independent-set) | `MaxPlus<f64>` | Combinatorial optimization | Optimal vertex selection |

---

## Bayesian Network Marginals

**Key insight:** Differentiation = Marginalization

### Problem

Given a chain-structured Bayesian network with 3 binary variables X₀ - X₁ - X₂, compute:
1. The partition function Z (sum over all configurations)
2. The marginal probability P(X₁ = 1)

### Mathematical Setup

**Vertex potentials** (unnormalized probabilities):
```
φ₀ = [1, 2]   → P(X₀=1) ∝ 2
φ₁ = [1, 3]   → P(X₁=1) ∝ 3
φ₂ = [1, 1]   → uniform
```

**Edge potentials** (encourage agreement):
```
ψ = [[2, 1],
     [1, 2]]
```

**Partition function** as einsum:
```
Z = Σ_{x₀,x₁,x₂} φ₀(x₀) × ψ₀₁(x₀,x₁) × φ₁(x₁) × ψ₁₂(x₁,x₂) × φ₂(x₂)
  = einsum("i,ij,j,jk,k->", φ₀, ψ₀₁, φ₁, ψ₁₂, φ₂)
```

### The Gradient-Marginal Connection

The beautiful insight from probabilistic graphical models:

```
∂Z/∂θᵥ = Σ_{configurations where xᵥ=1} (product of all other factors)
       = Z × P(xᵥ = 1)
```

Therefore:
```
P(xᵥ = 1) = (1/Z) × ∂Z/∂θᵥ = ∂log(Z)/∂θᵥ
```

**Differentiation through the tensor network gives marginal probabilities!**

### Manual Verification

All 8 configurations:

| X₀ | X₁ | X₂ | φ₀ | ψ₀₁ | φ₁ | ψ₁₂ | φ₂ | Product |
|----|----|----|----|----|----|----|----|----|
| 0 | 0 | 0 | 1 | 2 | 1 | 2 | 1 | 4 |
| 0 | 0 | 1 | 1 | 2 | 1 | 1 | 1 | 2 |
| 0 | 1 | 0 | 1 | 1 | 3 | 1 | 1 | 3 |
| 0 | 1 | 1 | 1 | 1 | 3 | 2 | 1 | 6 |
| 1 | 0 | 0 | 2 | 1 | 1 | 2 | 1 | 4 |
| 1 | 0 | 1 | 2 | 1 | 1 | 1 | 1 | 2 |
| 1 | 1 | 0 | 2 | 2 | 3 | 1 | 1 | 12 |
| 1 | 1 | 1 | 2 | 2 | 3 | 2 | 1 | 24 |

**Results:**
- Z = 57
- P(X₁=1) = (3+6+12+24)/57 = 45/57 ≈ 0.789

### Code

```rust
use omeinsum::{einsum, einsum_with_grad, Standard, Tensor, Cpu};

// Vertex potentials
let phi0 = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
let phi1 = Tensor::<f64, Cpu>::from_data(&[1.0, 3.0], &[2]);
let phi2 = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);

// Edge potentials (column-major)
let psi = Tensor::<f64, Cpu>::from_data(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);

// Contract step by step
let t1 = einsum::<Standard<f64>, _, _>(&[&phi0, &psi], &[&[0], &[0, 1]], &[1]);
// ... continue contracting to compute Z

// Use einsum_with_grad to get gradients
let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(
    &[&phi0, &psi], &[&[0], &[0, 1]], &[1]
);
let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&phi0, &psi]);
```

---

## Tensor Train (Quantum States)

**Key insight:** Gradients enable variational optimization of quantum states

### Problem

Represent a quantum state using a Matrix Product State (MPS) and compute contractions with complex numbers. Gradients ∂E/∂A give the optimization direction for finding ground states.

### Mathematical Setup

An MPS represents a quantum state as:
```
|ψ⟩ = Σ_{s₁,s₂,...} A¹[s₁] · A²[s₂] · ... |s₁s₂...⟩
```

Where each Aⁱ[sᵢ] is a complex matrix.

### Example: Two-Site Contraction

```
A1 = [[1+i,  0  ],      A2 = [[2,  i ],
      [0,   1-i]]            [-i, 3 ]]
```

Contraction: `result[s1,s2] = Σ_b A1[s1,b] × A2[b,s2]`

**Manual calculation:**
```
result[0,0] = (1+i)×2 + 0×(-i) = 2+2i
result[0,1] = (1+i)×i + 0×3 = -1+i
result[1,0] = 0×2 + (1-i)×(-i) = -1-i
result[1,1] = 0×i + (1-i)×3 = 3-3i
```

**Norm:** ⟨ψ|ψ⟩ = |2+2i|² + |-1+i|² + |-1-i|² + |3-3i|² = 8+2+2+18 = 30

### Code

```rust
use num_complex::Complex64 as C64;
use omeinsum::{einsum, einsum_with_grad, Standard, Tensor, Cpu};

let a1 = Tensor::<C64, Cpu>::from_data(&[
    C64::new(1.0, 1.0),   // 1+i
    C64::new(0.0, 0.0),   // 0
    C64::new(0.0, 0.0),   // 0
    C64::new(1.0, -1.0),  // 1-i
], &[2, 2]);

let a2 = Tensor::<C64, Cpu>::from_data(&[
    C64::new(2.0, 0.0),   // 2
    C64::new(0.0, -1.0),  // -i
    C64::new(0.0, 1.0),   // i
    C64::new(3.0, 0.0),   // 3
], &[2, 2]);

// Contract: result[s1,s2] = Σ_b A1[s1,b] × A2[b,s2]
let result = einsum::<Standard<C64>, _, _>(
    &[&a1, &a2],
    &[&[0, 1], &[1, 2]],  // contract over index 1
    &[0, 2]
);

// Compute gradients for optimization
let (result, grad_fn) = einsum_with_grad::<Standard<C64>, _, _>(
    &[&a1, &a2], &[&[0, 1], &[1, 2]], &[0, 2]
);
```

### Application: Variational Ground State

For a Heisenberg spin chain, the energy expectation ⟨ψ|H|ψ⟩ can be computed via tensor network contraction. The gradient ∂E/∂Aⁱ tells us how to update each tensor to lower the energy, converging to the ground state.

---

## Maximum Weight Independent Set

**Key insight:** Tropical gradients give optimal vertex selection

### Problem

Find the maximum weight independent set on a pentagon graph. An independent set contains no adjacent vertices.

### Graph

```
      0 (w=3)
     / \
    4   1
   (2) (5)
    |   |
    3---2
   (4) (1)
```

Edges: (0,1), (1,2), (2,3), (3,4), (4,0)

### Tropical Tensor Network

**Vertex tensor** for vertex v with weight wᵥ:
```
W[s] = [0, wᵥ]  where s ∈ {0, 1}
```
- s=0: vertex not selected, contributes 0 (tropical multiplicative identity)
- s=1: vertex selected, contributes wᵥ

**Edge tensor** enforcing independence constraint:
```
B[sᵤ, sᵥ] = [[0,   0 ],
             [0,  -∞ ]]
```
- B[1,1] = -∞ forbids selecting both endpoints (tropical zero)

**Tropical contraction** (MaxPlus: ⊕=max, ⊗=+):
```
result = max over all valid configurations of Σ(selected weights)
```

### Gradient = Selection Mask

From tropical autodiff theory:
```
∂(max_weight)/∂(wᵥ) = 1 if vertex v is in optimal set
                    = 0 otherwise
```

**The tropical gradient directly reveals the optimal selection!**

### Manual Verification

All independent sets of the pentagon:

| Set | Weight |
|-----|--------|
| {0} | 3 |
| {1} | 5 |
| {2} | 1 |
| {3} | 4 |
| {4} | 2 |
| {0,2} | 4 |
| {0,3} | 7 |
| **{1,3}** | **9** ← maximum |
| {1,4} | 7 |
| {2,4} | 3 |

**Optimal:** {1, 3} with weight 9

### Code

```rust
use omeinsum::{einsum, MaxPlus, Tensor, Cpu};

// Vertex tensors: W[s] = [0, weight]
let w0 = Tensor::<f64, Cpu>::from_data(&[0.0, 3.0], &[2]);
let w1 = Tensor::<f64, Cpu>::from_data(&[0.0, 5.0], &[2]);

// Edge constraint: B[1,1] = -∞ forbids both selected
let neg_inf = f64::NEG_INFINITY;
let edge = Tensor::<f64, Cpu>::from_data(&[0.0, 0.0, 0.0, neg_inf], &[2, 2]);

// Contract two vertices with edge constraint
// max_{s0,s1} (W0[s0] + B[s0,s1] + W1[s1])
let t0e = einsum::<MaxPlus<f64>, _, _>(&[&w0, &edge], &[&[0], &[0, 1]], &[1]);
let result = einsum::<MaxPlus<f64>, _, _>(&[&t0e, &w1], &[&[0], &[0]], &[]);

// Result: 5.0 (select vertex 1 only, since selecting both gives -∞)
```

### References

- Liu & Wang, "[Tropical Tensor Network for Ground States of Spin Glasses](https://arxiv.org/abs/2008.06888)", PRL 2021
- Liu et al., "[Computing Solution Space Properties via Generic Tensor Networks](https://epubs.siam.org/doi/10.1137/22M1501787)", SIAM 2023

---

## Summary

| Algebra | Operation | Gradient Meaning |
|---------|-----------|------------------|
| Standard (real) | Σ (sum), × (multiply) | Sensitivity / marginal probability |
| Standard (complex) | Σ, × with complex arithmetic | Optimization direction |
| MaxPlus (tropical) | max, + | Binary selection mask |

These examples demonstrate that **einsum with automatic differentiation** is a powerful tool for:
- Probabilistic inference (belief propagation)
- Quantum simulation (variational methods)
- Combinatorial optimization (finding optimal configurations)
