# Showcase Examples Design

**Goal:** Create three showcase test cases demonstrating gradients across different algebras (real, complex, tropical).

**Architecture:** Each example is a standalone test in `tests/showcase.rs` with detailed documentation explaining the math/physics, small enough to verify by hand.

**Tech Stack:** omeinsum with Standard<f64>, Standard<Complex64>, and MaxPlus<f64> algebras.

---

## Overview

| Example | Algebra | Problem | Gradient Meaning |
|---------|---------|---------|------------------|
| Bayesian Network | Standard<f64> | Compute partition function | Marginal probability |
| Tensor Train GS | Standard<Complex64> | Find ground state of 1D Hamiltonian | ∂(energy)/∂(tensors) |
| Max-Weight IS | MaxPlus<f64> | Find maximum weight independent set | Binary selection mask |

---

## Example 1: Bayesian Network Marginals (Real Numbers)

### Problem Statement

Given a simple Bayesian network (chain-structured factor graph), compute the partition function Z. The gradient ∂Z/∂θᵥ with respect to vertex potential gives the marginal probability P(v=1).

This is a beautiful connection: **differentiation = marginalization**.

### Mathematical Formulation

For a chain of 3 binary variables X₀ - X₁ - X₂ with:
- Vertex potentials: φᵥ(xᵥ) = [1, θᵥ] (unnormalized probability of xᵥ=0 vs xᵥ=1)
- Edge potentials: ψₑ(xᵤ, xᵥ) = compatibility matrix

The partition function:
```
Z = Σ_{x₀,x₁,x₂} φ₀(x₀) × ψ₀₁(x₀,x₁) × φ₁(x₁) × ψ₁₂(x₁,x₂) × φ₂(x₂)
```

In einsum notation:
```
Z = einsum("i,ij,j,jk,k->", φ₀, ψ₀₁, φ₁, ψ₁₂, φ₂)
```

### Gradient = Marginal Probability

The key insight:
```
∂Z/∂θᵥ = Σ_{configurations where xᵥ=1} (product of all other factors)
       = Z × P(xᵥ = 1)
```

Therefore:
```
P(xᵥ = 1) = (1/Z) × ∂Z/∂θᵥ = ∂log(Z)/∂θᵥ
```

**Differentiation through the tensor network gives marginal probabilities!**

### Test Case

```rust
// Chain: X₀ -- X₁ -- X₂ (3 binary variables)
//
// Vertex potentials (unnormalized):
// φ₀ = [1, 2]  → P(X₀=1) ∝ 2
// φ₁ = [1, 3]  → P(X₁=1) ∝ 3
// φ₂ = [1, 1]  → P(X₂=1) ∝ 1 (uniform)
//
// Edge potentials (encourage agreement):
// ψ = [[2, 1],   → same state: weight 2
//      [1, 2]]   → different state: weight 1
//
// Partition function:
// Z = einsum("i,ij,j,jk,k->", φ₀, ψ₀₁, φ₁, ψ₁₂, φ₂)
//
// To get marginal P(X₁=1):
// 1. Set φ₁ = [1, θ] with θ as variable
// 2. Compute Z(θ) and ∂Z/∂θ
// 3. P(X₁=1) = (θ/Z) × ∂Z/∂θ = ∂log(Z)/∂log(θ)
//
// Manual enumeration for verification:
// All 2³ = 8 configurations, sum weights, compute marginals
```

### Verification

Manual enumeration of all 8 configurations:
```
X₀ X₁ X₂ | φ₀  ψ₀₁  φ₁  ψ₁₂  φ₂  | Product
---------|--------------------------|--------
0  0  0  |  1   2    1    2    1  |    4
0  0  1  |  1   2    1    1    1  |    2
0  1  0  |  1   1    3    1    1  |    3
0  1  1  |  1   1    3    2    1  |    6
1  0  0  |  2   1    1    2    1  |    4
1  0  1  |  2   1    1    1    1  |    2
1  1  0  |  2   2    3    1    1  |   12
1  1  1  |  2   2    3    2    1  |   24
---------|--------------------------|--------
                              Z =    57

P(X₁=1) = (3+6+12+24)/57 = 45/57 ≈ 0.789
P(X₁=0) = (4+2+4+2)/57 = 12/57 ≈ 0.211

Gradient check: ∂Z/∂φ₁[1] should equal 45/3 = 15
(since φ₁[1] = 3 appears in all X₁=1 terms)
```

### References

- Probabilistic graphical models and belief propagation
- Connection to tropical semiring: max-marginals via MaxPlus algebra

---

## Example 2: Tensor Train Ground State (Complex Numbers)

### Problem Statement

Find the ground state of a 5-site spin-1/2 Heisenberg chain using a Matrix Product State (MPS) ansatz. Gradients enable variational optimization of the tensor network.

### Mathematical Formulation

An MPS represents a quantum state as a chain of tensors:

```
|ψ⟩ = Σ_{s₁...s₅} A¹[s₁] · A²[s₂] · A³[s₃] · A⁴[s₄] · A⁵[s₅] |s₁...s₅⟩
```

Where each `Aⁱ[sᵢ]` is a `χ × χ` complex matrix (χ = bond dimension).

The Heisenberg Hamiltonian:
```
H = Σᵢ (SxᵢSxᵢ₊₁ + SyᵢSyᵢ₊₁ + SzᵢSzᵢ₊₁)
```

Local 2-site Hamiltonian (4×4 matrix):
```
h = 0.25 × [[1,  0,  0, 0],
            [0, -1,  2, 0],
            [0,  2, -1, 0],
            [0,  0,  0, 1]]
```

Energy expectation:
```
E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ = Σᵢ ⟨ψ|hᵢ,ᵢ₊₁|ψ⟩ / ⟨ψ|ψ⟩
```

### Gradient Meaning

- `∂E/∂Aⁱ` = gradient of energy with respect to tensor at site i
- At optimum: gradient norm → 0 (variational minimum)
- Used for: gradient descent optimization toward ground state

### Test Case

```rust
// 5-site spin-1/2 Heisenberg chain
// Hilbert space dimension: 2⁵ = 32 (exactly diagonalizable)
//
// MPS with bond dimension χ=4 (sufficient for exact ground state):
// A1: shape [1, 2, 4]   (left boundary)
// A2: shape [4, 2, 4]   (bulk)
// A3: shape [4, 2, 4]   (bulk)
// A4: shape [4, 2, 4]   (bulk)
// A5: shape [4, 2, 1]   (right boundary)
//
// Einsum for wavefunction contraction:
// ψ[s1,s2,s3,s4,s5] = einsum("asb,bsc,csd,dse,esf->s1s2s3s4s5", A1,A2,A3,A4,A5)
//
// Energy = sum of local terms on 4 bonds:
// E = ⟨ψ|h₁₂|ψ⟩ + ⟨ψ|h₂₃|ψ⟩ + ⟨ψ|h₃₄|ψ⟩ + ⟨ψ|h₄₅|ψ⟩
//
// Exact ground state energy: E₀ ≈ -1.7608 (units of J)
//
// Optimization loop:
// 1. Initialize random complex MPS tensors
// 2. Compute E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩
// 3. Compute gradients ∂E/∂Aᵢ via einsum_with_grad
// 4. Update Aᵢ ← Aᵢ - η·∂E/∂Aᵢ
// 5. Repeat until E converges to E₀
```

### Verification

- Exact diagonalization of 32×32 Hamiltonian matrix
- Compare MPS energy against exact E₀ ≈ -1.7608
- Reference: https://tensornetwork.org/mps/algorithms/dmrg/

### Implementation Notes

Requires complex number support:
```rust
use num_complex::Complex64;

// Complex einsum uses Standard<Complex64> algebra
// Inner product: ⟨ψ|ψ⟩ requires conjugation of bra
// Energy: real-valued output from Hermitian operator
```

---

## Example 3: Maximum Weight Independent Set (Tropical Numbers)

### Problem Statement

Find the maximum weight independent set on a pentagon graph using tropical tensor network contraction. Gradients reveal which vertices are in the optimal set.

### Mathematical Formulation

**Vertex tensor** for vertex v with weight wᵥ:
```
W[sᵥ] = [0, wᵥ]  where sᵥ ∈ {0, 1}
```
- sᵥ = 0: vertex not in set (contributes 0 in tropical = multiplicative identity)
- sᵥ = 1: vertex in set (contributes wᵥ)

**Edge tensor** for edge (u,v):
```
B[sᵤ, sᵥ] = [[0,    0   ],
             [0, -∞ ]]
```
- B[1,1] = -∞ enforces independence constraint (tropical zero = additive identity)

**Tropical contraction** (MaxPlus algebra: ⊕ = max, ⊗ = +):
```
Result = max over all valid configurations of Σ(weights of selected vertices)
```

### Gradient Meaning

From "Tropical Tensor Network for Ground States of Spin Glasses":
> "Differentiating through the tensor network contraction gives the ground state configuration"

- `∂(max_weight)/∂(wᵥ)` = 1 if vertex v is in optimal set, 0 otherwise
- The tropical gradient directly reveals the optimal selection!

### Test Case

```rust
// Pentagon graph (5 vertices, 5 edges):
//
//       0 (w=3)
//      / \
//     4   1
//    (2) (5)
//     |   |
//     3---2
//    (4) (1)
//
// Edges: (0,1), (1,2), (2,3), (3,4), (4,0)
// Vertex weights: [3, 5, 1, 4, 2]
//
// Tensor network construction:
// - 5 vertex tensors: W₀, W₁, W₂, W₃, W₄
// - 5 edge tensors: B₀₁, B₁₂, B₂₃, B₃₄, B₄₀
//
// Contract all tensors:
// result = einsum("a,b,c,d,e,ab,bc,cd,de,ea->", W0,W1,W2,W3,W4,B01,B12,B23,B34,B40)
//
// Expected output:
// - Forward: max_weight = 9
// - Gradient: ∂result/∂w = [0, 1, 0, 1, 0]
//   → vertices 1 and 3 form the optimal independent set
```

### Verification

All independent sets of pentagon (manual enumeration):
```
Single vertices:
  {0}: 3, {1}: 5, {2}: 1, {3}: 4, {4}: 2

Pairs (non-adjacent only):
  {0,2}: 4, {0,3}: 7, {1,3}: 9, {1,4}: 7, {2,4}: 3

Maximum = 9, achieved by {1,3} ✓
```

### References

- Liu & Wang, "Tropical Tensor Network for Ground States of Spin Glasses", PRL 2021
  https://arxiv.org/abs/2008.06888
- Liu et al., "Computing Solution Space Properties via Generic Tensor Networks", SIAM 2023
  https://epubs.siam.org/doi/10.1137/22M1501787
- GenericTensorNetworks.jl documentation
  https://queracomputing.github.io/GenericTensorNetworks.jl/dev/

---

## Implementation Plan

### Task 1: Add Complex Number Support

**Files:**
- Modify: `src/algebra/standard.rs`
- Modify: `src/algebra/mod.rs`
- Create: `tests/complex.rs`

Add `Standard<Complex64>` algebra implementation with proper conjugation for inner products.

### Task 2: Create Bayesian Network Example Test

**Files:**
- Create: `tests/showcase.rs`

Implement `test_bayesian_network_marginals()` with:
- 3-node chain: X₀ - X₁ - X₂
- Partition function via einsum
- Gradient gives marginal probabilities
- Verify against manual enumeration (8 configurations)

### Task 3: Create Tensor Train Example Test

**Files:**
- Modify: `tests/showcase.rs`

Implement `test_tensor_train_ground_state()` with:
- 5-site Heisenberg chain
- MPS contraction via einsum
- Energy gradient computation
- Comparison against exact diagonalization

### Task 4: Create Independent Set Example Test

**Files:**
- Modify: `tests/showcase.rs`

Implement `test_max_weight_independent_set()` with:
- Pentagon graph
- Tropical tensor network construction
- MaxPlus contraction and gradient
- Verify gradient gives optimal selection

### Task 5: Add Documentation Examples

**Files:**
- Modify: `examples/basic_einsum.rs` or create new example files

Add runnable examples with explanatory comments for each showcase.

---

## Success Criteria

1. All three tests pass with correct numerical results
2. Gradients match expected values (manual calculation or reference implementation)
3. Examples are documented with clear physical/mathematical motivation
4. Code demonstrates practical use cases for einsum gradients
