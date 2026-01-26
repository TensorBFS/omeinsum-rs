# Unary Einsum Gradient Design

**Goal:** Support gradients for unary einsum operations (trace, sum, diagonal, transpose).

**Related issues:** #4 (Standard algebra), #9 (Tropical algebra)

---

## Problem

Currently `einsum_with_grad()` doesn't compute gradients for single-tensor operations because:
1. The optimizer creates `Leaf` nodes for single tensors
2. `execute_unary_naive()` handles forward pass but doesn't track gradient info
3. No `contract_unary_backward()` exists

## Solution: Index-Exchange Trick

The elegant insight from OMEinsum.jl: **backward is just einsum with swapped indices**.

### Standard Algebra

For any unary einsum:
- Forward: `y = einsum(ix -> iy, x)`
- Backward: `grad_x = einsum(iy -> ix, grad_y)`

| Forward | Backward | Operation |
|---------|----------|-----------|
| `ii->` | `->ii` | trace → embed to diagonal |
| `ij->i` | `i->ij` | sum over j → broadcast along j |
| `ij->ji` | `ji->ij` | transpose → transpose back |
| `ii->i` | `i->ii` | extract diagonal → embed to diagonal |
| `ij->` | `->ij` | sum all → broadcast to all |

### Tropical Algebra

Tropical gradients route through argmax (only winner gets gradient):

| Forward | Gradient behavior |
|---------|-------------------|
| `ii->` (max diagonal) | Only max diagonal element gets gradient |
| `ij->i` (row max) | Only argmax column per row gets gradient |
| `ij->` (global max) | Only single max element gets gradient |

Requires tracking argmax during forward pass.

---

## Implementation

### File: `src/einsum/backward.rs`

```rust
/// Backward pass for unary einsum.
pub fn contract_unary_backward<A, T, B>(
    grad_y: &Tensor<T, B>,
    x: &Tensor<T, B>,
    argmax: Option<&Tensor<u32, B>>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend + Default,
{
    if A::needs_argmax() {
        tropical_unary_backward::<A, T, B>(grad_y, x, argmax.unwrap(), ix, iy)
    } else {
        standard_unary_backward::<A, T, B>(grad_y, ix, iy, size_dict)
    }
}

/// Standard unary backward: just swap indices.
fn standard_unary_backward<A, T, B>(
    grad_y: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B> {
    // The magic: einsum(iy -> ix, grad_y)
    execute_unary_naive::<A, T, B>(grad_y, iy, ix, size_dict)
}

/// Tropical unary backward: scatter gradient to argmax positions.
fn tropical_unary_backward<A, T, B>(
    grad_y: &Tensor<T, B>,
    x: &Tensor<T, B>,
    argmax: &Tensor<u32, B>,
    ix: &[usize],
    iy: &[usize],
) -> Tensor<T, B> {
    // Create zero tensor, scatter grad_y to winner positions
    // Implementation depends on operation type
    todo!("Implement tropical unary backward")
}
```

### File: `src/einsum/engine.rs`

Add `execute_unary_with_argmax()`:

```rust
fn execute_unary_with_argmax<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> (Tensor<T, B>, Option<Tensor<u32, B>>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend + Default,
{
    let result = execute_unary_naive::<A, T, B>(tensor, ix, iy, size_dict);

    let argmax = if A::needs_argmax() {
        Some(compute_unary_argmax::<A, T, B>(tensor, ix, iy, size_dict))
    } else {
        None
    };

    (result, argmax)
}
```

Update Leaf handling in `execute_with_argmax()`:

```rust
if let NestedEinsum::Leaf { tensor_index } = tree {
    let (result, argmax) = execute_unary_with_argmax::<A, T, B>(
        tensors[*tensor_index],
        &self.ixs[*tensor_index],
        &self.iy,
        &self.size_dict,
    );
    if let Some(am) = argmax {
        argmax_cache.push(am);
    }
    // Store unary info for backward pass
    return result;
}
```

### File: `src/einsum/mod.rs`

Update `GradientInfo`:

```rust
pub struct GradientInfo {
    // ... existing fields ...

    /// Info for unary operations (if applicable)
    pub unary_info: Option<UnaryGradInfo>,
}

pub struct UnaryGradInfo {
    pub ix: Vec<usize>,
    pub iy: Vec<usize>,
    pub size_dict: HashMap<usize, usize>,
}
```

---

## Tasks

1. Add `contract_unary_backward()` in backward.rs
2. Add `standard_unary_backward()` - index swap trick
3. Add `tropical_unary_backward()` - argmax scatter (can be follow-up)
4. Add `execute_unary_with_argmax()` in engine.rs
5. Update `GradientInfo` struct
6. Update Leaf handling in `execute_with_argmax()`
7. Un-ignore `test_trace_gradient`, `test_sum_gradient`

## Success Criteria

1. `einsum_with_grad("ii->", A)` returns correct trace gradient (identity matrix structure)
2. `einsum_with_grad("ij->i", A)` returns correct sum gradient (broadcast)
3. Existing binary gradient tests still pass
4. Issues #4 closed (Standard), #9 updated (Tropical design documented)
