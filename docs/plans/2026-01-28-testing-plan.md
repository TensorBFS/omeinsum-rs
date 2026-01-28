# Test Coverage Plan: Porting OMEinsum.jl Tests to omeinsum-rs

## Executive Summary

This plan covers porting all relevant tests from Julia OMEinsum.jl (`~/.julia/dev/OMEinsum`) to Rust omeinsum-rs. The goal is comprehensive test coverage matching or exceeding the Julia implementation.

---

## Current State Analysis

### Julia OMEinsum Test Files (14 files, ~2100 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `einsum.jl` | 346 | Core einsum operations, size dict, string parsing |
| `unaryrules.jl` | 59 | Unary ops: Duplicate, Diag, Repeat, Tr, Permutedims, Sum |
| `binaryrules.jl` | 86 | Binary ops: analyze_binary, SimpleBinaryRule patterns |
| `matchrule.jl` | 117 | Rule matching: match_rule_binary, match_rule_unary |
| `autodiff.jl` | 148 | Gradient computation with bpcheck utility |
| `bp.jl` | 17 | Back-propagation with cost_and_gradient |
| `Core.jl` | 83 | EinCode types, constructors |
| `interfaces.jl` | 20 | @ein_str macro, string parsing |
| `contractionorder.jl` | 133 | Optimizer tests, complexity |
| `slicing.jl` | 39 | SliceIterator, memory-efficient execution |
| `utils.jl` | 54 | Utility functions |
| `cueinsum.jl` | 373 | CUDA backend tests |
| `roceinsum.jl` | 181 | AMD GPU tests |

### Rust omeinsum-rs Test Files (12 files, ~1800 LOC)

| File | Status | Coverage |
|------|--------|----------|
| `integration.rs` | Complete | Basic matmul, tropical |
| `omeinsum_compat.rs` | Complete | Core einsum, gradients, tropical |
| `backward.rs` | Complete | Gradient backward pass |
| `unary_ops.rs` | Complete | Trace, diag, sum, permute |
| `binary_rules.rs` | Complete | Binary contractions |
| `tropical.rs` | Complete | Tropical algebra |
| `cuda.rs` | Complete | CUDA backend |
| `showcase.rs` | Complete | Demo examples |
| `coverage.rs` | Partial | General coverage |
| `einsum_core.rs` | Partial | Core tests |
| `optimizer.rs` | Partial | Optimizer tests |
| `backend_contract.rs` | Complete | Backend trait tests |

---

## Gap Analysis: Missing Tests

### 1. Match Rule Tests (Priority: HIGH)

**Julia source:** `test/matchrule.jl` (117 LOC)
**Status:** Not ported to Rust

Tests needed for `tests/match_rule.rs`:
- [ ] `match_rule_binary` edge cases with repeated indices
- [ ] SimpleBinaryRule pattern matching for all 18+ variants
- [ ] DefaultRule fallback detection
- [ ] Unary rule classification: Tr, Sum, Diag, Permutedims, Identity, Duplicate
- [ ] `nopermute` helper function
- [ ] `isbatchmul` detection

**Key Julia tests to port:**
```julia
@test match_rule(((1,1),), ()) == Tr()
@test match_rule(((1,2),), ()) == Sum()
@test match_rule(((1,1),), (1,)) == Diag()
@test match_rule(((1,2),), (2,1)) == Permutedims()
@test match_rule(((1,2),(2,3)), (1,3)) == SimpleBinaryRule(ein"ij,jk->ik")
@test match_rule_binary([3], [3], [3,3]) isa DefaultRule  # repeated indices
@test match_rule_binary([3,3], [3], [3,3]) isa DefaultRule
```

### 2. Advanced Binary Rules (Priority: HIGH)

**Julia source:** `test/binaryrules.jl` (86 LOC)
**Status:** Partially ported

Missing tests for `tests/binary_rules.rs`:
- [ ] `analyze_binary` function with complex index patterns
- [ ] All 100+ SimpleBinaryRule pattern combinations with batch
- [ ] Complex einsum patterns:
  - `ein"abb,bc->ac"` (diagonal in first tensor)
  - `ein"ab,bc->acc"` (diagonal in output)
  - `ein"ab,bce->ac"` (sum over extra index)
  - `ein"bal,bcl->lcae"` (permutation in both inputs)
  - `ein"ddebal,bcf->lcac"` (all transformations combined)
- [ ] Regression test: 8D tensor × 5D tensor contraction
- [ ] Polynomial scalar multiplication (if supported)

**Julia test to port:**
```julia
@testset "binary rules" begin
    for has_batch in [true, false]
        for i1 in [(), ('i',), ('j',), ('i','j'), ('j','i')]
            for i2 in [(), ('k',), ('j',), ('k','j'), ('j','k')]
                for i3 in [(), ('i',), ('k',), ('i','k'), ('k','i')]
                    # Test all 125+ combinations
                end
            end
        end
    end
end
```

### 3. Advanced Unary Rules (Priority: MEDIUM)

**Julia source:** `test/unaryrules.jl` (59 LOC)
**Status:** Partially ported

Missing tests for `tests/unary_ops.rs`:
- [ ] **Duplicate** operation (`ijk -> k,l,j,i,i,l` - repeat indices)
- [ ] **Diag** with alpha/beta scaling (`α*result + β*output`)
- [ ] **Repeat** operation (broadcast along new axis)
- [ ] **Identity** with scaling factors
- [ ] **Sum** with scaling factors

**Julia tests to port:**
```julia
# Duplicate: adds repeated indices to output
@test unary_einsum!(Duplicate(), ix, iy, x, y, true, false) ≈ loop_einsum(...)

# Diag with scaling
@test unary_einsum!(Diag(), ix, iy, x, copy(y), 2.0, 3.0) ≈ 2*result + 3y

# Repeat: broadcasts along new dimension
@test unary_einsum!(Repeat(), (1,2,3), (3,4,2,1), x, y, true, false) ≈ ...
```

### 4. Einsum Core Tests (Priority: HIGH)

**Julia source:** `test/einsum.jl` (346 LOC)
**Status:** Partially ported

Missing tests for `tests/einsum_core.rs`:
- [ ] **Size dictionary validation** (`get_size_dict`)
- [ ] **Tensor order checking** (argument count validation)
- [ ] **Output array type inference**
- [ ] **Unicode index support** (α, β, γ, δ) - low priority
- [ ] **Allow loops** configuration toggle
- [ ] **Star contraction**: `ein"ai,bi,ci->abc"` (3 tensors share one index)
- [ ] **Star and contract**: `ein"(1,2),(1,2),(1,3)->(3,)"`
- [ ] **Projection to diagonal**: `ein"ii->ii"` (zeros off-diagonal)
- [ ] **Combined operations**: `ein"(1,1,2,2)->()"` (double trace)
- [ ] **Issue #136 regression**: empty dimension handling
- [ ] **Macro input tests**: @ein, @ein! equivalents

**Key Julia tests to port:**
```julia
# Size dict validation
@test_throws DimensionMismatch get_size_dict((('i','j'), ('j','k')), (a, a))

# Tensor order check
@test_throws ArgumentError get_size_dict(ixs, (a,a,a))  # wrong count

# Star contraction
aaa = einsum(EinCode(((1,2),(1,3),(1,4)),(2,3,4)), (a,a,a))

# Projection to diagonal (ii->ii zeros off-diagonal)
a2 = [a[1] 0; 0 a[4]]
@test einsum(EinCode(((1,1),), (1,1)), (a,)) ≈ a2

# Issue #136: empty dimensions
@test EinCode(((1,2,3),(2,)),(1,3))(ones(2,2,0), ones(2)) == reshape(zeros(0), 2, 0)
```

### 5. Comprehensive Autodiff Tests (Priority: HIGH)

**Julia source:** `test/autodiff.jl` (148 LOC)
**Status:** Partially ported

Missing tests for `tests/backward.rs`:
- [ ] **`bpcheck` utility** (finite difference gradient validation)
- [ ] **Complex number gradients** (Complex64) - already partial
- [ ] **Gradient type preservation**
- [ ] **Hessian computation** (if supported)
- [ ] **Nested einsum gradients**: `ein"(ab,abcd),c->ad"`
- [ ] **Sequence specification gradients**
- [ ] **Partial trace gradients**: `ein"(1,2,2,3)->(1,3)"`
- [ ] **Diag gradients**: `ein"(1,2,2,3)->(1,2,3)"`
- [ ] **Outer product gradients**

**Key Julia tests to port:**
```julia
# bpcheck: finite difference validation
function bpcheck(f, args...; η=1e-5)
    g = gradient(f, args...)
    dy_ref = η * sum(abs2, g)
    dy = f(args...) - f([arg .- η .* gi for (arg, gi) in zip(args, g)]...)
    isapprox(dy, dy_ref, rtol=1e-2, atol=1e-8)
end

# Partial trace gradient
@test bpcheck(aa -> einsum(EinCode(((1,2,2,3),), (1,3)), (aa,)) |> abs ∘ sum, aa)

# Nested einsum gradient
@test bpcheck((a,t,v) -> ein"(ab,abcd),c->ad"(a,t,v) |> abs ∘ sum, a, t, v)

# Hessian
@test Zygote.hessian(loss, x) == ForwardDiff.hessian(loss, x)
```

### 6. Back-propagation Tests (Priority: MEDIUM)

**Julia source:** `test/bp.jl` (17 LOC)
**Status:** Not ported

New file `tests/bp.rs`:
- [ ] `cost_and_gradient` function
- [ ] `gradient_tree` construction
- [ ] `extract_leaves` for leaf gradients
- [ ] Optimized code gradients (TreeSA optimizer)

**Julia tests to port:**
```julia
cost, mg = OMEinsum.cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))
@test cost[] ≈ cost0
@test all(zg .≈ mg)  # matches Zygote gradient
```

### 7. Contraction Order Optimizer Tests (Priority: HIGH)

**Julia source:** `test/contractionorder.jl` (133 LOC)
**Status:** Minimal in Rust

Enhance `tests/optimizer.rs`:
- [ ] Greedy optimizer produces valid tree
- [ ] TreeSA optimizer (if available)
- [ ] FLOP complexity calculation
- [ ] Space complexity calculation
- [ ] Optimal contraction path verification
- [ ] Fullerene C60 benchmark

---

## Implementation Plan

### Phase 1: Match Rule Tests (~300 LOC)
**File:** `tests/match_rule.rs` (new)

```rust
// tests/match_rule.rs structure
mod match_rule_binary_tests {
    fn test_simple_matmul_pattern() { }
    fn test_all_transposed_patterns() { }
    fn test_repeated_index_fallback() { }
    fn test_batch_patterns() { }
}

mod match_rule_unary_tests {
    fn test_trace_pattern() { }
    fn test_sum_pattern() { }
    fn test_diag_pattern() { }
    fn test_permutedims_pattern() { }
    fn test_identity_pattern() { }
    fn test_duplicate_pattern() { }
}

mod nopermute_tests {
    fn test_nopermute_positive() { }
    fn test_nopermute_negative() { }
}

mod isbatchmul_tests {
    fn test_batched_matmul_detection() { }
    fn test_non_batch_patterns() { }
}
```

### Phase 2: Enhance Binary Rules (~200 LOC)
**File:** `tests/binary_rules.rs` (extend)

```rust
// Add to existing binary_rules.rs
mod analyze_binary_tests {
    fn test_analyze_complex_indices() { }
    fn test_analyze_all_combinations() { }
}

mod complex_patterns_tests {
    fn test_diagonal_in_input() { }
    fn test_diagonal_in_output() { }
    fn test_sum_in_contraction() { }
    fn test_permutation_in_both() { }
    fn test_all_transforms_combined() { }
}

mod regression_tests {
    fn test_8d_5d_contraction() { }
}
```

### Phase 3: Enhance Unary Rules (~150 LOC)
**File:** `tests/unary_ops.rs` (extend)

```rust
// Add to existing unary_ops.rs
mod duplicate_tests {
    fn test_duplicate_single_index() { }
    fn test_duplicate_multiple_indices() { }
}

mod scaling_tests {
    fn test_diag_with_alpha_beta() { }
    fn test_sum_with_scaling() { }
    fn test_identity_with_scaling() { }
}

mod repeat_tests {
    fn test_repeat_along_new_axis() { }
    fn test_repeat_multiple_axes() { }
}
```

### Phase 4: Einsum Core Enhancement (~250 LOC)
**File:** `tests/einsum_core.rs` (extend)

```rust
// Add to existing einsum_core.rs
mod size_dict_tests {
    fn test_size_dict_validation() { }
    fn test_dimension_mismatch() { }
}

mod tensor_order_tests {
    fn test_argument_count_validation() { }
    fn test_ndim_validation() { }
}

mod star_contraction_tests {
    fn test_star_contraction_3_tensors() { }
    fn test_star_and_contract() { }
}

mod combined_ops_tests {
    fn test_projection_to_diagonal() { }
    fn test_double_trace() { }
}

mod empty_dimension_tests {
    fn test_issue_136_empty_dimension() { }
}
```

### Phase 5: Autodiff Enhancement (~200 LOC)
**File:** `tests/backward.rs` (extend)

```rust
// Add to existing backward.rs
mod bpcheck_tests {
    fn bpcheck<F>(f: F, args: &[&Tensor<f64>]) -> bool { }
    fn test_matmul_bpcheck() { }
    fn test_trace_bpcheck() { }
    fn test_partial_trace_bpcheck() { }
}

mod complex_gradient_tests {
    fn test_complex64_matmul_gradient() { }
    fn test_complex64_type_preservation() { }
}

mod nested_einsum_tests {
    fn test_nested_einsum_gradient() { }
    fn test_sequence_spec_gradient() { }
}
```

### Phase 6: New Test Files (~150 LOC)

**File:** `tests/bp.rs` (new)
```rust
mod cost_and_gradient_tests {
    fn test_cost_and_gradient_matches_backward() { }
    fn test_optimized_code_gradient() { }
}
```

**File:** `tests/contraction_order.rs` (new or extend optimizer.rs)
```rust
mod optimizer_tests {
    fn test_greedy_optimizer() { }
    fn test_complexity_calculation() { }
    fn test_fullerene_c60_benchmark() { }
}
```

---

## Test Matrix: Julia → Rust Mapping

| Julia Test | Rust Target | Priority | Status |
|------------|-------------|----------|--------|
| `matchrule.jl:match_rule_binary` | `match_rule.rs` | HIGH | ❌ New |
| `matchrule.jl:match_rule (unary)` | `match_rule.rs` | HIGH | ❌ New |
| `matchrule.jl:isbatchmul` | `match_rule.rs` | HIGH | ❌ New |
| `binaryrules.jl:analyze_binary` | `binary_rules.rs` | HIGH | ❌ Add |
| `binaryrules.jl:all_patterns` | `binary_rules.rs` | HIGH | ⚠️ Partial |
| `binaryrules.jl:regression` | `binary_rules.rs` | MEDIUM | ❌ Add |
| `unaryrules.jl:Duplicate` | `unary_ops.rs` | MEDIUM | ❌ Add |
| `unaryrules.jl:Diag scaling` | `unary_ops.rs` | MEDIUM | ❌ Add |
| `unaryrules.jl:Repeat` | `unary_ops.rs` | MEDIUM | ❌ Add |
| `einsum.jl:get_size_dict` | `einsum_core.rs` | HIGH | ❌ Add |
| `einsum.jl:tensor_order` | `einsum_core.rs` | HIGH | ❌ Add |
| `einsum.jl:star_contraction` | `einsum_core.rs` | HIGH | ⚠️ Partial |
| `einsum.jl:projection_diag` | `einsum_core.rs` | MEDIUM | ❌ Add |
| `einsum.jl:issue_136` | `einsum_core.rs` | LOW | ❌ Add |
| `autodiff.jl:bpcheck` | `backward.rs` | HIGH | ❌ Add |
| `autodiff.jl:Complex64` | `backward.rs` | HIGH | ✅ Done |
| `autodiff.jl:nested` | `backward.rs` | MEDIUM | ❌ Add |
| `autodiff.jl:hessian` | `backward.rs` | LOW | ❌ Future |
| `bp.jl:cost_and_gradient` | `bp.rs` | MEDIUM | ❌ New |
| `contractionorder.jl:greedy` | `optimizer.rs` | HIGH | ⚠️ Partial |
| `contractionorder.jl:complexity` | `optimizer.rs` | MEDIUM | ❌ Add |

---

## Estimated Effort

| Phase | New LOC | Effort |
|-------|---------|--------|
| Phase 1: Match Rules | ~300 | 2-3 hours |
| Phase 2: Binary Rules | ~200 | 1-2 hours |
| Phase 3: Unary Rules | ~150 | 1-2 hours |
| Phase 4: Einsum Core | ~250 | 2-3 hours |
| Phase 5: Autodiff | ~200 | 2-3 hours |
| Phase 6: BP + Optimizer | ~150 | 1-2 hours |
| **Total** | **~1250** | **9-15 hours** |

---

## Success Criteria

1. All HIGH priority tests pass
2. Test coverage matches or exceeds Julia (~97%)
3. All existing tests continue to pass
4. `cargo test --features "tropical"` passes
5. CI/CD passes with new tests
6. No regressions in performance

---

## Notes

- **SymEngine equivalent:** Julia uses SymEngine for symbolic tests - use numeric equivalents in Rust
- **Unicode indices:** Low priority, API uses usize indices in Rust
- **Hessian:** Requires second-order AD, may not be implemented
- **Scaling factors (alpha/beta):** Julia has `einsum!(... , α, β)` - check if Rust supports this
- **Focus:** Computational correctness over API parity with Julia
