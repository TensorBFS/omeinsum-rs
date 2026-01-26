//! CPU tensor contraction via reshape→GEMM→reshape.

use std::collections::HashSet;

/// Classify modes into batch, left-only, right-only, and contracted.
///
/// - batch: in both A and B, and in output C
/// - left: only in A (free indices from A)
/// - right: only in B (free indices from B)
/// - contracted: in both A and B, but NOT in output C
pub(super) fn classify_modes(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let a_set: HashSet<i32> = modes_a.iter().copied().collect();
    let b_set: HashSet<i32> = modes_b.iter().copied().collect();
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();

    let mut batch = Vec::new();
    let mut left = Vec::new();
    let mut contracted = Vec::new();

    for &m in modes_a {
        if b_set.contains(&m) && c_set.contains(&m) {
            if !batch.contains(&m) {
                batch.push(m);
            }
        } else if b_set.contains(&m) && !c_set.contains(&m) {
            if !contracted.contains(&m) {
                contracted.push(m);
            }
        } else if !left.contains(&m) {
            left.push(m);
        }
    }

    let right: Vec<i32> = modes_b
        .iter()
        .filter(|m| !a_set.contains(m))
        .copied()
        .collect();

    (batch, left, right, contracted)
}

/// Find the position of a mode in a modes array.
pub(super) fn mode_position(modes: &[i32], mode: i32) -> usize {
    modes.iter().position(|&m| m == mode).expect("mode not found")
}

/// Compute the product of dimensions for given modes.
pub(super) fn product_of_dims(modes: &[i32], all_modes: &[i32], shape: &[usize]) -> usize {
    modes
        .iter()
        .map(|&m| shape[mode_position(all_modes, m)])
        .product::<usize>()
        .max(1)
}

/// Compute permutation to reorder modes to [first..., second..., third...].
pub(super) fn compute_permutation(
    current: &[i32],
    first: &[i32],
    second: &[i32],
    third: &[i32],
) -> Vec<usize> {
    let target: Vec<i32> = first
        .iter()
        .chain(second.iter())
        .chain(third.iter())
        .copied()
        .collect();

    target
        .iter()
        .map(|m| mode_position(current, *m))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_modes_matmul() {
        // ij,jk->ik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1], &[1, 2], &[0, 2]);

        assert!(batch.is_empty());
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![2]);
        assert_eq!(contracted, vec![1]);
    }

    #[test]
    fn test_classify_modes_batched() {
        // bij,bjk->bik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(batch, vec![0]);
        assert_eq!(left, vec![1]);
        assert_eq!(right, vec![3]);
        assert_eq!(contracted, vec![2]);
    }

    #[test]
    fn test_product_of_dims() {
        let modes = &[0, 1, 2];
        let shape = &[2, 3, 4];

        assert_eq!(product_of_dims(&[0], modes, shape), 2);
        assert_eq!(product_of_dims(&[1, 2], modes, shape), 12);
        assert_eq!(product_of_dims(&[], modes, shape), 1);
    }

    #[test]
    fn test_compute_permutation() {
        // Current: [0, 1, 2], want: [0, 2, 1]
        let perm = compute_permutation(&[0, 1, 2], &[0], &[2], &[1]);
        assert_eq!(perm, vec![0, 2, 1]);
    }
}
