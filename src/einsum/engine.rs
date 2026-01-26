//! Einsum execution engine with omeco integration.

use std::collections::{HashMap, HashSet};

use omeco::{optimize_code, EinCode, GreedyMethod, Label, NestedEinsum, TreeSA};

use crate::algebra::{Algebra, Scalar};
use crate::backend::Backend;
use crate::tensor::Tensor;

/// Einsum specification and execution engine.
///
/// Supports contraction order optimization via omeco.
///
/// # Example
///
/// ```rust,ignore
/// use omeinsum::Einsum;
/// use std::collections::HashMap;
///
/// // A[i,j] × B[j,k] × C[k,l] → D[i,l]
/// let sizes: HashMap<usize, usize> = [(0, 10), (1, 20), (2, 30), (3, 40)].into();
///
/// let mut ein = Einsum::new(
///     vec![vec![0, 1], vec![1, 2], vec![2, 3]],
///     vec![0, 3],
///     sizes,
/// );
///
/// ein.optimize_greedy();
/// let result = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a, &b, &c]);
/// ```
pub struct Einsum<L: Label = usize> {
    /// Input index labels for each tensor
    pub ixs: Vec<Vec<L>>,

    /// Output index labels
    pub iy: Vec<L>,

    /// Dimension sizes for each index
    pub size_dict: HashMap<L, usize>,

    /// Optimized contraction tree (after optimization)
    optimized: Option<NestedEinsum<L>>,
}

impl<L: Label> Einsum<L> {
    /// Create a new einsum specification.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Index labels for each input tensor
    /// * `iy` - Output index labels
    /// * `size_dict` - Mapping from index labels to dimension sizes
    pub fn new(ixs: Vec<Vec<L>>, iy: Vec<L>, size_dict: HashMap<L, usize>) -> Self {
        Self {
            ixs,
            iy,
            size_dict,
            optimized: None,
        }
    }

    /// Get the einsum code specification.
    pub fn code(&self) -> EinCode<L> {
        EinCode::new(self.ixs.clone(), self.iy.clone())
    }

    /// Optimize contraction order using greedy algorithm.
    ///
    /// Fast O(n²) algorithm, good for most cases.
    pub fn optimize_greedy(&mut self) -> &mut Self {
        let code = self.code();
        let optimizer = GreedyMethod::new(0.0, 0.0);
        self.optimized = optimize_code(&code, &self.size_dict, &optimizer);
        self
    }

    /// Optimize contraction order using simulated annealing.
    ///
    /// Slower but finds better orderings for complex networks.
    pub fn optimize_treesa(&mut self) -> &mut Self {
        let code = self.code();
        let optimizer = TreeSA::default();
        self.optimized = optimize_code(&code, &self.size_dict, &optimizer);
        self
    }

    /// Check if optimization has been performed.
    pub fn is_optimized(&self) -> bool {
        self.optimized.is_some()
    }

    /// Get the optimized contraction tree.
    pub fn contraction_tree(&self) -> Option<&NestedEinsum<L>> {
        self.optimized.as_ref()
    }
}

impl Einsum<usize> {
    /// Execute the einsum contraction.
    ///
    /// # Type Parameters
    ///
    /// * `A` - The algebra to use (e.g., `Standard<f32>`, `MaxPlus<f32>`)
    /// * `T` - The scalar type
    /// * `B` - The backend type
    pub fn execute<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        assert_eq!(
            tensors.len(),
            self.ixs.len(),
            "Number of tensors {} doesn't match number of index specs {}",
            tensors.len(),
            self.ixs.len()
        );

        match &self.optimized {
            Some(tree) => {
                // Handle top-level Leaf (single tensor) specially to apply unary transformations
                if let NestedEinsum::Leaf { tensor_index } = tree {
                    execute_unary_naive::<A, T, B>(
                        tensors[*tensor_index],
                        &self.ixs[*tensor_index],
                        &self.iy,
                        &self.size_dict,
                    )
                } else {
                    self.execute_tree::<A, T, B>(tree, tensors)
                }
            }
            None => self.execute_pairwise::<A, T, B>(tensors),
        }
    }

    /// Execute with argmax tracking for backpropagation.
    ///
    /// Returns `(result, argmax_cache)` where `argmax_cache` contains argmax
    /// tensors for each binary contraction in the execution tree.
    pub fn execute_with_argmax<A, T, B>(
        &self,
        tensors: &[&Tensor<T, B>],
    ) -> (Tensor<T, B>, Vec<Tensor<u32, B>>)
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        assert_eq!(
            tensors.len(),
            self.ixs.len(),
            "Number of tensors {} doesn't match number of index specs {}",
            tensors.len(),
            self.ixs.len()
        );

        let mut argmax_cache = Vec::new();

        let result = match &self.optimized {
            Some(tree) => self.execute_tree_with_argmax::<A, T, B>(tree, tensors, &mut argmax_cache),
            None => self.execute_pairwise_with_argmax::<A, T, B>(tensors, &mut argmax_cache),
        };

        (result, argmax_cache)
    }

    /// Execute an optimized contraction tree with argmax tracking.
    #[allow(clippy::only_used_in_recursion)]
    fn execute_tree_with_argmax<A, T, B>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[&Tensor<T, B>],
        argmax_cache: &mut Vec<Tensor<u32, B>>,
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        match tree {
            NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone(),
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "Expected binary contraction tree");

                let left = self.execute_tree_with_argmax::<A, T, B>(&args[0], tensors, argmax_cache);
                let right = self.execute_tree_with_argmax::<A, T, B>(&args[1], tensors, argmax_cache);

                let ia = &eins.ixs[0];
                let ib = &eins.ixs[1];
                let iy = &eins.iy;

                if A::needs_argmax() {
                    let (result, argmax) = left.contract_binary_with_argmax::<A>(&right, ia, ib, iy);
                    argmax_cache.push(argmax);
                    result
                } else {
                    left.contract_binary::<A>(&right, ia, ib, iy)
                }
            }
        }
    }

    /// Execute pairwise contraction with argmax tracking.
    fn execute_pairwise_with_argmax<A, T, B>(
        &self,
        tensors: &[&Tensor<T, B>],
        argmax_cache: &mut Vec<Tensor<u32, B>>,
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        if tensors.is_empty() {
            panic!("Cannot execute einsum with no tensors");
        }

        if tensors.len() == 1 {
            return execute_unary_naive::<A, T, B>(
                tensors[0],
                &self.ixs[0],
                &self.iy,
                &self.size_dict,
            );
        }

        // Contract left to right
        let mut result = tensors[0].clone();
        let mut current_indices = self.ixs[0].clone();

        for i in 1..tensors.len() {
            let other = tensors[i];
            let other_indices = &self.ixs[i];

            let intermediate_output = if i == tensors.len() - 1 {
                self.iy.clone()
            } else {
                compute_intermediate_output(&current_indices, other_indices, &self.iy)
            };

            if A::needs_argmax() {
                let (new_result, argmax) = result.contract_binary_with_argmax::<A>(
                    other,
                    &current_indices,
                    other_indices,
                    &intermediate_output,
                );
                argmax_cache.push(argmax);
                result = new_result;
            } else {
                result = result.contract_binary::<A>(
                    other,
                    &current_indices,
                    other_indices,
                    &intermediate_output,
                );
            }
            current_indices = intermediate_output;
        }

        result
    }

    /// Execute an optimized contraction tree.
    #[allow(clippy::only_used_in_recursion)]
    fn execute_tree<A, T, B>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[&Tensor<T, B>],
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        match tree {
            NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone(),
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "Expected binary contraction tree");

                let left = self.execute_tree::<A, T, B>(&args[0], tensors);
                let right = self.execute_tree::<A, T, B>(&args[1], tensors);

                let ia = &eins.ixs[0];
                let ib = &eins.ixs[1];
                let iy = &eins.iy;

                left.contract_binary::<A>(&right, ia, ib, iy)
            }
        }
    }

    /// Execute using simple pairwise contraction (no optimization).
    fn execute_pairwise<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend + Default,
    {
        if tensors.is_empty() {
            panic!("Cannot execute einsum with no tensors");
        }

        if tensors.len() == 1 {
            // Single tensor: just trace/reduce if needed
            return execute_unary_naive::<A, T, B>(
                tensors[0],
                &self.ixs[0],
                &self.iy,
                &self.size_dict,
            );
        }

        // Contract left to right
        let mut result = tensors[0].clone();
        let mut current_indices = self.ixs[0].clone();

        for i in 1..tensors.len() {
            let other = tensors[i];
            let other_indices = &self.ixs[i];

            // Determine output indices for this contraction
            let intermediate_output = if i == tensors.len() - 1 {
                // Last contraction: use final output
                self.iy.clone()
            } else {
                // Intermediate: keep all non-contracted indices
                compute_intermediate_output(&current_indices, other_indices, &self.iy)
            };

            result = result.contract_binary::<A>(
                other,
                &current_indices,
                other_indices,
                &intermediate_output,
            );
            current_indices = intermediate_output;
        }

        result
    }
}

/// Compute intermediate output indices for pairwise contraction.
fn compute_intermediate_output(
    ia: &[usize],
    ib: &[usize],
    final_output: &[usize],
) -> Vec<usize> {
    let final_set: std::collections::HashSet<_> = final_output.iter().copied().collect();
    let ia_set: std::collections::HashSet<_> = ia.iter().copied().collect();
    let ib_set: std::collections::HashSet<_> = ib.iter().copied().collect();

    // Keep indices that are in the final output OR appear in only one input
    let mut output = Vec::new();

    for &i in ia {
        if (final_set.contains(&i) || !ib_set.contains(&i)) && !output.contains(&i) {
            output.push(i);
        }
    }

    for &i in ib {
        if (final_set.contains(&i) || !ia_set.contains(&i)) && !output.contains(&i) {
            output.push(i);
        }
    }

    output
}

/// Convert linear index to multi-dimensional index (column-major).
///
/// Given a flat/linear index and a shape, returns the multi-dimensional
/// coordinates for column-major storage order.
///
/// # Arguments
///
/// * `linear` - The flat index into the tensor
/// * `shape` - The shape of the tensor
///
/// # Returns
///
/// A vector of indices, one per dimension
fn linear_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut multi = vec![0; shape.len()];
    for i in 0..shape.len() {
        multi[i] = linear % shape[i];
        linear /= shape[i];
    }
    multi
}

/// Compute input tensor position from index values (column-major).
///
/// Given index labels and their current values, computes the flat position
/// in the input tensor using column-major ordering.
///
/// # Arguments
///
/// * `ix` - The index labels for the input tensor
/// * `idx_values` - Mapping from index label to current value
/// * `shape` - The shape of the input tensor
///
/// # Returns
///
/// The flat position in the tensor
fn compute_input_position(
    ix: &[usize],
    idx_values: &HashMap<usize, usize>,
    shape: &[usize],
) -> usize {
    let mut pos = 0;
    let mut stride = 1;
    for (dim, &idx) in ix.iter().enumerate() {
        pos += idx_values[&idx] * stride;
        stride *= shape[dim];
    }
    pos
}

/// Execute unary einsum operation using naive loop.
/// Handles trace, diagonal, sum, permutation uniformly.
///
/// # Type Parameters
///
/// * `A` - The algebra to use for accumulation
/// * `T` - The scalar type
/// * `B` - The backend type
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `ix` - Input index labels (may contain repeated indices for trace/diagonal)
/// * `iy` - Output index labels
/// * `size_dict` - Mapping from index labels to dimension sizes
///
/// # Key Insight
///
/// For repeated indices like `ix = [0, 1, 1, 2]` (ijjk), positions 1 and 2 both map
/// to index label `1`. This automatically handles diagonal extraction because
/// `compute_input_position` uses `idx_values[&idx]` - when the same index label
/// appears multiple times in `ix`, those positions will use the same value.
fn execute_unary_naive<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T>,
    T: Scalar,
    B: Backend + Default,
{
    // 1. Classify indices
    // outer = output indices
    // inner = indices that appear in input but not in output (summed over)
    let outer: &[usize] = iy;
    let outer_set: HashSet<usize> = outer.iter().copied().collect();
    let inner_vec: Vec<usize> = ix
        .iter()
        .copied()
        .filter(|i| !outer_set.contains(i))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    // 2. Build output shape
    let out_shape: Vec<usize> = outer.iter().map(|&idx| size_dict[&idx]).collect();
    let out_size = out_shape.iter().product::<usize>().max(1);

    // 3. Build inner ranges (dimensions to sum over)
    let inner_ranges: Vec<usize> = inner_vec.iter().map(|&idx| size_dict[&idx]).collect();
    let inner_size = inner_ranges.iter().product::<usize>().max(1);

    // 4. Allocate output
    let mut out_data = vec![A::zero().to_scalar(); out_size];

    // 5. Loop over output positions
    for out_linear in 0..out_size {
        let out_multi = linear_to_multi(out_linear, &out_shape);

        // Map: outer index label -> value
        let mut idx_values: HashMap<usize, usize> = outer
            .iter()
            .zip(out_multi.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();

        // 6. Accumulate over inner indices
        let mut acc = A::zero();
        for inner_linear in 0..inner_size {
            let inner_multi = linear_to_multi(inner_linear, &inner_ranges);
            for (&idx, &val) in inner_vec.iter().zip(inner_multi.iter()) {
                idx_values.insert(idx, val);
            }

            // 7. Compute input position and accumulate
            let in_pos = compute_input_position(ix, &idx_values, tensor.shape());
            acc = acc.add(A::from_scalar(tensor.get(in_pos)));
        }

        out_data[out_linear] = acc.to_scalar();
    }

    if out_shape.is_empty() {
        Tensor::from_data(&out_data, &[1])
    } else {
        Tensor::from_data(&out_data, &out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::backend::Cpu;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_einsum_matmul() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        // Without optimization
        let c1 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c1.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

        // With optimization
        ein.optimize_greedy();
        let c2 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c2.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_tropical() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        let c = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_einsum_chain() {
        // A[i,j] × B[j,k] × C[k,l] → D[i,l]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let mut ein = Einsum::new(
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![0, 3],
            sizes,
        );

        ein.optimize_greedy();
        let d = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);

        assert_eq!(d.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_trace() {
        // Trace: A[i,i] -> scalar (sum of diagonal)
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // trace = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_einsum_diagonal() {
        // Diagonal: A[i,i] -> B[i] (extract diagonal)
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![0], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // diagonal = [1, 4]
        assert_eq!(result.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_einsum_sum_axis() {
        // Reduction: A[i,j] -> B[i] (sum over j)
        // Column-major: data [1,2,3,4] for shape [2,2] represents:
        // [[1, 3],
        //  [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // sum over j: [1+3, 2+4] = [4, 6]
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_einsum_sum_all() {
        // Sum all: A[i,j] -> scalar
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // sum = 1 + 2 + 3 + 4 = 10
        assert_eq!(result.to_vec()[0], 10.0);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_trace_tropical() {
        // Trace with max-plus algebra: A[i,i] -> scalar
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        let result = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a]);
        // tropical trace = max(1, 4) = 4
        assert_eq!(result.to_vec()[0], 4.0);
    }

    // Tests for helper functions

    #[test]
    fn test_linear_to_multi_empty_shape() {
        // Empty shape should return empty multi-index
        let result = linear_to_multi(0, &[]);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_linear_to_multi_1d() {
        // 1D array: linear index equals multi-index
        assert_eq!(linear_to_multi(0, &[5]), vec![0]);
        assert_eq!(linear_to_multi(3, &[5]), vec![3]);
        assert_eq!(linear_to_multi(4, &[5]), vec![4]);
    }

    #[test]
    fn test_linear_to_multi_2d() {
        // 2D array with shape [2, 3] (column-major)
        // Linear 0 -> (0, 0)
        // Linear 1 -> (1, 0)
        // Linear 2 -> (0, 1)
        // Linear 3 -> (1, 1)
        // Linear 4 -> (0, 2)
        // Linear 5 -> (1, 2)
        assert_eq!(linear_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3]), vec![1, 0]);
        assert_eq!(linear_to_multi(2, &[2, 3]), vec![0, 1]);
        assert_eq!(linear_to_multi(3, &[2, 3]), vec![1, 1]);
        assert_eq!(linear_to_multi(4, &[2, 3]), vec![0, 2]);
        assert_eq!(linear_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_linear_to_multi_3d() {
        // 3D array with shape [2, 3, 4] (column-major)
        // Strides: [1, 2, 6]
        // Linear 0 -> (0, 0, 0)
        // Linear 1 -> (1, 0, 0)
        // Linear 2 -> (0, 1, 0)
        // Linear 6 -> (0, 0, 1)
        // Linear 7 -> (1, 0, 1)
        assert_eq!(linear_to_multi(0, &[2, 3, 4]), vec![0, 0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3, 4]), vec![1, 0, 0]);
        assert_eq!(linear_to_multi(2, &[2, 3, 4]), vec![0, 1, 0]);
        assert_eq!(linear_to_multi(6, &[2, 3, 4]), vec![0, 0, 1]);
        assert_eq!(linear_to_multi(7, &[2, 3, 4]), vec![1, 0, 1]);
        // Last element: linear 23 -> (1, 2, 3)
        assert_eq!(linear_to_multi(23, &[2, 3, 4]), vec![1, 2, 3]);
    }

    #[test]
    fn test_compute_input_position_1d() {
        // 1D tensor with index label 0
        let ix = vec![0];
        let shape = vec![5];

        let mut idx_values = HashMap::new();
        idx_values.insert(0, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        idx_values.insert(0, 3);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 3);
    }

    #[test]
    fn test_compute_input_position_2d() {
        // 2D tensor with shape [2, 3], index labels (0, 1)
        // Column-major: position = i + j * 2
        let ix = vec![0, 1];
        let shape = vec![2, 3];

        let mut idx_values = HashMap::new();

        // (0, 0) -> position 0
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        // (1, 0) -> position 1
        idx_values.insert(0, 1);
        idx_values.insert(1, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 1);

        // (0, 1) -> position 2
        idx_values.insert(0, 0);
        idx_values.insert(1, 1);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 2);

        // (1, 2) -> position 1 + 2*2 = 5
        idx_values.insert(0, 1);
        idx_values.insert(1, 2);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 5);
    }

    #[test]
    fn test_compute_input_position_3d() {
        // 3D tensor with shape [2, 3, 4], index labels (0, 1, 2)
        // Column-major: position = i + j * 2 + k * 6
        let ix = vec![0, 1, 2];
        let shape = vec![2, 3, 4];

        let mut idx_values = HashMap::new();

        // (0, 0, 0) -> position 0
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        // (1, 0, 0) -> position 1
        idx_values.insert(0, 1);
        idx_values.insert(1, 0);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 1);

        // (0, 1, 0) -> position 2
        idx_values.insert(0, 0);
        idx_values.insert(1, 1);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 2);

        // (0, 0, 1) -> position 6
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        idx_values.insert(2, 1);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 6);

        // (1, 2, 3) -> position 1 + 2*2 + 3*6 = 1 + 4 + 18 = 23
        idx_values.insert(0, 1);
        idx_values.insert(1, 2);
        idx_values.insert(2, 3);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 23);
    }

    #[test]
    fn test_linear_to_multi_roundtrip() {
        // Verify that linear_to_multi and compute_input_position are consistent
        let shape = vec![2, 3, 4];
        let ix: Vec<usize> = (0..shape.len()).collect();
        let total_size: usize = shape.iter().product();

        for linear in 0..total_size {
            let multi = linear_to_multi(linear, &shape);

            // Build idx_values from multi
            let mut idx_values = HashMap::new();
            for (dim, &val) in multi.iter().enumerate() {
                idx_values.insert(dim, val);
            }

            let computed_pos = compute_input_position(&ix, &idx_values, &shape);
            assert_eq!(
                computed_pos, linear,
                "Roundtrip failed for linear={}, multi={:?}",
                linear, multi
            );
        }
    }

    // ========================================================================
    // Tests for execute_unary_naive
    // ========================================================================

    #[test]
    fn test_unary_naive_transpose() {
        // Transpose: A[i,j] -> B[j,i]
        // Input matrix (column-major): [[1, 3], [2, 4]]
        // data = [1, 2, 3, 4], shape = [2, 2]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![1, 0]; // B[j,i]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // After transpose: [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_unary_naive_trace() {
        // Trace: A[i,i] -> scalar (sum of diagonal)
        // Matrix (column-major): [[1, 3], [2, 4]]
        // data = [1, 2, 3, 4], shape = [2, 2]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i] - repeated index means diagonal
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // trace = A[0,0] + A[1,1] = 1 + 4 = 5
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_unary_naive_diagonal() {
        // Diagonal extraction: A[i,i] -> B[i]
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i] - repeated index
        let iy = vec![0]; // output B[i]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // diagonal = [A[0,0], A[1,1]] = [1, 4]
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_unary_naive_sum_axis() {
        // Sum over axis: A[i,j] -> B[i] (sum over j)
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![0]; // B[i] - j is summed out

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // sum over j: B[i] = sum_j A[i,j]
        // B[0] = A[0,0] + A[0,1] = 1 + 3 = 4
        // B[1] = A[1,0] + A[1,1] = 2 + 4 = 6
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_unary_naive_sum_all() {
        // Sum all: A[i,j] -> scalar
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // sum all = 1 + 2 + 3 + 4 = 10
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.to_vec()[0], 10.0);
    }

    #[test]
    fn test_unary_naive_partial_trace() {
        // Partial trace: A[i,j,i] -> B[j] (trace over i, keeping j)
        // 3D tensor with shape [2, 3, 2]
        // This is like having a batch of 2x2 matrices and taking the trace of each
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a = Tensor::<f32, Cpu>::from_data(&data, &[2, 3, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let ix = vec![0, 1, 0]; // A[i,j,i] - i is repeated at positions 0 and 2
        let iy = vec![1]; // B[j] - output keeps only j

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // For each j, we sum A[0,j,0] + A[1,j,1]
        // Column-major layout: data[i + j*2 + k*6]
        // j=0: A[0,0,0] + A[1,0,1] = data[0] + data[1+0*2+1*6] = data[0] + data[7] = 1 + 8 = 9
        // j=1: A[0,1,0] + A[1,1,1] = data[0+1*2+0*6] + data[1+1*2+1*6] = data[2] + data[9] = 3 + 10 = 13
        // j=2: A[0,2,0] + A[1,2,1] = data[0+2*2+0*6] + data[1+2*2+1*6] = data[4] + data[11] = 5 + 12 = 17
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![9.0, 13.0, 17.0]);
    }

    #[test]
    fn test_unary_naive_3d_transpose() {
        // 3D permutation: A[i,j,k] -> B[k,i,j]
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let a = Tensor::<f32, Cpu>::from_data(&data, &[2, 2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let ix = vec![0, 1, 2]; // A[i,j,k]
        let iy = vec![2, 0, 1]; // B[k,i,j]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        assert_eq!(result.shape(), &[2, 2, 2]);

        // Verify by checking specific elements
        // B[k,i,j] = A[i,j,k]
        // Build expected output manually
        let mut expected = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    // A[i,j,k] at position i + j*2 + k*4 in column-major
                    let a_pos = i + j * 2 + k * 4;
                    // B[k,i,j] at position k + i*2 + j*4 in column-major
                    let b_pos = k + i * 2 + j * 4;
                    expected[b_pos] = data[a_pos];
                }
            }
        }
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_unary_naive_identity() {
        // Identity: A[i,j] -> B[i,j] (no change)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![0, 1]; // B[i,j]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), a.to_vec());
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_unary_naive_trace_tropical() {
        // Trace with max-plus algebra: A[i,i] -> scalar
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i]
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<MaxPlus<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // tropical trace = max(A[0,0], A[1,1]) = max(1, 4) = 4
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.to_vec()[0], 4.0);
    }
}
