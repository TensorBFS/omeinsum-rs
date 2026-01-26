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
            Some(tree) => self.execute_tree::<A, T, B>(tree, tensors),
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
            return self.execute_unary::<A, T, B>(tensors[0], &self.ixs[0]);
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
            return self.execute_unary::<A, T, B>(tensors[0], &self.ixs[0]);
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

    /// Execute unary operation (trace/diagonal/reduction).
    ///
    /// Handles:
    /// - Trace: `A[i,i] -> scalar` - sum of diagonal elements
    /// - Diagonal: `A[i,i] -> B[i]` - extract diagonal
    /// - Reduction: `A[i,j] -> B[i]` - sum over j
    fn execute_unary<A, T, B>(&self, tensor: &Tensor<T, B>, ix: &[usize]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T>,
        T: Scalar,
        B: Backend + Default,
    {
        // Find repeated indices (for trace/diagonal)
        let mut index_counts: HashMap<usize, usize> = HashMap::new();
        for &i in ix {
            *index_counts.entry(i).or_insert(0) += 1;
        }

        let repeated: Vec<usize> = index_counts
            .iter()
            .filter(|(_, &count)| count > 1)
            .map(|(&idx, _)| idx)
            .collect();

        // Find indices to sum over (in input but not in output)
        let output_set: HashSet<_> = self.iy.iter().copied().collect();

        // Handle repeated indices (trace/diagonal)
        if !repeated.is_empty() && tensor.ndim() == 2 {
            let diag = tensor.diagonal();
            if self.iy.is_empty() {
                // Trace: sum the diagonal
                let sum = diag.sum::<A>();
                return Tensor::from_data(&[sum], &[1]);
            } else {
                // Diagonal extraction
                return diag;
            }
        }

        // Handle reduction (sum over axes not in output)
        // Find which axis positions need to be summed
        let mut axes_to_sum: Vec<usize> = Vec::new();
        for (axis, &idx) in ix.iter().enumerate() {
            if !output_set.contains(&idx) {
                axes_to_sum.push(axis);
            }
        }

        if !axes_to_sum.is_empty() {
            // Sum over axes from highest to lowest to maintain correct indexing
            axes_to_sum.sort();
            axes_to_sum.reverse();

            let mut result = tensor.clone();
            for axis in axes_to_sum {
                result = result.sum_axis::<A>(axis);
            }
            return result;
        }

        // No operation needed, just return the tensor
        tensor.clone()
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
}
