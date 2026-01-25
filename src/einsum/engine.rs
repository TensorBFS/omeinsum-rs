//! Einsum execution engine with omeco integration.

use std::collections::HashMap;

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
        B: Backend,
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
    pub fn execute_with_argmax<A, T, B>(
        &self,
        tensors: &[&Tensor<T, B>],
    ) -> (Tensor<T, B>, Vec<Tensor<u32, B>>)
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend,
    {
        // TODO: Implement proper argmax tracking through tree execution
        let result = self.execute::<A, T, B>(tensors);
        (result, vec![])
    }

    /// Execute an optimized contraction tree.
    fn execute_tree<A, T, B>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[&Tensor<T, B>],
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar,
        B: Backend,
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
        B: Backend,
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

    /// Execute unary operation (trace/reduction).
    fn execute_unary<A, T, B>(&self, tensor: &Tensor<T, B>, _ix: &[usize]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T>,
        T: Scalar,
        B: Backend,
    {
        // For now, just return the tensor
        // TODO: Implement trace and reduction
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
        if final_set.contains(&i) || !ib_set.contains(&i) {
            if !output.contains(&i) {
                output.push(i);
            }
        }
    }

    for &i in ib {
        if final_set.contains(&i) || !ia_set.contains(&i) {
            if !output.contains(&i) {
                output.push(i);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MaxPlus, Standard};
    use crate::backend::Cpu;

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
}
