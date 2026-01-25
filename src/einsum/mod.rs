//! Einstein summation engine with contraction order optimization.
//!
//! This module provides the [`Einsum`] type for specifying and executing
//! tensor network contractions, with optional optimization via omeco.

mod backward;
mod builder;
mod engine;

pub use builder::EinBuilder;
pub use engine::Einsum;

use crate::algebra::{Algebra, Scalar};
use crate::backend::Backend;
use crate::tensor::Tensor;

/// One-shot einsum with automatic optimization.
///
/// # Arguments
///
/// * `tensors` - Input tensors
/// * `ixs` - Index labels for each input tensor
/// * `iy` - Output index labels
///
/// # Example
///
/// ```rust,ignore
/// use omeinsum::{einsum, Tensor, Cpu};
/// use omeinsum::algebra::MaxPlus;
///
/// let a = Tensor::<f32, Cpu>::from_data(&data_a, &[2, 3]);
/// let b = Tensor::<f32, Cpu>::from_data(&data_b, &[3, 4]);
///
/// // C[i,k] = max_j (A[i,j] + B[j,k])
/// let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
/// ```
pub fn einsum<A, T, B>(tensors: &[&Tensor<T, B>], ixs: &[&[usize]], iy: &[usize]) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend + Default,
{
    let size_dict = infer_size_dict(tensors, ixs);
    let ixs_owned: Vec<Vec<usize>> = ixs.iter().map(|ix| ix.to_vec()).collect();

    let mut ein = Einsum::new(ixs_owned, iy.to_vec(), size_dict);
    ein.optimize_greedy();
    ein.execute::<A, T, B>(tensors)
}

/// Einsum with gradient computation.
///
/// Returns `(result, gradient_fn)` where `gradient_fn` can be called
/// with the output gradient to compute input gradients.
pub fn einsum_with_grad<A, T, B>(
    tensors: &[&Tensor<T, B>],
    ixs: &[&[usize]],
    iy: &[usize],
) -> (Tensor<T, B>, EinsumGradient<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend + Default,
{
    let size_dict = infer_size_dict(tensors, ixs);
    let ixs_owned: Vec<Vec<usize>> = ixs.iter().map(|ix| ix.to_vec()).collect();

    let mut ein = Einsum::new(ixs_owned.clone(), iy.to_vec(), size_dict);
    ein.optimize_greedy();

    let (result, argmax_cache) = ein.execute_with_argmax::<A, T, B>(tensors);

    let gradient = EinsumGradient {
        ixs: ixs_owned,
        iy: iy.to_vec(),
        argmax_cache,
        _phantom: std::marker::PhantomData,
    };

    (result, gradient)
}

/// Gradient computation helper for einsum.
pub struct EinsumGradient<T: Scalar, B: Backend> {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
    argmax_cache: Vec<Tensor<u32, B>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, B: Backend> EinsumGradient<T, B> {
    /// Compute gradients for all inputs given the output gradient.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the einsum output
    /// * `inputs` - Original input tensors (same as passed to forward)
    ///
    /// # Returns
    ///
    /// Vector of gradients, one for each input tensor.
    pub fn backward<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        grad_output: &Tensor<T, B>,
        inputs: &[&Tensor<T, B>],
    ) -> Vec<Tensor<T, B>> {
        assert_eq!(
            inputs.len(),
            self.ixs.len(),
            "Number of inputs {} doesn't match stored indices {}",
            inputs.len(),
            self.ixs.len()
        );

        // Handle single input case: gradient passes through unchanged
        if inputs.len() == 1 {
            return vec![grad_output.clone()];
        }

        // For a single binary contraction (2 inputs), we can directly compute gradients
        if inputs.len() == 2 {
            let argmax = if A::needs_argmax() && !self.argmax_cache.is_empty() {
                Some(&self.argmax_cache[0])
            } else {
                None
            };

            let (grad_a, grad_b) = backward::contract_binary_backward::<A, T, B>(
                grad_output,
                inputs[0],
                inputs[1],
                argmax,
                &self.ixs[0],
                &self.ixs[1],
                &self.iy,
            );

            return vec![grad_a, grad_b];
        }

        // For more complex contractions with >2 tensors, we need to reverse through
        // the contraction tree. This requires storing intermediate results from forward pass.
        // For now, implement the simple case.
        //
        // TODO: Implement full backward pass for multi-tensor contractions
        // This would require:
        // 1. Storing intermediate results during forward pass
        // 2. Reversing through the contraction tree
        // 3. Accumulating gradients for each input
        unimplemented!(
            "Backward pass for {} inputs not yet implemented. \
             Currently only 2-input contractions are supported.",
            inputs.len()
        )
    }
}

/// Infer size dictionary from tensors and their index labels.
fn infer_size_dict<T: Scalar, B: Backend>(
    tensors: &[&Tensor<T, B>],
    ixs: &[&[usize]],
) -> std::collections::HashMap<usize, usize> {
    let mut size_dict = std::collections::HashMap::new();

    for (tensor, ix) in tensors.iter().zip(ixs.iter()) {
        assert_eq!(
            tensor.ndim(),
            ix.len(),
            "Index count {} doesn't match tensor ndim {}",
            ix.len(),
            tensor.ndim()
        );

        for (dim, &label) in ix.iter().enumerate() {
            let size = tensor.shape()[dim];
            if let Some(&existing) = size_dict.get(&label) {
                assert_eq!(
                    existing, size,
                    "Inconsistent size for index {}: {} vs {}",
                    label, existing, size
                );
            } else {
                size_dict.insert(label, size);
            }
        }
    }

    size_dict
}
