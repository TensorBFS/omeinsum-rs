//! Einstein summation engine with contraction order optimization.
//!
//! This module provides the [`Einsum`] type for specifying and executing
//! tensor network contractions, with optional optimization via omeco.

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
    pub fn backward<A: Algebra<Scalar = T>>(
        &self,
        _grad_output: &Tensor<T, B>,
        _inputs: &[&Tensor<T, B>],
    ) -> Vec<Tensor<T, B>> {
        // TODO: Implement backward pass using argmax cache
        unimplemented!("Backward pass not yet implemented")
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
