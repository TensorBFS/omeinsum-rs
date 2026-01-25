//! Tensor view types for borrowing.

use super::Tensor;
use crate::algebra::Scalar;
use crate::backend::Backend;

/// A borrowed view into a tensor.
///
/// This is useful for passing tensors to functions without ownership transfer.
pub struct TensorView<'a, T: Scalar, B: Backend> {
    tensor: &'a Tensor<T, B>,
}

impl<'a, T: Scalar, B: Backend> TensorView<'a, T, B> {
    /// Create a view from a tensor reference.
    pub fn new(tensor: &'a Tensor<T, B>) -> Self {
        Self { tensor }
    }

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        self.tensor.strides()
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    /// Get total number of elements.
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// Check if contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    /// Get the underlying tensor.
    pub fn as_tensor(&self) -> &Tensor<T, B> {
        self.tensor
    }
}

impl<'a, T: Scalar, B: Backend> From<&'a Tensor<T, B>> for TensorView<'a, T, B> {
    fn from(tensor: &'a Tensor<T, B>) -> Self {
        Self::new(tensor)
    }
}
