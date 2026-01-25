//! Algebraic structures for tensor operations.
//!
//! This module defines the [`Semiring`] trait and implementations for:
//! - [`Standard<T>`]: Standard arithmetic `(+, ×)`
//! - [`MaxPlus<T>`]: Tropical max-plus `(max, +)`
//! - [`MinPlus<T>`]: Tropical min-plus `(min, +)`
//! - [`MaxMul<T>`]: Tropical max-mul `(max, ×)`

mod semiring;
mod standard;
mod tropical;

pub use semiring::{Algebra, Semiring};
pub use standard::Standard;
pub use tropical::{MaxMul, MaxPlus, MinPlus};

/// Marker trait for scalar types that can be used in tensors.
pub trait Scalar:
    Copy + Clone + Send + Sync + Default + std::fmt::Debug + 'static + bytemuck::Pod
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
