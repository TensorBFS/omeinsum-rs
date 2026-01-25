//! CPU backend implementation.

use super::traits::{Backend, Storage};
use crate::algebra::{Algebra, Scalar};

/// CPU backend using Vec storage.
#[derive(Clone, Debug, Default)]
pub struct Cpu;

impl<T: Scalar> Storage<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn get(&self, index: usize) -> T {
        self[index]
    }

    #[inline]
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }

    #[inline]
    fn to_vec(&self) -> Vec<T> {
        self.clone()
    }

    #[inline]
    fn from_slice(data: &[T]) -> Self {
        data.to_vec()
    }

    #[inline]
    fn zeros(len: usize) -> Self {
        vec![T::default(); len]
    }
}

impl Backend for Cpu {
    type Storage<T: Scalar> = Vec<T>;

    fn name() -> &'static str {
        "cpu"
    }

    fn synchronize(&self) {
        // No-op for CPU
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
        vec![T::default(); len]
    }

    fn from_slice<T: Scalar>(&self, data: &[T]) -> Vec<T> {
        data.to_vec()
    }

    fn copy_strided<T: Scalar>(
        &self,
        src: &Vec<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Vec<T> {
        let numel: usize = shape.iter().product();
        let mut dst = vec![T::default(); numel];

        // Iterate over all indices and copy
        let mut indices = vec![0usize; shape.len()];
        for flat_idx in 0..numel {
            // Compute source offset using strides
            let src_offset: usize = offset
                + indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>();

            dst[flat_idx] = src[src_offset];

            // Increment indices (row-major order)
            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }

        dst
    }

    fn gemm<A: Algebra>(
        &self,
        a: &Vec<A::Scalar>,
        m: usize,
        k: usize,
        b: &Vec<A::Scalar>,
        n: usize,
    ) -> Vec<A::Scalar> {
        // Try to use optimized tropical-gemm if available
        #[cfg(feature = "tropical-kernels")]
        {
            if let Some(result) = try_tropical_gemm::<A>(a, m, k, b, n) {
                return result;
            }
        }

        // Fallback to generic loop implementation
        generic_gemm::<A>(a, m, k, b, n)
    }

    fn gemm_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Vec<A::Scalar>,
        m: usize,
        k: usize,
        b: &Vec<A::Scalar>,
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        // Try to use optimized tropical-gemm if available
        #[cfg(feature = "tropical-kernels")]
        {
            if let Some(result) = try_tropical_gemm_with_argmax::<A>(a, m, k, b, n) {
                return result;
            }
        }

        // Fallback to generic loop implementation
        generic_gemm_with_argmax::<A>(a, m, k, b, n)
    }

    fn gemm_backward_a<A: Algebra>(
        &self,
        grad_c: &Vec<A::Scalar>,
        argmax: &Vec<u32>,
        _b: &Vec<A::Scalar>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<A::Scalar> {
        let mut grad_a = vec![A::Scalar::default(); m * k];

        // For tropical: grad_a[i, argmax[i,j]] += grad_c[i,j]
        // For standard: grad_a = grad_c @ b.T
        if A::needs_argmax() {
            for i in 0..m {
                for j in 0..n {
                    let idx = argmax[i * n + j] as usize;
                    // Accumulate gradient using AddAssign
                    grad_a[i * k + idx] += grad_c[i * n + j];
                }
            }
        }

        grad_a
    }

    fn gemm_backward_b<A: Algebra>(
        &self,
        grad_c: &Vec<A::Scalar>,
        argmax: &Vec<u32>,
        _a: &Vec<A::Scalar>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<A::Scalar> {
        let mut grad_b = vec![A::Scalar::default(); k * n];

        if A::needs_argmax() {
            for i in 0..m {
                for j in 0..n {
                    let idx = argmax[i * n + j] as usize;
                    // Accumulate gradient using AddAssign
                    grad_b[idx * n + j] += grad_c[i * n + j];
                }
            }
        }

        grad_b
    }
}

/// Generic GEMM using semiring operations.
fn generic_gemm<A: Algebra>(a: &[A::Scalar], m: usize, k: usize, b: &[A::Scalar], n: usize) -> Vec<A::Scalar> {
    let mut c = vec![A::zero().to_scalar(); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = A::zero();
            for kk in 0..k {
                let a_val = A::from_scalar(a[i * k + kk]);
                let b_val = A::from_scalar(b[kk * n + j]);
                let prod = a_val.mul(b_val);
                acc = acc.add(prod);
            }
            c[i * n + j] = acc.to_scalar();
        }
    }

    c
}

/// Generic GEMM with argmax tracking.
fn generic_gemm_with_argmax<A: Algebra<Index = u32>>(
    a: &[A::Scalar],
    m: usize,
    k: usize,
    b: &[A::Scalar],
    n: usize,
) -> (Vec<A::Scalar>, Vec<u32>) {
    let mut c = vec![A::zero().to_scalar(); m * n];
    let mut argmax = vec![0u32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = A::zero();
            let mut best_k = 0u32;

            for kk in 0..k {
                let a_val = A::from_scalar(a[i * k + kk]);
                let b_val = A::from_scalar(b[kk * n + j]);
                let prod = a_val.mul(b_val);
                let (new_acc, winner) = acc.add_with_argmax(best_k, prod, kk as u32);
                acc = new_acc;
                best_k = winner;
            }

            c[i * n + j] = acc.to_scalar();
            argmax[i * n + j] = best_k;
        }
    }

    (c, argmax)
}

// Optional: Use tropical-gemm for optimized kernels
#[cfg(feature = "tropical-kernels")]
fn try_tropical_gemm<A: Algebra>(
    _a: &[A::Scalar],
    _m: usize,
    _k: usize,
    _b: &[A::Scalar],
    _n: usize,
) -> Option<Vec<A::Scalar>> {
    // TODO: Dispatch to tropical-gemm based on A type
    // For now, fall back to generic
    None
}

#[cfg(feature = "tropical-kernels")]
fn try_tropical_gemm_with_argmax<A: Algebra>(
    _a: &[A::Scalar],
    _m: usize,
    _k: usize,
    _b: &[A::Scalar],
    _n: usize,
) -> Option<(Vec<A::Scalar>, Vec<u32>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_cpu_gemm_standard() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2

        let c = cpu.gemm::<Standard<f32>>(&a, 2, 2, &b, 2);

        // [1 2] × [1 2] = [1*1+2*3  1*2+2*4] = [7  10]
        // [3 4]   [3 4]   [3*1+4*3  3*2+4*4]   [15 22]
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_cpu_gemm_maxplus() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2

        let c = cpu.gemm::<MaxPlus<f32>>(&a, 2, 2, &b, 2);

        // MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
        // C[0,0] = max(1+1, 2+3) = max(2, 5) = 5
        // C[0,1] = max(1+2, 2+4) = max(3, 6) = 6
        // C[1,0] = max(3+1, 4+3) = max(4, 7) = 7
        // C[1,1] = max(3+2, 4+4) = max(5, 8) = 8
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_cpu_gemm_with_argmax() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let (c, argmax) = cpu.gemm_with_argmax::<MaxPlus<f32>>(&a, 2, 2, &b, 2);

        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
        // All winners should be k=1 (second column of A, second row of B)
        assert_eq!(argmax, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_copy_strided() {
        let cpu = Cpu;
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 row-major

        // Transpose: shape [3, 2], strides [1, 3]
        let dst = cpu.copy_strided(&src, &[3, 2], &[1, 3], 0);

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
