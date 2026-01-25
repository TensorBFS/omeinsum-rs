//! Basic einsum examples demonstrating standard and tropical operations.

use omeinsum::algebra::{MaxPlus, MinPlus, Standard};
use omeinsum::backend::Cpu;
use omeinsum::einsum::Einsum;
use omeinsum::tensor::Tensor;
use std::collections::HashMap;

fn main() {
    println!("=== OMEinsum Basic Examples ===\n");

    // Example 1: Standard matrix multiplication
    println!("1. Standard Matrix Multiplication");
    println!("   C[i,k] = Σ_j A[i,j] × B[j,k]\n");

    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = a.gemm::<Standard<f32>>(&b);
    println!("   A = [[1, 2], [3, 4]]");
    println!("   B = [[1, 2], [3, 4]]");
    println!("   C = {:?}\n", c.to_vec());

    // Example 2: Tropical (max-plus) matrix multiplication
    println!("2. Tropical (MaxPlus) Matrix Multiplication");
    println!("   C[i,k] = max_j (A[i,j] + B[j,k])\n");

    let c_tropical = a.gemm::<MaxPlus<f32>>(&b);
    println!("   Same A, B as above");
    println!("   C (tropical) = {:?}\n", c_tropical.to_vec());

    // Example 3: Min-plus for shortest paths
    println!("3. MinPlus Matrix Multiplication (Shortest Paths)");
    println!("   C[i,k] = min_j (A[i,j] + B[j,k])\n");

    let c_minplus = a.gemm::<MinPlus<f32>>(&b);
    println!("   C (minplus) = {:?}\n", c_minplus.to_vec());

    // Example 4: Einsum with optimization
    println!("4. Einsum with Contraction Order Optimization");
    println!("   A[i,j] × B[j,k] → C[i,k]\n");

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

    // Without optimization
    let c1 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
    println!("   Without optimization: {:?}", c1.to_vec());

    // With optimization
    ein.optimize_greedy();
    let c2 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
    println!("   With greedy optimization: {:?}", c2.to_vec());
    println!("   Optimized: {}\n", ein.is_optimized());

    // Example 5: Tensor permutation (transpose)
    println!("5. Tensor Permutation (Zero-Copy)");
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("   Original shape: {:?}, contiguous: {}", t.shape(), t.is_contiguous());

    let t_transposed = t.permute(&[1, 0]);
    println!(
        "   Transposed shape: {:?}, contiguous: {}",
        t_transposed.shape(),
        t_transposed.is_contiguous()
    );

    let t_contiguous = t_transposed.contiguous();
    println!(
        "   After contiguous(): {:?}, contiguous: {}",
        t_contiguous.shape(),
        t_contiguous.is_contiguous()
    );
    println!("   Data: {:?}\n", t_contiguous.to_vec());

    println!("=== Done ===");
}
