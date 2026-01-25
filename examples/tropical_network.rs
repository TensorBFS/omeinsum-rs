//! Tropical tensor network contraction example.
//!
//! Demonstrates contracting a chain of tensors using tropical algebra,
//! useful for MPE (Most Probable Explanation) inference in graphical models.

use omeinsum::algebra::MaxPlus;
use omeinsum::backend::Cpu;
use omeinsum::einsum::Einsum;
use omeinsum::tensor::Tensor;
use std::collections::HashMap;

fn main() {
    println!("=== Tropical Tensor Network Contraction ===\n");

    // Create a chain of factor tensors (simulating a factor graph)
    // Factor graph: X0 -- f01 -- X1 -- f12 -- X2 -- f23 -- X3
    //
    // Each factor f_{i,i+1} represents log-probabilities
    // Tropical contraction computes: max_{x1,x2} [f01(x0,x1) + f12(x1,x2) + f23(x2,x3)]
    // which gives the max log-probability for each (x0, x3) pair

    let n = 3; // Each variable has 3 states

    // Factor tensors (random log-probabilities)
    let f01 = Tensor::<f32, Cpu>::from_data(
        &[
            0.1, 0.5, 0.2, // x0=0: log-prob for x1=0,1,2
            0.3, 0.1, 0.4, // x0=1
            0.2, 0.3, 0.1, // x0=2
        ],
        &[n, n],
    );

    let f12 = Tensor::<f32, Cpu>::from_data(
        &[
            0.2, 0.1, 0.3, // x1=0
            0.4, 0.2, 0.1, // x1=1
            0.1, 0.5, 0.2, // x1=2
        ],
        &[n, n],
    );

    let f23 = Tensor::<f32, Cpu>::from_data(
        &[
            0.3, 0.2, 0.1, // x2=0
            0.1, 0.4, 0.3, // x2=1
            0.2, 0.1, 0.5, // x2=2
        ],
        &[n, n],
    );

    println!("Factor graph: X0 -- f01 -- X1 -- f12 -- X2 -- f23 -- X3");
    println!("Each factor is {}x{}\n", n, n);

    // Contract: f01[x0,x1] × f12[x1,x2] × f23[x2,x3] → result[x0,x3]
    // Using tropical max-plus algebra:
    // result[x0,x3] = max_{x1,x2} [f01[x0,x1] + f12[x1,x2] + f23[x2,x3]]

    let sizes: HashMap<usize, usize> = [(0, n), (1, n), (2, n), (3, n)].into();

    let mut ein = Einsum::new(
        vec![
            vec![0, 1], // f01[x0, x1]
            vec![1, 2], // f12[x1, x2]
            vec![2, 3], // f23[x2, x3]
        ],
        vec![0, 3], // result[x0, x3]
        sizes,
    );

    // Optimize contraction order
    ein.optimize_greedy();
    println!("Contraction optimized: {}", ein.is_optimized());

    // Execute with tropical algebra
    let result = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&f01, &f12, &f23]);

    println!("\nMax log-probability for each (x0, x3) pair:");
    println!("Shape: {:?}", result.shape());

    let data = result.to_vec();
    for x0 in 0..n {
        for x3 in 0..n {
            println!("  (x0={}, x3={}) -> {:.3}", x0, x3, data[x0 * n + x3]);
        }
    }

    // Find the global maximum
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_idx = data.iter().position(|&x| x == max_val).unwrap();
    let best_x0 = max_idx / n;
    let best_x3 = max_idx % n;

    println!(
        "\nGlobal max: {:.3} at (x0={}, x3={})",
        max_val, best_x0, best_x3
    );

    println!("\n=== Done ===");
}
