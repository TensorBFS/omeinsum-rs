#!/bin/bash
# scripts/test_features.sh
# Tests all feature combinations to ensure lean default builds work correctly
set -e

echo "Testing: no features (standard only)"
cargo build --no-default-features
cargo test --no-default-features

echo "Testing: tropical feature"
cargo build --no-default-features --features tropical
cargo test --no-default-features --features tropical

echo "Testing: tropical-kernels feature"
cargo build --no-default-features --features tropical-kernels
cargo test --no-default-features --features tropical-kernels

echo "Testing: full feature (tropical-kernels + parallel)"
cargo build --features full
cargo test --features full

# Note: CUDA feature is not tested here as it requires the cuda backend module
# which is not yet implemented. Use --features cuda when CUDA support is added.

echo "All feature combinations pass!"
