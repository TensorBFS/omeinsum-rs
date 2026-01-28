# CUDA Setup Guide for omeinsum-rs

This guide explains how to set up CUDA support for running GPU-accelerated tensor contractions.

## Requirements

1. **NVIDIA GPU** with CUDA compute capability 3.5 or higher
2. **NVIDIA Driver** version 470+ (for CUDA 11) or 525+ (for CUDA 12)
3. **CUDA Toolkit** version 11.0 or higher (CUDA 12 recommended)
4. **cuTENSOR Library** version 2.0 or higher (REQUIRED - version 1.x will NOT work)

## Installation Steps

### 1. Install NVIDIA Driver

Check if you have an NVIDIA GPU and driver installed:
```bash
nvidia-smi
```

If not installed, follow [NVIDIA's driver installation guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

### 2. Install CUDA Toolkit

Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

Verify installation:
```bash
nvcc --version
```

If `nvcc` is not found, add CUDA to your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Install cuTENSOR

cuTENSOR 2.0+ is **required**. The 1.x API is incompatible.

#### Option A: Conda (Recommended)
```bash
conda install -c nvidia cutensor-cu12  # For CUDA 12
# or
conda install -c nvidia cutensor-cu11  # For CUDA 11
```

#### Option B: pip
```bash
pip install cutensor-cu12
```

Note: pip-installed cuTENSOR may require creating a symlink:
```bash
# Find the installation path
python -c "import cutensor; print(cutensor.__path__)"

# Create symlink (adjust path as needed)
ln -s libcutensor.so.2 $CUTENSOR_PATH/libcutensor.so
```

#### Option C: Direct Download
Download from [NVIDIA cuTENSOR Downloads](https://developer.nvidia.com/cutensor-downloads).

### 4. Set Environment Variables

```bash
# CUDA paths
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# cuTENSOR path (adjust based on installation method)
export CUTENSOR_PATH=/path/to/cutensor/lib
export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

## Building with CUDA Support

```bash
cargo build --features cuda
cargo test --features cuda
```

## Troubleshooting

### "nvcc not found"
Add CUDA bin directory to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### "libcutensor.so not found"
Set CUTENSOR_PATH to the directory containing the library:
```bash
export CUTENSOR_PATH=/path/to/cutensor/lib
export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH
```

### "libcublasLt.so.13 not found" (CUDA version mismatch)
This occurs when the runtime expects a different CUDA version than installed.
- Ensure your CUDA toolkit version matches what cuTENSOR was compiled against
- If using Julia's cuTENSOR artifacts, you may need CUDA 13 libraries

### Using Julia's CUDA Artifacts
If you have Julia with CUDA.jl installed, you can use its artifacts:
```bash
# Find Julia's CUDA libraries
find ~/.julia/artifacts -name "libcublas*.so.*" | head -5

# Find cuTENSOR
find ~/.julia/artifacts -name "libcutensor.so.*" | head -5

# Set paths (example)
export CUTENSOR_PATH=~/.julia/artifacts/<hash>/lib
export LD_LIBRARY_PATH=~/.julia/artifacts/<cuda-hash>/lib:$CUTENSOR_PATH:$LD_LIBRARY_PATH
```

## Known Limitations

1. **Broadcast via `einsum()` size inference not supported**: Operations like `i -> ij` that add new dimensions are not handled by the convenience `einsum()` helper on GPU. These broadcast/repeat patterns are supported on GPU when using `Einsum::new` with explicit size specification.

2. **Multi-tensor operations**: Some complex multi-tensor einsum operations may produce different results between CPU and GPU due to different contraction orderings.

3. **Complex numbers**: Complex number support via `CudaComplex` wrapper, not native Rust complex types.

## Verifying Setup

Run the CUDA test suite:
```bash
cargo test --features cuda --test cuda
```

Expected output: all CUDA tests pass, with only known limitations ignored (if any).
