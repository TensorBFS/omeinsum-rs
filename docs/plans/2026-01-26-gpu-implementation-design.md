# GPU Implementation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU support for omeinsum using cuTENSOR (Standard algebra) and tropical-gemm (Tropical algebra).

**Architecture:** cudarc for memory management, direct cuTENSOR FFI bindings for Standard algebra tensor contractions, tropical-gemm for Tropical algebra operations.

**Tech Stack:** cudarc, cuTENSOR (NVIDIA), tropical-gemm

---

## Feature Flags

```toml
[features]
# Tropical algebra (with optimized kernels)
tropical = ["dep:tropical-gemm"]

# GPU backend
cuda = ["dep:cudarc"]

# Everything
full = ["tropical", "cuda"]
```

| Command | Standard CPU | Tropical CPU | Standard GPU | Tropical GPU |
|---------|--------------|--------------|--------------|--------------|
| `cargo build` | faer | - | - | - |
| `--features tropical` | faer | tropical-gemm | - | - |
| `--features cuda` | faer | - | cuTENSOR | - |
| `--features full` | faer | tropical-gemm | cuTENSOR | tropical-gemm |

---

## Module Structure

```
src/backend/cuda/
├── mod.rs              # Cuda backend, Backend trait impl
├── storage.rs          # CudaStorage<T> wrapper
└── cutensor/
    ├── mod.rs          # Public API
    ├── sys.rs          # Raw FFI bindings
    ├── handle.rs       # RAII wrappers
    └── contract.rs     # Plan caching, execution
```

---

## Task 1: CUDA Storage Layer

**Files:**
- Create: `src/backend/cuda/mod.rs`
- Create: `src/backend/cuda/storage.rs`
- Modify: `src/backend/mod.rs`
- Modify: `Cargo.toml`

### Step 1: Update Cargo.toml

```toml
[dependencies]
cudarc = { version = "0.12", optional = true }

[features]
tropical = ["dep:tropical-gemm"]
cuda = ["dep:cudarc"]
full = ["tropical", "cuda"]
```

### Step 2: Create CudaStorage

```rust
// src/backend/cuda/storage.rs
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::sync::Arc;
use crate::algebra::Scalar;

pub struct CudaStorage<T> {
    slice: CudaSlice<T>,
    device: Arc<CudaDevice>,
}

impl<T> CudaStorage<T> {
    pub fn new(slice: CudaSlice<T>, device: Arc<CudaDevice>) -> Self {
        Self { slice, device }
    }

    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }
}

impl<T: DeviceRepr + Clone> CudaStorage<T> {
    pub fn to_vec(&self) -> Vec<T> {
        self.device.dtoh_sync_copy(&self.slice).unwrap()
    }
}
```

### Step 3: Create Cuda backend

```rust
// src/backend/cuda/mod.rs
mod storage;
pub use storage::CudaStorage;

use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct Cuda {
    device: Arc<CudaDevice>,
}

impl Cuda {
    pub fn new() -> Result<Self, CudaError> {
        Self::on_device(0)
    }

    pub fn on_device(ordinal: usize) -> Result<Self, CudaError> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(Self { device: Arc::new(device) })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

#[derive(Debug)]
pub enum CudaError {
    Device(String),
    Alloc(String),
    Cutensor(String),
}
```

### Step 4: Update backend/mod.rs

```rust
// Add to src/backend/mod.rs
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{Cuda, CudaStorage, CudaError};
```

### Step 5: Test

```bash
cargo build --features cuda
```

### Step 6: Commit

```bash
git add -A && git commit -m "feat(cuda): add CUDA storage layer"
```

---

## Task 2: cuTENSOR FFI Bindings

**Files:**
- Create: `src/backend/cuda/cutensor/sys.rs`
- Create: `src/backend/cuda/cutensor/mod.rs`
- Create: `build.rs`

### Step 1: Create build.rs

```rust
// build.rs
fn main() {
    #[cfg(feature = "cuda")]
    {
        if let Ok(path) = std::env::var("CUTENSOR_PATH") {
            println!("cargo:rustc-link-search=native={}", path);
        } else if let Ok(cuda) = std::env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
        println!("cargo:rustc-link-lib=dylib=cutensor");
        println!("cargo:rerun-if-env-changed=CUTENSOR_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
    }
}
```

### Step 2: Create FFI types

```rust
// src/backend/cuda/cutensor/sys.rs
#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cutensorHandle_t = *mut c_void;
pub type cutensorTensorDescriptor_t = *mut c_void;
pub type cutensorOperationDescriptor_t = *mut c_void;
pub type cutensorPlanPreference_t = *mut c_void;
pub type cutensorPlan_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cutensorStatus_t {
    SUCCESS = 0,
    NOT_INITIALIZED = 1,
    ALLOC_FAILED = 3,
    INVALID_VALUE = 7,
    ARCH_MISMATCH = 8,
    NOT_SUPPORTED = 15,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorDataType_t {
    R_32F = 0,
    R_64F = 1,
    C_32F = 4,
    C_64F = 5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorComputeDescriptor_t {
    COMPUTE_32F = 0x40,
    COMPUTE_64F = 0x41,
}

#[repr(C)]
pub enum cutensorWorksizePreference_t {
    DEFAULT = 0,
}

pub const ALGO_DEFAULT: u32 = 0;
pub const JIT_MODE_NONE: u32 = 0;
```

### Step 3: Create FFI functions

```rust
// Add to src/backend/cuda/cutensor/sys.rs

#[link(name = "cutensor")]
extern "C" {
    pub fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t;
    pub fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t;

    pub fn cutensorCreateTensorDescriptor(
        handle: cutensorHandle_t,
        desc: *mut cutensorTensorDescriptor_t,
        num_modes: u32,
        extent: *const i64,
        stride: *const i64,
        data_type: cutensorDataType_t,
        alignment: u32,
    ) -> cutensorStatus_t;
    pub fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;

    pub fn cutensorCreateContraction(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        desc_a: cutensorTensorDescriptor_t, modes_a: *const i32, align_a: u32,
        desc_b: cutensorTensorDescriptor_t, modes_b: *const i32, align_b: u32,
        desc_c: cutensorTensorDescriptor_t, modes_c: *const i32, align_c: u32,
        desc_d: cutensorTensorDescriptor_t, modes_d: *const i32, align_d: u32,
        compute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;
    pub fn cutensorDestroyOperationDescriptor(desc: cutensorOperationDescriptor_t) -> cutensorStatus_t;

    pub fn cutensorCreatePlanPreference(
        handle: cutensorHandle_t,
        pref: *mut cutensorPlanPreference_t,
        algo: u32,
        jit: u32,
    ) -> cutensorStatus_t;
    pub fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t;

    pub fn cutensorEstimateWorkspaceSize(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_pref: cutensorWorksizePreference_t,
        ws_size: *mut u64,
    ) -> cutensorStatus_t;

    pub fn cutensorCreatePlan(
        handle: cutensorHandle_t,
        plan: *mut cutensorPlan_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_size: u64,
    ) -> cutensorStatus_t;
    pub fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t;

    pub fn cutensorContract(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        ws_size: u64,
        stream: *mut c_void,
    ) -> cutensorStatus_t;
}
```

### Step 4: Create cutensor/mod.rs

```rust
// src/backend/cuda/cutensor/mod.rs
pub mod sys;

use sys::cutensorStatus_t;

#[derive(Debug)]
pub enum CutensorError {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    NotSupported,
    Other(i32),
}

pub fn check(status: cutensorStatus_t) -> Result<(), CutensorError> {
    match status {
        cutensorStatus_t::SUCCESS => Ok(()),
        cutensorStatus_t::NOT_INITIALIZED => Err(CutensorError::NotInitialized),
        cutensorStatus_t::ALLOC_FAILED => Err(CutensorError::AllocFailed),
        cutensorStatus_t::INVALID_VALUE => Err(CutensorError::InvalidValue),
        cutensorStatus_t::NOT_SUPPORTED => Err(CutensorError::NotSupported),
        other => Err(CutensorError::Other(other as i32)),
    }
}
```

### Step 5: Commit

```bash
git add -A && git commit -m "feat(cuda): add cuTENSOR FFI bindings"
```

---

## Task 3: Safe cuTENSOR Wrappers

**Files:**
- Create: `src/backend/cuda/cutensor/handle.rs`
- Modify: `src/backend/cuda/cutensor/mod.rs`

### Step 1: Create Handle wrapper

```rust
// src/backend/cuda/cutensor/handle.rs
use super::sys::*;
use super::{check, CutensorError};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct Handle {
    raw: cutensorHandle_t,
    device: Arc<CudaDevice>,
}

impl Handle {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CutensorError> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { cutensorCreate(&mut raw) })?;
        Ok(Self { raw, device })
    }

    pub fn raw(&self) -> cutensorHandle_t { self.raw }
    pub fn device(&self) -> &Arc<CudaDevice> { &self.device }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { cutensorDestroy(self.raw) };
    }
}
```

### Step 2: Create type mapping trait

```rust
// Add to src/backend/cuda/cutensor/handle.rs
use crate::algebra::{Complex32, Complex64};

pub trait CutensorType: Copy {
    const DATA: cutensorDataType_t;
    const COMPUTE: cutensorComputeDescriptor_t;
}

impl CutensorType for f32 {
    const DATA: cutensorDataType_t = cutensorDataType_t::R_32F;
    const COMPUTE: cutensorComputeDescriptor_t = cutensorComputeDescriptor_t::COMPUTE_32F;
}

impl CutensorType for f64 {
    const DATA: cutensorDataType_t = cutensorDataType_t::R_64F;
    const COMPUTE: cutensorComputeDescriptor_t = cutensorComputeDescriptor_t::COMPUTE_64F;
}

impl CutensorType for Complex32 {
    const DATA: cutensorDataType_t = cutensorDataType_t::C_32F;
    const COMPUTE: cutensorComputeDescriptor_t = cutensorComputeDescriptor_t::COMPUTE_32F;
}

impl CutensorType for Complex64 {
    const DATA: cutensorDataType_t = cutensorDataType_t::C_64F;
    const COMPUTE: cutensorComputeDescriptor_t = cutensorComputeDescriptor_t::COMPUTE_64F;
}
```

### Step 3: Create TensorDescriptor wrapper

```rust
// Add to src/backend/cuda/cutensor/handle.rs

pub struct TensorDesc {
    raw: cutensorTensorDescriptor_t,
}

impl TensorDesc {
    pub fn new<T: CutensorType>(
        handle: &Handle,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, CutensorError> {
        let extent: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let stride: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            cutensorCreateTensorDescriptor(
                handle.raw(), &mut raw,
                shape.len() as u32,
                extent.as_ptr(), stride.as_ptr(),
                T::DATA, 0,
            )
        })?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> cutensorTensorDescriptor_t { self.raw }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe { cutensorDestroyTensorDescriptor(self.raw) };
    }
}
```

### Step 4: Create Plan wrapper

```rust
// Add to src/backend/cuda/cutensor/handle.rs

pub struct Plan {
    raw: cutensorPlan_t,
    pub workspace_size: u64,
}

impl Plan {
    pub fn new<T: CutensorType>(
        handle: &Handle,
        desc_a: &TensorDesc, modes_a: &[i32],
        desc_b: &TensorDesc, modes_b: &[i32],
        desc_c: &TensorDesc, modes_c: &[i32],
    ) -> Result<Self, CutensorError> {
        // Create operation descriptor
        let mut op = std::ptr::null_mut();
        check(unsafe {
            cutensorCreateContraction(
                handle.raw(), &mut op,
                desc_a.raw(), modes_a.as_ptr(), 0,
                desc_b.raw(), modes_b.as_ptr(), 0,
                desc_c.raw(), modes_c.as_ptr(), 0,
                desc_c.raw(), modes_c.as_ptr(), 0,
                T::COMPUTE,
            )
        })?;

        // Create plan preference
        let mut pref = std::ptr::null_mut();
        check(unsafe {
            cutensorCreatePlanPreference(handle.raw(), &mut pref, ALGO_DEFAULT, JIT_MODE_NONE)
        })?;

        // Estimate workspace
        let mut workspace_size = 0u64;
        check(unsafe {
            cutensorEstimateWorkspaceSize(
                handle.raw(), op, pref,
                cutensorWorksizePreference_t::DEFAULT,
                &mut workspace_size,
            )
        })?;

        // Create plan
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            cutensorCreatePlan(handle.raw(), &mut raw, op, pref, workspace_size)
        })?;

        // Cleanup
        unsafe {
            cutensorDestroyPlanPreference(pref);
            cutensorDestroyOperationDescriptor(op);
        }

        Ok(Self { raw, workspace_size })
    }

    pub fn raw(&self) -> cutensorPlan_t { self.raw }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { cutensorDestroyPlan(self.raw) };
    }
}
```

### Step 5: Update cutensor/mod.rs

```rust
// src/backend/cuda/cutensor/mod.rs
pub mod sys;
mod handle;

pub use handle::{Handle, TensorDesc, Plan, CutensorType};

// ... error types
```

### Step 6: Commit

```bash
git add -A && git commit -m "feat(cuda): add safe cuTENSOR wrappers"
```

---

## Task 4: Plan Caching and Contraction

**Files:**
- Create: `src/backend/cuda/cutensor/contract.rs`
- Modify: `src/backend/cuda/cutensor/mod.rs`

### Step 1: Create plan cache

```rust
// src/backend/cuda/cutensor/contract.rs
use super::{Handle, TensorDesc, Plan, CutensorType, CutensorError, check};
use super::sys::cutensorContract;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct CacheKey {
    shapes: Vec<Vec<usize>>,
    strides: Vec<Vec<usize>>,
    modes: Vec<Vec<i32>>,
    dtype: u32,
}

pub struct PlanCache {
    cache: HashMap<CacheKey, Plan>,
    capacity: usize,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        Self { cache: HashMap::new(), capacity }
    }

    pub fn get_or_create<T: CutensorType>(
        &mut self,
        handle: &Handle,
        key: CacheKey,
        desc_a: &TensorDesc, modes_a: &[i32],
        desc_b: &TensorDesc, modes_b: &[i32],
        desc_c: &TensorDesc, modes_c: &[i32],
    ) -> Result<&Plan, CutensorError> {
        if !self.cache.contains_key(&key) {
            if self.cache.len() >= self.capacity {
                let k = self.cache.keys().next().cloned().unwrap();
                self.cache.remove(&k);
            }
            let plan = Plan::new::<T>(handle, desc_a, modes_a, desc_b, modes_b, desc_c, modes_c)?;
            self.cache.insert(key.clone(), plan);
        }
        Ok(self.cache.get(&key).unwrap())
    }
}
```

### Step 2: Create contract function

```rust
// Add to src/backend/cuda/cutensor/contract.rs

pub fn contract<T>(
    handle: &Handle,
    plan: &Plan,
    alpha: T,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
) -> Result<(), CutensorError>
where
    T: CutensorType + DeviceRepr + num_traits::Zero,
{
    let workspace = if plan.workspace_size > 0 {
        Some(handle.device().alloc_zeros::<u8>(plan.workspace_size as usize)
            .map_err(|_| CutensorError::AllocFailed)?)
    } else {
        None
    };

    let ws_ptr = workspace.as_ref()
        .map(|w| *w.device_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    let beta = T::zero();

    check(unsafe {
        cutensorContract(
            handle.raw(), plan.raw(),
            &alpha as *const T as *const _,
            *a.device_ptr() as *const _,
            *b.device_ptr() as *const _,
            &beta as *const T as *const _,
            *c.device_ptr() as *const _,
            *c.device_ptr() as *mut _,
            ws_ptr, plan.workspace_size,
            std::ptr::null_mut(),
        )
    })
}
```

### Step 3: Update mod.rs

```rust
// src/backend/cuda/cutensor/mod.rs
pub mod sys;
mod handle;
mod contract;

pub use handle::{Handle, TensorDesc, Plan, CutensorType};
pub use contract::{PlanCache, CacheKey, contract};
```

### Step 4: Commit

```bash
git add -A && git commit -m "feat(cuda): add plan caching and contraction"
```

---

## Task 5: Integrate with Cuda Backend

**Files:**
- Modify: `src/backend/cuda/mod.rs`

### Step 1: Add cuTENSOR to Cuda struct

```rust
// src/backend/cuda/mod.rs
mod storage;
mod cutensor;

pub use storage::CudaStorage;

use cutensor::{Handle, PlanCache};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::cell::RefCell;

pub struct Cuda {
    device: Arc<CudaDevice>,
    handle: RefCell<Option<Handle>>,
    cache: RefCell<PlanCache>,
}

impl Cuda {
    pub fn new() -> Result<Self, CudaError> {
        Self::on_device(0)
    }

    pub fn on_device(ordinal: usize) -> Result<Self, CudaError> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(Self {
            device: Arc::new(device),
            handle: RefCell::new(None),
            cache: RefCell::new(PlanCache::new(64)),
        })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn handle(&self) -> Result<std::cell::Ref<Handle>, CudaError> {
        {
            let mut h = self.handle.borrow_mut();
            if h.is_none() {
                *h = Some(Handle::new(self.device.clone())
                    .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?);
            }
        }
        Ok(std::cell::Ref::map(self.handle.borrow(), |h| h.as_ref().unwrap()))
    }
}
```

### Step 2: Add contraction method

```rust
// Add to src/backend/cuda/mod.rs
use cutensor::{TensorDesc, CacheKey, CutensorType, contract};
use crate::algebra::Scalar;

impl Cuda {
    pub fn contract<T>(
        &self,
        a: &CudaStorage<T>, shape_a: &[usize], strides_a: &[usize], modes_a: &[i32],
        b: &CudaStorage<T>, shape_b: &[usize], strides_b: &[usize], modes_b: &[i32],
        shape_c: &[usize], strides_c: &[usize], modes_c: &[i32],
    ) -> Result<CudaStorage<T>, CudaError>
    where
        T: Scalar + CutensorType + cudarc::driver::DeviceRepr + num_traits::One,
    {
        let handle = self.handle()?;

        let desc_a = TensorDesc::new::<T>(&handle, shape_a, strides_a)
            .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?;
        let desc_b = TensorDesc::new::<T>(&handle, shape_b, strides_b)
            .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?;
        let desc_c = TensorDesc::new::<T>(&handle, shape_c, strides_c)
            .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?;

        let key = CacheKey {
            shapes: vec![shape_a.to_vec(), shape_b.to_vec(), shape_c.to_vec()],
            strides: vec![strides_a.to_vec(), strides_b.to_vec(), strides_c.to_vec()],
            modes: vec![modes_a.to_vec(), modes_b.to_vec(), modes_c.to_vec()],
            dtype: T::DATA as u32,
        };

        let plan = self.cache.borrow_mut()
            .get_or_create::<T>(&handle, key, &desc_a, modes_a, &desc_b, modes_b, &desc_c, modes_c)
            .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?;

        let len: usize = shape_c.iter().product();
        let mut c = self.device.alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))?;

        contract::<T>(&handle, plan, T::one(), a.slice(), b.slice(), &mut c)
            .map_err(|e| CudaError::Cutensor(format!("{:?}", e)))?;

        Ok(CudaStorage::new(c, self.device.clone()))
    }
}
```

### Step 3: Commit

```bash
git add -A && git commit -m "feat(cuda): integrate cuTENSOR with Cuda backend"
```

---

## Task 6: Tests

**Files:**
- Create: `tests/cuda.rs`

### Step 1: Create GPU tests

```rust
// tests/cuda.rs
#![cfg(feature = "cuda")]

use omeinsum::backend::{Cuda, CudaStorage};

#[test]
fn test_cuda_init() {
    let cuda = Cuda::new();
    assert!(cuda.is_ok());
}

#[test]
fn test_storage_roundtrip() {
    let cuda = Cuda::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let slice = cuda.device().htod_sync_copy(&data).unwrap();
    let storage = CudaStorage::new(slice, cuda.device().clone());
    assert_eq!(storage.to_vec(), data);
}

#[test]
fn test_matmul_f32() {
    let cuda = Cuda::new().unwrap();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

    let a = CudaStorage::new(cuda.device().htod_sync_copy(&a_data).unwrap(), cuda.device().clone());
    let b = CudaStorage::new(cuda.device().htod_sync_copy(&b_data).unwrap(), cuda.device().clone());

    // C[i,k] = sum_j A[i,j] * B[j,k]
    let c = cuda.contract::<f32>(
        &a, &[2, 3], &[3, 1], &[0, 1],
        &b, &[3, 2], &[2, 1], &[1, 2],
        &[2, 2], &[2, 1], &[0, 2],
    ).unwrap();

    let result = c.to_vec();
    // Expected: [[22, 28], [49, 64]]
    assert!((result[0] - 22.0).abs() < 1e-5);
    assert!((result[1] - 28.0).abs() < 1e-5);
    assert!((result[2] - 49.0).abs() < 1e-5);
    assert!((result[3] - 64.0).abs() < 1e-5);
}
```

### Step 2: Commit

```bash
git add -A && git commit -m "test(cuda): add GPU tests"
```

---

## Success Criteria

1. `cargo build --features cuda` compiles
2. `cargo test --features cuda` passes on GPU machine
3. Plan caching works (same contraction reuses plan)
4. Results match CPU within floating-point tolerance

## Environment Requirements

- CUDA Toolkit 12.x
- cuTENSOR 2.x (`CUTENSOR_PATH` or in `$CUDA_PATH/lib64`)
- GPU with compute capability 7.0+
