//! Raw FFI bindings to cuTENSOR library.
//!
//! This module provides low-level bindings to NVIDIA's cuTENSOR library
//! for high-performance tensor contractions on CUDA GPUs.

#![allow(non_camel_case_types)]

use std::ffi::c_void;

// Opaque handle types
pub type cutensorHandle_t = *mut c_void;
pub type cutensorTensorDescriptor_t = *mut c_void;
pub type cutensorOperationDescriptor_t = *mut c_void;
pub type cutensorPlanPreference_t = *mut c_void;
pub type cutensorPlan_t = *mut c_void;

/// Status codes returned by cuTENSOR functions.
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

/// Data types supported by cuTENSOR.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorDataType_t {
    R_32F = 0,
    R_64F = 1,
    C_32F = 4,
    C_64F = 5,
}

/// Compute descriptor types for specifying computation precision.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorComputeDescriptor_t {
    COMPUTE_32F = 0x40,
    COMPUTE_64F = 0x41,
}

/// Workspace size preference for operation planning.
#[repr(C)]
pub enum cutensorWorksizePreference_t {
    DEFAULT = 0,
}

/// Default algorithm selection.
pub const ALGO_DEFAULT: u32 = 0;

/// Disable JIT compilation.
pub const JIT_MODE_NONE: u32 = 0;

#[link(name = "cutensor")]
extern "C" {
    /// Create a cuTENSOR handle.
    pub fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t;

    /// Destroy a cuTENSOR handle.
    pub fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t;

    /// Create a tensor descriptor.
    pub fn cutensorCreateTensorDescriptor(
        handle: cutensorHandle_t,
        desc: *mut cutensorTensorDescriptor_t,
        num_modes: u32,
        extent: *const i64,
        stride: *const i64,
        data_type: cutensorDataType_t,
        alignment: u32,
    ) -> cutensorStatus_t;

    /// Destroy a tensor descriptor.
    pub fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;

    /// Create a contraction operation descriptor.
    pub fn cutensorCreateContraction(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        desc_a: cutensorTensorDescriptor_t,
        modes_a: *const i32,
        align_a: u32,
        desc_b: cutensorTensorDescriptor_t,
        modes_b: *const i32,
        align_b: u32,
        desc_c: cutensorTensorDescriptor_t,
        modes_c: *const i32,
        align_c: u32,
        desc_d: cutensorTensorDescriptor_t,
        modes_d: *const i32,
        align_d: u32,
        compute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;

    /// Destroy an operation descriptor.
    pub fn cutensorDestroyOperationDescriptor(
        desc: cutensorOperationDescriptor_t,
    ) -> cutensorStatus_t;

    /// Create a plan preference object.
    pub fn cutensorCreatePlanPreference(
        handle: cutensorHandle_t,
        pref: *mut cutensorPlanPreference_t,
        algo: u32,
        jit: u32,
    ) -> cutensorStatus_t;

    /// Destroy a plan preference object.
    pub fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t;

    /// Estimate the workspace size required for an operation.
    pub fn cutensorEstimateWorkspaceSize(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_pref: cutensorWorksizePreference_t,
        ws_size: *mut u64,
    ) -> cutensorStatus_t;

    /// Create an execution plan for a contraction.
    pub fn cutensorCreatePlan(
        handle: cutensorHandle_t,
        plan: *mut cutensorPlan_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_size: u64,
    ) -> cutensorStatus_t;

    /// Destroy an execution plan.
    pub fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t;

    /// Execute a tensor contraction.
    ///
    /// Computes: D = alpha * A * B + beta * C
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
