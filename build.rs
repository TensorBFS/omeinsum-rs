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
