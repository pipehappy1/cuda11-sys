

fn main() {
    println!("cargo:include={}", "/usr/local/cuda/include");
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib={}", "cublasLt");
    println!("cargo:rustc-link-lib=dylib={}", "cudart");
    //println!("cargo:rustc-link-lib=dylib={}", "cublas");
    println!("cargo:rerun-if-changed=build.rs");

    

    #[cfg(feature="generate")]
    {
        use std::path::PathBuf;

        println!("cargo:warning=Running bindgen(cublas-sys), make sure to have all required host libs installed!");
        
        let bindings = bindgen::Builder::default()
            .rust_target(bindgen::RustTarget::Stable_1_40)
            .ctypes_prefix("::libc")
            .size_t_is_usize(true)
            .clang_arg("-I")
            .clang_arg("/usr/local/cuda/include")
            .header("wrapper.h")
            .rustified_non_exhaustive_enum("cuda.*")
            .whitelist_function("cu.*")
            .whitelist_type("[Cc][Uu].*")
            .default_alias_style(bindgen::AliasVariation::TypeAlias )
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from("src").join("cuda.rs");
        
        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");
    }


    
}
