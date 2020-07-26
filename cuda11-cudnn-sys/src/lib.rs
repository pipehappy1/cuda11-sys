mod cudnn;

pub use cudnn::*;


#[cfg(test)]
mod tests {
    use super::*;
    use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpyAsync, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind};

    fn checkCudaStatus(status: cuda11_cudart_sys::cudaError_t ) {
        if status != cuda11_cudart_sys::cudaError::cudaSuccess {
            print!("cuda API failed with status \n");
            panic!();
        }
    }



    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {

        unsafe {
            
        }

    }
}
