mod cudart;

pub use crate::cudart::*;


fn checkCudaStatus(status: cuda11_cudart_sys::cudaError_t ) {
    if status != cuda11_cudart_sys::cudaError::cudaSuccess {
        print!("cuda API failed with status \n");
        panic!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

}
