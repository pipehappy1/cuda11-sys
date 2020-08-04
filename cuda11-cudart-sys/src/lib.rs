mod cudart;

pub use crate::cudart::*;


fn checkCudaStatus(status: cudaError_t ) {
    if status != cudaError::cudaSuccess {
        print!("cuda API failed with status \n");
        panic!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

}
