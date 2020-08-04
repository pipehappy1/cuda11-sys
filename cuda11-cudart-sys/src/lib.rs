mod cudart;

pub use crate::cudart::*;


pub fn check_cuda_status(status: cudaError_t ) {
    if status != cudaError::cudaSuccess {
        print!("cuda API failed with status \n");
        panic!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

}
