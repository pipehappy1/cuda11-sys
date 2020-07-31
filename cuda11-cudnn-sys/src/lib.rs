mod cudnn;

pub use cudnn::*;


#[cfg(test)]
mod tests {
    use super::*;
    use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpy, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind};

    fn checkCudaStatus(status: cuda11_cudart_sys::cudaError_t ) {
        if status != cuda11_cudart_sys::cudaError::cudaSuccess {
            print!("cuda API failed with status \n");
            panic!();
        }
    }

    // not a test for get driver version, but a test for linking
    #[test]
    fn link_test() {
        struct CudaTensor {
            device_data: *mut f32,
            dim: Vec<usize>,
            mm_data: Vec<f32>,
        }
        impl CudaTensor {
            fn new() -> CudaTensor {
                CudaTensor {
                    device_data: std::ptr::null_mut(),
                    dim: Vec::new(),
                    mm_data: Vec::new(),
                }
            }
            fn new_raw(data: &[f32], shape: &[usize]) -> CudaTensor {
                
                let mut device_data: *mut f32 = std::ptr::null_mut();
                let elems: usize = shape.iter().product();
                if elems != data.len() {
                    panic!();
                }

                unsafe {
                    println!("cudaMalloc");
                    checkCudaStatus(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                               std::mem::size_of::<f32>()*elems));
                    println!("cudaMemcpy");
                    cudaMemcpy(device_data as *mut _,
                               data.as_ptr() as *mut _,
                               std::mem::size_of::<f32>()*elems,
                               cudaMemcpyKind::cudaMemcpyHostToDevice);
                }
                
                CudaTensor {
                    device_data: device_data,
                    dim: shape.to_vec(),
                    mm_data: data.to_vec(),
                }
            }
            
            fn _sync(&mut self) {
                let elems: usize = self.dim.iter().product();
                
                unsafe {
                    cudaMemcpy(self.mm_data.as_mut_ptr() as *mut _,
                               self.device_data as *mut _,
                               std::mem::size_of::<f32>()*elems,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost);
                }
            }
            
        }
        impl Drop for CudaTensor {
            fn drop(&mut self) {
                if self.device_data != std::ptr::null_mut() {
	            unsafe {
                        println!("cudaFree");
                        checkCudaStatus(cudaFree(self.device_data as _));                    
                    }
                }
            }
        }
        impl std::fmt::Debug for CudaTensor {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

                write!(f, "{:?}\n", self.dim)?;
                write!(f, "{:?}", self.mm_data)

            }
        }

        struct CudaConv {
            a: f32,
        }
        impl CudaConv {
            fn new() -> CudaConv {
                CudaConv {
                    a: 0.,
                }
            }
            fn forward(&self,
                       input: &CudaTensor,
                       filter: &CudaTensor,
                       alpha: f32,
                       beta: f32,
            ) -> CudaTensor {
                unsafe {

                    let mut cudnnHandle: cudnnHandle_t = std::ptr::null_mut();
                    cudnnCreate(&mut cudnnHandle);

                    let mut srcTensorDesc: cudnnTensorDescriptor_t;
                    let mut dstTensorDesc: cudnnTensorDescriptor_t;
                    let mut biasTensorDesc: cudnnTensorDescriptor_t;

                    let mut filterDesc: cudnnFilterDescriptor_t;

                    let mut  convDesc: cudnnConvolutionDescriptor_t;

                    cudnnCreateTensorDescriptor(&srcTensorDesc as *mut _ as _);
                    cudnnCreateTensorDescriptor(&dstTensorDesc as *mut _ as _);
                    cudnnCreateTensorDescriptor(&biasTensorDesc as *mut _ as _);

                    cudnnCreateFilterDescriptor(&filterDesc as *mut _ as _);

                    cudnnCreateConvolutionDescriptor(&convDesc as *mut _ as _);

                    cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                               filterDimA);

                    cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType);

                    cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                          srcTensorDesc,
                                                          filterDesc,
                                                          tensorDims,
                                                          tensorOuputDimA);

                    setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

                    cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         convDesc,
                                                         dstTensorDesc,
                                                         requestedAlgoCount,
                                                         &returnedAlgoCount,
                                                         results);

                    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                            srcTensorDesc,
                                                            filterDesc,
                                                            convDesc,
                                                            dstTensorDesc,
                                                            algo,
                                                            &sizeInBytes);

                    cudaMalloc(&workSpace,sizeInBytes);
                    
                    cudnnConvolutionForward(cudnnHandle,
                                            &alpha as *const _ as _,
                                            srcTensorDesc,
                                            input.device_data as _,
                                            filterDesc,
                                            filter.device_data as _,
                                            convDesc,
                                            algo,
                                            workSpace,
                                            sizeInBytes,
                                            &beta as *const _ as _,
                                            dstTensorDesc,
                                            *dstData);

                    cudnnDestroyConvolutionDescriptor(convDesc);

                    cudnnDestroyFilterDescriptor(filterDesc);
                    
                    cudnnDestroyTensorDescriptor(srcTensorDesc);
                    cudnnDestroyTensorDescriptor(dstTensorDesc);
                    cudnnDestroyTensorDescriptor(biasTensorDesc);
                    
                    cudnnDestroy(cudnnHandle);
                }
                CudaTensor::new()
            }
        }

        unsafe {

            println!("cudnn version {:?} compiled against cudart version {:?}",
                     cudnnGetVersion(),
                     cudnnGetCudartVersion());
            
            let mut stream: cudaStream_t = std::ptr::null_mut();
            checkCudaStatus(cudaStreamCreate(&mut stream as *mut _ as _));


            
            let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], &vec![1, 1, 3, 3]);
            let mut filter = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], &vec![1, 1, 3, 3]);
            let mut conv = CudaConv::new();
            let mut output = conv.forward(&input, &filter, 1., 0.);

            input._sync();
            println!("{:?}", input);


            
            cudaStreamSynchronize(stream as _);
            checkCudaStatus(cudaStreamDestroy(stream as _));
            
        }

    }
}
