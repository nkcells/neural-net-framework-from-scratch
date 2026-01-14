#include <stdio.h>
#include <iostream>

__global__ void add(int* a){ // global => says to run on the gpu
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    a[i] = (a[i] + a[i]);
}

// __managed__ int vector_a[256], vector_b[256], vector_c[256];// can be accessed by the host cpu AND gpu without having to copy between them
// __device__
// int *gpuData;
 
// cudaMalloc(&gpuData, sizeof(int) );

int main() {
    std::cout << "starting" <<std::endl;
    const int n = 600000;
    int h_c[n];
    int *d_c;
    for (int i = 0; i < n; i++){
        h_c[i] = i;
        // vector_b[i] = i;

    }

    
    
    cudaMalloc((void**)&d_c, sizeof(int)*n);
    cudaMemcpy(d_c,h_c, sizeof(int)*n, cudaMemcpyHostToDevice);
        std::cout << "starting" <<std::endl;

    // cudaDeviceSynchronize();
    add<<<n/256+ 255,256>>>(d_c);
        std::cout << "starting" <<std::endl;

  

    cudaMemcpy(h_c, d_c, sizeof(int)*n, cudaMemcpyDeviceToHost);
        std::cout << "starting" <<std::endl;




    // cudaDeviceSynchronize();

    

    for (int i = 0; i < n; i++){
        std::cout << "Sum {" << i << "}: " << h_c[i] << std::endl;
    }
    cudaFree(d_c);
    // free(h_c);


}
