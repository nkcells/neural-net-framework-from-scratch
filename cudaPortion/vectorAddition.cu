#include <stdio.h>
#include <iostream>

__global__ void add(int* a, int* b, int* c){ // global => says to run on the gpu
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];// can be accessed by the host cpu AND gpu without having to copy between them


int main() {
    for (int i = 0; i < 256; i++){
        vector_a[i] = i;
        vector_b[i] = i;

    }

    add<<<4,1024>>>(vector_a, vector_b, vector_c);

    cudaDeviceSynchronize();

    

    for (int i = 0; i < 256; i++){
        std::cout << "Sum {" << i << "}: " << vector_c[i] << std::endl;
    }


}
