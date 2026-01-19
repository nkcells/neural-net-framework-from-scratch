#include <stdio.h>
#include <iostream>
#define lo 100


__global__ void wxb(const int m1x, const int m2y, const int  m1ym2x, float* g_matrix1, float* g_matrix2, float* g_product ){ //kernel good that does loxlo * lo*lo multiplicaiton 
    
    int threadID = blockIdx.x * lo + threadIdx.x; 

    int threadIDInverse = threadIdx.x * lo + blockIdx.y;

    float myVal = g_matrix1[threadID] * g_matrix2[threadIDInverse];

    atomicAdd(g_product + (blockIdx.x * lo + blockIdx.y),myVal ); // makes it serial


}


int main(){
    const int matrixLength = lo;

    float h_myMatrix1[matrixLength*matrixLength];
    float h_myMatrix2[matrixLength*matrixLength];
    float h_product[matrixLength*matrixLength];
    float* d_product;
    for (int i = 0; i < pow(matrixLength,2); i++){
        h_myMatrix1[i] = i;
        h_myMatrix2[i] = i;
        h_product[i] = 0;
    }

    float* d_myMatrix1;
    float* d_myMatrix2;

    int shared_memory_required_in_bytes = sizeof(float)* pow(matrixLength,2);

    cudaMalloc((void**)&d_myMatrix1, shared_memory_required_in_bytes);
    cudaMalloc((void**)&d_myMatrix2, shared_memory_required_in_bytes);
    
    cudaMalloc((void**)&d_product, sizeof(float)* pow(matrixLength,2));

    cudaMemcpy(d_myMatrix1, h_myMatrix1, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_myMatrix2, h_myMatrix2, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_product, h_product, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);
    

    
    
    
    dim3 gridSize(lo,lo);
    dim3 blockSize(lo);
    wxb<<<gridSize,blockSize, shared_memory_required_in_bytes>>>(lo,lo,lo,d_myMatrix1,d_myMatrix2, d_product);

    cudaMemcpy(h_product, d_product, shared_memory_required_in_bytes, cudaMemcpyDeviceToHost);
    
    // cudaDeviceSynchronize();

    // for (int i = 0; i < matrixLength; i++){
    //     for (int j = 0; j < matrixLength; j++){
    //         std::cout << h_product[j * matrixLength + i] << ',';
    //     }
    //     std::cout << "\n\n";
    // }

    std::cout << "donesies" << std::endl;

    

    
}
