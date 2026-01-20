#include <stdio.h>
#include <iostream>
#define lo 32
#define tile_size_x 32 //should be divisible by 32
#define tile_size_y 32 //should be divisible by 32


//each block running this kernel will have 32x32 threads
__global__ void wxb(const int N, float* g_matrix1, float* g_matrix2, float* g_product ){ //kernel good that does loxlo * lo*lo multiplicaiton 
    
    // each blok is 2d
    const int tidX = threadIdx.x;
    const int tidY = threadIdx.y;


    __shared__ float tile1[tile_size_x * tile_size_x];
    __shared__ float tile2[tile_size_x * tile_size_x];
    __shared__ float dot_product[tile_size_x * tile_size_x];

    dot_product[(blockIdx.x * lo) + (tidY * lo + tidX)];//this is the tile of the output block we are targetting
    // const int N_squared = pow(N,2);

   
    const int steps = N + (tile_size_x-1) / tile_size_x; // ceil
    int index_x;
    int index_y;
    for (int i = 0; i < steps; i++ ){

        index_x = (tile_size_x + ((blockIdx.x * tile_size_x*N)+(tidY * N + tidX))); // this is correct for any m*m tile size
        index_y = (tile_size_x*N + (blockIdx.x * tile_size_x) + tidY*N + tidX); 
        if (index_x < N){
            tile1[tidY * tile_size_x + tidX] = g_matrix1[index_x]; //correctly retrieves a tile from global memory
            
        }
        if (index_y < N){
            tile2[tidY * tile_size_x + tidX] = g_matrix2[index_y]; //correctly retrieves tile from global memory
        }
        
       
    }
    if (threadIdx.x < N && threadIdx.y < N){
        tile1[tidY * lo + tidX] = g_matrix1[];
        tile2[tidY * lo + tidX] = g_matrix1[(blockIdx.x * lo) + (tidY * lo + tidX)];
    }
    

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
    
    cudaMalloc((void**)& d_product, sizeof(float)* pow(matrixLength,2));

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
