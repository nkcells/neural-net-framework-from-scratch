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


    __shared__ float s_tile1[tile_size_x * tile_size_x];
    __shared__ float s_tile2[tile_size_x * tile_size_x];
    __shared__ float s_dot_product[tile_size_x * tile_size_x];

    // s_dot_product[(blockIdx.x * lo) + (tidY * lo + tidX)];//this is the tile of the output block we are targetting
    // const int N_squared = pow(N,2);

   
    const int steps = N + ((tile_size_x*tile_size_x)-1) / tile_size_x*tile_size_x; // ceil
    int index_x;
    int index_y;
    if (tidX < N && tidY < N){ // since tidX & tidY are typically 32,64,96, thus can be larger than the NxN matrix
        // loads tiles into shared memory from global memory
        for (int i = 0; i < steps; i++ ){

            
            index_x = (tile_size_x + ((blockIdx.x * tile_size_x*N)+(tidY * N + tidX))); // this is correct for any m*m tile size
            index_y = (tile_size_x*N + (blockIdx.x * tile_size_x) + tidY*N + tidX); 
            if (index_x < N*N){ // not sure if this check is redudant yet; should try removing later
                s_tile1[tidY * tile_size_x + tidX] = g_matrix1[index_x]; //correctly retrieves a tile from global memory
                
            }
            if (index_y < N*N){ // not sure if this check is redudant yet; should try removing later
                s_tile2[tidY * tile_size_x + tidX] = g_matrix2[index_y]; //correctly retrieves tile from global memory
            }
            
        }
        
        __syncthreads();
        
        if (tidX < tile_size_x && tidY == 0){
            // tidX dicates which row in s_dot_product we write to
            for (int i = 0; i < tile_size_x; i++){
                for (int k = 0; k < tile_size_x; k++){
                    // if i == 0 first row of m1 and first column of m2
                    
                    s_dot_product[tidX * tile_size_x + i] += (s_tile1[i * tile_size_x + k] * s_tile2[k * tile_size_x + i]);
                }

                //s_tile1[(tidX * tile_size_x) + i] * s_tile2[k + (tidX * tile_size_x)] // * s_tile2[tidX + (i * tile_size_x)]
            }

            
        }

        __syncthreads();

        if (tidX < tile_size_x && tidY < tile_size_x){

            g_product[index_x] =  s_dot_product[tidY * tile_size_x + tidX];
            // this yields interesting results g_product[index_x] = 1;
        }
       //even more intersting g_product[index_x] = 1;

       
    }   

    // tile 1 threads will do the matrix multiplication while tile 2 threads do nothing




    

    

    // int threadID = blockIdx.x * lo + threadIdx.x; 

    // int threadIDInverse = threadIdx.x * lo + blockIdx.y;

    // float myVal = g_matrix1[threadID] * g_matrix2[threadIDInverse];

    // atomicAdd(g_product + (blockIdx.x * lo + blockIdx.y),myVal ); // makes it serial


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
    

    
    
    
    dim3 gridSize(2);
    dim3 blockSize(32,32);
    wxb<<<gridSize,blockSize, shared_memory_required_in_bytes>>>(64,d_myMatrix1,d_myMatrix2, d_product);

    cudaMemcpy(h_product, d_product, shared_memory_required_in_bytes, cudaMemcpyDeviceToHost);
    
    // cudaDeviceSynchronize();

    for (int i = 0; i < matrixLength; i++){
        for (int j = 0; j < matrixLength; j++){
            std::cout << h_product[j * matrixLength + i] << ',';
        }
        std::cout << "\n\n";
    }

    std::cout << "donesies" << std::endl;

    

    
}
