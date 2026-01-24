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

   
    const int steps = ((N*N + ((tile_size_x*tile_size_x)-1)) / (tile_size_x*tile_size_x)); // ceil
    int g_index_x;
    int g_index_y;
    if (tidX < N && tidY < N){ // since tidX & tidY are typically 32,64,96, thus can be larger than the NxN matrix
        // loads tiles into shared memory from global memory
        for (int i = 0; i < steps; i++ ){

            //2 + 0
            g_index_x = (i*tile_size_x + (blockIdx.x * tile_size_x*N) + (tidY * tile_size_x + tidX)); //right + down this is correct for any m*m tile size
            g_index_y = (i*tile_size_x*N + (blockIdx.x * tile_size_x) + (tidY * tile_size_x + tidX)); //down + right
            if (g_index_x < N*N){ // not sure if this check is redudant yet; should try removing later
                s_tile1[tidY * tile_size_x + tidX] = g_matrix1[g_index_x]; //correctly retrieves a tile from global memory
                
            }
            if (g_index_y < N*N){ // not sure if this check is redudant yet; should try removing later
                s_tile2[tidY * tile_size_x + tidX] = g_matrix2[g_index_y]; //correctly retrieves tile from global memory
            }


            __syncthreads();
        
            if (tidX < tile_size_x && tidY == 0){
                // tidX dicates which row in s_dot_product we write to
                for (int j = 0; j < tile_size_x; j++){
                    if ( i == 0){ // on first step
                        s_dot_product[tidX * tile_size_x + j] = 0;
                    }
                    for (int k = 0; k < tile_size_x; k++){
                        // if i == 0 first row of m1 and first column of m2
                        // j,k
                        // 0,0 = 0 * 0
                        // +
                        // 0,1 = 1 * 2

                        // 1,0 = 2 * 1
                        // +
                        // 1,1 = 3 * 3

                        s_dot_product[tidX * tile_size_x + j] += (s_tile1[j * tile_size_x + k] * s_tile2[k * tile_size_x + j]);
                        
                    }

                }

            }
            

            // __syncthreads();
        }
        if (tidX < tile_size_x  && tidY < tile_size_x){
            g_product[g_index_x] = s_dot_product[tidY * tile_size_x + tidX];
        }
        

    }

}


int main(){
    const int matrixLength = lo;

    float h_myMatrix1[matrixLength*matrixLength];
    float h_myMatrix2[matrixLength*matrixLength];
    float h_product[matrixLength*matrixLength];
    float* d_product;
    for (int i = 0; i < pow(matrixLength,2); i++){
        h_myMatrix1[i] = 2;
        h_myMatrix2[i] = 2;
        h_product[i] = 65;
    }

    float* d_myMatrix1;
    float* d_myMatrix2;

    int shared_memory_required_in_bytes = sizeof(float)* pow(matrixLength,2);

    cudaMalloc((void**)&d_myMatrix1, shared_memory_required_in_bytes);
    cudaMalloc((void**)&d_myMatrix2, shared_memory_required_in_bytes);
    
    cudaMalloc((void**)& d_product, shared_memory_required_in_bytes);

    cudaMemcpy(d_myMatrix1, h_myMatrix1, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_myMatrix2, h_myMatrix2, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_product, h_product, shared_memory_required_in_bytes, cudaMemcpyHostToDevice);
    

    
    
    
    dim3 gridSize(floor((matrixLength + 31) / 32));
    dim3 blockSize(32,32);
    wxb<<<gridSize,blockSize>>>(32,d_myMatrix1,d_myMatrix2, d_product);

    cudaMemcpy(h_product, d_product, shared_memory_required_in_bytes, cudaMemcpyDeviceToHost);
    
    // cudaDeviceSynchronize();

    for (int i = 0; i < matrixLength; i++){
        for (int j = 0; j < matrixLength; j++){
            std::cout << h_product[j * matrixLength + i] << ',';
        }
        std::cout << "\n\n";
    }


    cudaFree(d_myMatrix1);
    cudaFree(d_myMatrix2);
    cudaFree(d_product);
    std::cout << "donesies" << std::endl;

    

    
}
