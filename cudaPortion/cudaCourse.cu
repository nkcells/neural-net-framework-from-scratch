
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y, int *indexSize, int *strideSize)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  *indexSize = index;
  *strideSize = stride;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<28;
  float *x, *y;
  int* indexSize = 0;
  int *strideSize = 0;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&indexSize, sizeof(int));
  cudaMallocManaged(&strideSize, sizeof(int));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  std::cout << *indexSize << std::endl;
  std::cout << *strideSize << std::endl;


  // Run kernel on 1M elements on the GPU
  add<<<100, 256>>>(N, x, y, indexSize, strideSize);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
// << 20
//174380489
//1128818
//1471544
//1260851 {with 40 blocks} 7157077
//1122546 {with 01 blocks} 6678109



// << 28


//442222901 {with 01 blocks} 2308879828
//399122172 {with 40 blocks} 2533455851
//
