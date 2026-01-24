#include <iostream>
#include <cuda_runtime.h>

void checkFragmentation() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // We try to find the largest contiguous block via a binary search
    size_t low = 0, high = free_mem;
    size_t largest_block = 0;

    while (low <= high) {
        size_t mid = low + (high - low) / 2;
        void* ptr = nullptr;
        
        if (cudaMalloc(&ptr, mid) == cudaSuccess) {
            largest_block = mid;
            cudaFree(ptr);
            low = mid + 1;
        } else {
            high = mid - 1;
            cudaGetLastError(); // Clear the last error
        }
    }

    double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
    double largest_gb = largest_block / (1024.0 * 1024.0 * 1024.0);
    
    // Fragmentation Ratio: 0 is perfect (no fragmentation), 1 is unusable.
    double frag_ratio = 1.0 - (static_cast<double>(largest_block) / free_mem);

    std::cout << "Total Free Memory: " << free_gb << " GB" << std::endl;
    std::cout << "Largest Contiguous Block: " << largest_gb << " GB" << std::endl;
    std::cout << "Fragmentation Ratio: " << frag_ratio << std::endl;

    if (frag_ratio > 0.1) {
        std::cout << "Warning: High fragmentation detected!" << std::endl;
    }
}

int main() {
    checkFragmentation();
    return 0;
}
