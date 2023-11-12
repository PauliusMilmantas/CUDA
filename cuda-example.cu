#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void parallelSum(int* inputArray, int* output) {
    // Declare a shared memory array
    __shared__ int partialSum[BLOCK_SIZE];

    // Calculate global thread ID
    int tid = threadIdx.x + blockDim.x *  blockIdx.x;

    // Each thread loads an element from global memory to shared memory
    partialSum[threadIdx.x] = inputArray[tid];

    // Synchronize to make sure all threads have finished copying
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

int main() {
    // Set the size of the array
    int arraySize = 1024;

    // Allocate and initialize the array on the host
    int* h_inputArray = (int*)malloc(arraySize * sizeof(int));
    for (int i = 0; i < arraySize; ++i) {
        h_inputArray[i] = i;
    }

    // Allocate memory on the device
    int* d_inputArray, * d_output;
    cudaMalloc((void**)&d_inputArray, arraySize * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copy the input array from the host to the device
    cudaMemcpy(d_inputArray, h_inputArray, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1, 1);

    // Launch the kernel
    parallelSum << <gridSize, blockSize >> > (d_inputArray, d_output);

    // Copy the result back to the host
    int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Sum: %d\n", h_output);

    // Free allocated memory
    free(h_inputArray);
    cudaFree(d_inputArray);
    cudaFree(d_output);

    return 0;
}