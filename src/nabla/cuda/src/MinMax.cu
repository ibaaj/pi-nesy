#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

__global__ void MinMaxKernel(
    float *A,    // Matrix A
    float *x,    // Vector x
    float *result, // Output result, one element per row of A
    int n,       // Number of rows in A
    int m,       // Number of columns in A
    float epsilon // Small threshold for numerical stability, not used here
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float minMax = FLT_MAX; 
        for (int col = 0; col < m; ++col) {
            float currentElement = A[row * m + col];
            float vectorElement = x[col];
            float maxVal = fmax(currentElement, vectorElement);
            minMax = fmin(minMax, maxVal);
        }
        result[row] = minMax; 
    }
}

void MinMax(const float* A, const float* x, float* result, int n, int m, float epsilon) {
    float* A_gpu = nullptr;
    float* x_gpu = nullptr;
    float* result_gpu = nullptr;

    cudaMalloc(&A_gpu, n * m * sizeof(float));
    cudaMalloc(&x_gpu, m * sizeof(float));
    cudaMalloc(&result_gpu, n * sizeof(float));

    cudaMemcpy(A_gpu, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(512); // Adjust this based on the device capabilities
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start);
    MinMaxKernel<<<gridSize, blockSize>>>(A_gpu, x_gpu, result_gpu, n, m, epsilon);
    
    cudaEventRecord(stop);

    
    cudaEventSynchronize(stop);

   
    cudaMemcpy(result, result_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(A_gpu);
    cudaFree(x_gpu);
    cudaFree(result_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
