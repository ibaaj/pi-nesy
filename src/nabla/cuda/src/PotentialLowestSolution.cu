#include <math.h>
#include <float.h>
#include <stdio.h>
#include "cudaPotentialLowestSolution.h"


__global__ void computePotentialLowestSolutionKernel(
    float *A, float *b, float *result, unsigned int N, unsigned int M, float epsilon) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < M) {
        float col = -DBL_MAX; 
        float a_val, b_val;
        for (unsigned int i = 0; i < N; ++i) {
            a_val = A[j * N + i];
            b_val = b[i];
            float epsilonTerm;
            if (abs(a_val - b_val) <= max(epsilon * max(abs(a_val), abs(b_val)), 1e-9)) {
                epsilonTerm = 0.0;
            } else {
                epsilonTerm = a_val < b_val ? b_val : 0.0;
            }
            col = max(epsilonTerm, col);
        }
        result[j] = col;
    }
}

void transposeCPU(const float* input, float* output, unsigned int N, unsigned int M) {
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < M; ++j) {
            output[j * N + i] = input[i * M + j];
        }
    }
}


void PotentialLowestSolution(const float* A, const float* b, float* result, unsigned int N, unsigned int M, float epsilon) {
   float *A_gpu, *b_gpu, *result_gpu, *transposed_A;
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = N * sizeof(float);
    size_t sizeResult = M * sizeof(float);

    
    transposed_A = new float[N * M];

    
    transposeCPU(A, transposed_A, N, M);

    cudaMalloc(&A_gpu, sizeA);
    cudaMalloc(&b_gpu, sizeB);
    cudaMalloc(&result_gpu, sizeResult);

    
    cudaMemcpy(A_gpu, transposed_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeB, cudaMemcpyHostToDevice);

    dim3 blockSize(512);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x);

    
    computePotentialLowestSolutionKernel<<<gridSize, blockSize>>>(A_gpu, b_gpu, result_gpu, N, M, epsilon);

    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in PotentialLowestSolution: %s\n", cudaGetErrorString(error));
    }

    
    cudaMemcpy(result, result_gpu, sizeResult, cudaMemcpyDeviceToHost);

    
    cudaFree(A_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);
    delete[] transposed_A; 
}

