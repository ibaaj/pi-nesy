#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

__device__ float maxZero(float x) {
    return max(x, 0.0);
}

__device__ float sigmaEpsilon(float u, float v, float w) {
    return min(maxZero((w - u) / 2.0), maxZero(w - v));
}

__global__ void computeChebyshevDistanceNablaKernel(
    float *A, float *b, float *result, unsigned int n, unsigned int m, float epsilon
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float nabla_i = 1.0;

        for (unsigned int j = 0; j < m; ++j) {
            float maxTerm = 0.0;
            for (unsigned int k = 0; k < n; ++k) {
                float a_ij = A[i * m + j]; 
                float a_kj = A[k * m + j]; 
                float b_i = b[i]; 
                float b_k = b[k]; 
                maxTerm = max(maxTerm, maxZero(a_ij - b_i));
                maxTerm = max(maxTerm, sigmaEpsilon(b_i, a_kj, b_k));

            }
            nabla_i = min(nabla_i, maxTerm);
        }

        result[i] = nabla_i;
    }
}


void ChebyshevDistanceNabla(const float* A, const float* b, float* result, int n, int m, float epsilon) {
    
    float *A_d, *b_d, *result_d;

    
    cudaMalloc(&A_d, n * m * sizeof(float));
    cudaMalloc(&b_d, n * sizeof(float)); 
    cudaMalloc(&result_d, n * sizeof(float));

    
    cudaMemcpy(A_d, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(512);
    dim3 grid((n + block.x - 1) / block.x);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start);

    computeChebyshevDistanceNablaKernel<<<grid, block>>>(A_d, b_d, result_d, n, m, epsilon);

    
    cudaEventRecord(stop);

    
    cudaEventSynchronize(stop);

    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if(milliseconds > 100000)
        printf("Chebyshev distance nabla operation took %f milliseconds\n", milliseconds);

    
    cudaMemcpy(result, result_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(cudaStatus));
       
    }

    
    cudaFree(A_d);
    cudaFree(b_d);
    cudaFree(result_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
