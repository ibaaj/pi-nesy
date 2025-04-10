//
//  ComputeNabla.metal
//
//  Created by Ismaïl Baaj on 02.22.24
//

// https://stackoverflow.com/questions/52992783/metallib-error-reading-module-invalid-bitcode-signature
// xcrun -sdk macosx metal -fcikernel ComputeNabla.metal -c -o ComputeNabla.ir 
// xcrun -sdk macosx metallib -o ComputeNabla.metallib ComputeNabla.ir 

// xcrun -sdk macosx metal -fcikernel ComputeNabla.metal -c -o ComputeNabla.ir && xcrun -sdk macosx metallib -o ComputeNabla.metallib ComputeNabla.ir 


#include <metal_stdlib>
using namespace metal;


// Function to compute max(x, 0)
inline float maxZero(float x) {
    return max(x, 0.0f);
}

// Function to compute sigma_epsilon(u, v, w)
inline float sigmaEpsilon(float u, float v, float w) {
    return min(maxZero((w - u) / 2.0f), maxZero(w - v));
}

kernel void computeChebyshevDistanceNabla(
    device float *A [[buffer(0)]], // Matrix A
    device float *b [[buffer(1)]], // Vector b
    device float *result [[buffer(2)]], // Output result
    constant uint &n [[buffer(3)]], // Number of rows in A and b
    constant uint &m [[buffer(4)]], // Number of columns in A
    constant float &epsilon [[buffer(5)]], // epsilon so that 1-e ~= 1, e~=0
    uint i [[thread_position_in_grid]]
) {

    float nabla_i = FLT_MAX;

    for (uint j = 0; j < m; ++j) {
        float maxTerm = 0.0f;
        for (uint k = 0; k < n; ++k) {
            float a_ij = A[i * m + j]; // Element of A at (i, j)
            float a_kj = A[k * m + j]; // Element of A at (k, j)
            float b_i = b[i]; // Element of b at i
            float b_k = b[k]; // Element of b at k
            maxTerm = max(maxTerm, maxZero(a_ij - b_i));
            maxTerm = max(maxTerm, sigmaEpsilon(b_i, a_kj, b_k));

            if(maxTerm > 1.0 - epsilon){
                    break;
            }
                

        }
        nabla_i = min(nabla_i, maxTerm);
        if(nabla_i < epsilon){
            break;
        }
    }

    result[i] = nabla_i; 
}
