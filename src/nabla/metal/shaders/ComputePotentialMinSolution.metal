//
//  ComputePotentialMinSolution.metal
//
//  Created by Ismaïl Baaj on 02.22.24
//

// compilation
// 1) xcrun -sdk macosx metal -fcikernel ComputePotentialMinSolution.metal -c -o ComputePotentialMinSolution.ir
// for -fcikernel and -c see https://stackoverflow.com/questions/float/metallib-error-reading-module-invalid-bitcode-signature
// 2) xcrun -sdk macosx metallib -o ComputePotentialMinSolution.metallib ComputePotentialMinSolution.ir 
// one-liner compilation... : 
// xcrun -sdk macosx metal -fcikernel ./shaders/ComputePotentialMinSolution.metal -c -o ./shaders/ComputePotentialMinSolution.ir && xcrun -sdk macosx metallib -o ./shaders/ComputePotentialMinSolution.metallib ./shaders/ComputePotentialMinSolution.ir

#include <metal_stdlib>
using namespace metal;





kernel void ComputePotentialMinSolution(
    device float* A [[buffer(0)]], // Transposed A, size M*N
    device float* b [[buffer(1)]],            // Vector b, size N
    device float* result [[buffer(2)]],       // Result vector, size M
    constant uint& N [[buffer(3)]],           // Size N
    constant uint& M [[buffer(4)]],           // Size M
    constant float& epsilon [[buffer(5)]],    // epsilon precision
    uint j [[thread_position_in_grid]]
) {
    float epsilonTerm;
    float col;
    if (j < M) {
        
        col = -FLT_MAX;
        for (unsigned int i = 0; i < N; ++i){

                
                if (abs(A[j*N +i]-b[i]) <= fmax(epsilon * max(abs(A[j*N +i]), abs(b[i])), 1e-9)) {
                    epsilonTerm =  0.0; // Return 0 if they are approximately equal
                }
                else 
                {
                    epsilonTerm =  A[j*N +i] < b[i] ? b[i] : 0.0;
                }

                col =fmax(epsilonTerm, col);

            if(col > 1.0 - epsilon)
                break;
        }

        result[j] = col;
    }
}
