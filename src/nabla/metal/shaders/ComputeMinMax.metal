//
//  computeMinMax.metal
//
//  Created by Ismaïl Baaj on 02.22.24
//

// compilation
// 1) xcrun -sdk macosx metal -fcikernel ComputeMinMax.metal -c -o ComputeMinMax.ir
// for -fcikernel and -c see https://stackoverflow.com/questions/52992783/metallib-error-reading-module-invalid-bitcode-signature
// 2) xcrun -sdk macosx metallib -o ComputeMinMax.metallib ComputeMinMax.ir 
// one-liner compilation... : 
// xcrun -sdk macosx metal -fcikernel ComputeMinMax.metal -c -o ComputeMinMax.ir && xcrun -sdk macosx metallib -o ComputeMinMax.metallib ComputeMinMax.ir

#include <metal_stdlib>
using namespace metal;




kernel void computeMinMax(
    device float *A [[buffer(0)]], // Matrix A
    device float *x [[buffer(1)]], // Vector x
    device float *result [[buffer(2)]], // Output result
    constant uint &m [[buffer(3)]], // Number of columns in A and x
    constant float &epsilon [[buffer(4)]], // epsilon so that 1-e ~= 1, e~=0
    uint i [[thread_position_in_grid]]
) {

    float row = FLT_MAX;
    float maxTerm;
    for (uint j = 0; j < m; ++j) {
        maxTerm = max(A[i*m +j], x[j]);
        row = min(maxTerm, row);

        if(row < epsilon)
            break;
    }

    result[i] = row; 
}
