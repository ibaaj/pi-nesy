#include <metal_stdlib>
using namespace metal;

// Function to compute sigma_epsilon, adapted for Metal
float sigmaEpsilon(float u, float v, float w) {
    return fmin(fmax(w - u, 0.0) / 2.0, fmax(w - v, 0.0));
}

// Kernel function that correctly includes `j` from chunkColumnIndices
kernel void computeNablaForFixedIJWithChunk(
    device float* flat_A_chunk, // Flattened chunk of A
    device float* b, // Vector b, with the same number of rows as each column in A_chunk
    device size_t* chunkColumnIndices, // Actual column indices in the original matrix A
    device float* maxNabla_ij_values, // Output buffer for maxNabla_ij values
    constant float& b_i, // The i-th element of b, for a fixed row i
    constant float& epsilon, // Epsilon value for computation
    constant size_t& numRows, // Number of rows in A_chunk and b
    uint id [[thread_position_in_grid]]) // Thread ID, each thread handles one column of A_chunk
{
    size_t j = chunkColumnIndices[id]; // `j` corresponds to the actual column index in the original A
    float maxNabla_ij = 0.0;

    // Iterate over each row in the column and corresponding b element
    for (size_t k = 0; k < numRows; ++k) {
        float A_ij = flat_A_chunk[id * numRows + k]; // Accessing the k-th row in the j-th column of the chunk
        float tmp = fmax(0.0, A_ij - b_i);
        float nabla_ijk = fmax(tmp, sigmaEpsilon(b_i, A_ij, b[k]));
        maxNabla_ij = fmax(maxNabla_ij, nabla_ijk);
        
        // Break early if maxNabla_ij exceeds threshold
        if (maxNabla_ij > 1.0 - epsilon) break;
    }

    // Write the result for this column
    maxNabla_ij_values[id] = maxNabla_ij;
}
