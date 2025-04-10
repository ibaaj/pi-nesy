// LowestApproxSolution.cpp
#include "LowestApproxSolution.hpp"
#include <cmath> 
#include "cudaPotentialLowestSolution.h" 

void LowestApproxSolution(const float* A, const float* b, float* result, float nabla, int N, int M, float epsilon) {
    
    float* modified_b = new float[N];

    // Modify b as per the formula: max(b_i - nabla, 0)
    for (int i = 0; i < N; ++i) {
        modified_b[i] = std::fmax(b[i] - nabla, 0.0);
    }

    
    PotentialLowestSolution(A, modified_b, result, N, M, epsilon);

    
    delete[] modified_b;
}
