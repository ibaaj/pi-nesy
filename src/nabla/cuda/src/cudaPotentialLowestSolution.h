// cudaPotentialLowestSolution.h
#ifndef CUDA_POTENTIALLOWESTSOLUTION_H
#define CUDA_POTENTIALLOWESTSOLUTION_H

void PotentialLowestSolution(const float* A, const float* b, float* result, unsigned int N, unsigned int M, float epsilon);

#endif
