//
// Computations.hpp
//
#ifndef MULTITHREADCOMPUTATIONS_H
#define MULTITHREADCOMPUTATIONS_H

#include <vector>
#include <cmath>
#include <algorithm>

void ComputeNabla(const std::vector<std::vector<float>>& A, 
                  const std::vector<float>& b, 
                  std::vector<float>& nablaResults, 
                  const float epsilon);

void ComputeMinMax(const std::vector<std::vector<float>>& A_in, const std::vector<float>& x_in, std::vector<float>& minMaxResults, float epsilon);

void ComputePotentialMinSolution(const std::vector<std::vector<float>>& A, 
                                 const std::vector<float>& b, 
                                 std::vector<float>& sol, 
                                 const float epsilon);

void LowestApproxSolution(const std::vector<std::vector<float>>& A, 
                                 const std::vector<float>& b, 
                                 std::vector<float>& res, 
                                 const float nabla,
                                 const float epsilon);
#endif 
