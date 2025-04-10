// single_thread.hpp
#ifndef single_thread_H
#define single_thread_H

#include <vector>
#include <limits>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <tuple>

using namespace std;

std::vector<float> single_thread_computeMinMax(const float* A, const float* x, int N, int M, float epsilon);

void verify_results_computeMinMax(std::vector<float> output, std::vector<float> output_single_thread, float epsilon);

std::tuple<float,std::vector<float>> single_thread_computeChebyshevDistance(const float* A, const float* b, int N, int M, float epsilon);

void verify_results_single_thread_computeChebyshevDistance(float nabla, std::vector<float> nabla_i, float nablasingle_thread, std::vector<float> nabla_isingle_thread, float epsilon);

std::vector<float> single_thread_ComputeLowestApproxSolution(float* A, float* b, float nabla, int N, int M, float epsilon);

void verify_results_ComputeLowestApproxSolution(std::vector<float> output, std::vector<float> output_single_thread, float epsilon);

#endif
