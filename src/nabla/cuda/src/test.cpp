//
// test.cpp
//
#include <iostream>
#include <limits>
#include <vector>
#include <tuple>
#include <cmath>
#include <cstdlib> 
#include <ctime> 
#include <algorithm>

#include "cudaChebyshevDistanceNabla.h"
#include "cudaPotentialLowestSolution.h"
#include "cudaMinMax.h"
#include "LowestApproxSolution.hpp"
#include "SingleThread.hpp"

int main(int argc, const char* argv[]) {
    const unsigned int N = 200;
    const unsigned int M = 30;
    const float epsilon = 0.000001;

    srand(static_cast<unsigned>(time(nullptr)));

    for (int k = 0; k < 100; ++k) {
        
        

        float* result_nabla = new float[N];
        float* result_minmax = new float[N];
        float* result_sol = new float[M];

        float* flat_A = new float[N*M];
        float* b = new float[N];
        float* x = new float[M];

        std::vector<vector<float>> A_vec;
        std::vector<float> b_vec;
        std:vector<float> x_vec;
        std::vector<float> tmp;

        float val;
        for(int i = 0; i < N; ++i){
            tmp.clear();
            for(int j = 0; j < M; ++j){
                val = static_cast<float>(rand()) / RAND_MAX;
                flat_A[i*M+j] = val;
                tmp.emplace_back(val);
            }
            b[i] = static_cast<float>(rand()) / RAND_MAX;
            A_vec.emplace_back(tmp);
            b_vec.emplace_back(b[i]);
            
        }

        for(int j = 0; j < M; ++j){
           x[j] = static_cast<float>(rand()) / RAND_MAX;
           x_vec.emplace_back(x[j]);
        }

        ChebyshevDistanceNabla(flat_A, b, result_nabla, N, M, epsilon);

        std::vector<float> res_nabla;
        float nabla_gpu = 0.0f;
        for(int i = 0; i < N; ++i){
            res_nabla.push_back(result_nabla[i]);
            nabla_gpu = std::fmax(nabla_gpu,result_nabla[i]);
        }

        MinMax(flat_A, x, result_minmax, N, M, epsilon);

        std::vector<float> res_minmax;
        for(int i = 0; i < N; ++i){
            res_minmax.push_back(result_minmax[i]);
        }
        
        
        LowestApproxSolution(flat_A, b, result_sol, nabla_gpu, N, M, epsilon);

         std::vector<float> res_sol;
        for(int i = 0; i < M; ++i){
            res_sol.push_back(result_sol[i]);
        }

       // CPU computations
        std::tuple<float, std::vector<float>> single_thread_results = single_thread_computeChebyshevDistance(flat_A, b, N, M, epsilon);
        float nabla_cpu = std::get<0>(single_thread_results);
        std::vector<float> nabla_iCPU = std::get<1>(single_thread_results);

        auto result_minmax_CPU = single_thread_computeMinMax(flat_A, x, N, M, epsilon);
        auto result_sol_CPU = single_thread_ComputeLowestApproxSolution(flat_A, b, nabla_gpu, N, M, epsilon);

        // Verification
        verify_results_single_thread_computeChebyshevDistance(nabla_gpu, res_nabla, nabla_cpu, nabla_iCPU, epsilon);
        verify_results_computeMinMax(res_minmax, result_minmax_CPU, epsilon);
        verify_results_ComputeLowestApproxSolution( res_sol, result_sol_CPU, epsilon);


        std::cout << "Iteration " << k << " finished." << std::endl;








        delete[] result_nabla;
        delete[] result_minmax;
        delete[] result_sol;

        delete[] flat_A;
        delete[] b;
        delete[] x;
    }

    return 0; 
}



