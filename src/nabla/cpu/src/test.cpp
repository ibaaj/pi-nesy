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
#include <iostream>

#include "Computations.hpp"
#include "SingleThread.hpp"

int main(int argc, const char* argv[]) {
    const unsigned int N = 200;
    const unsigned int M = 30;
    const float epsilon = 0.000001;

    srand(static_cast<unsigned>(time(nullptr)));

    for (int k = 0; k < 100; ++k) {
        
        std::vector<float> result_nabla(N), result_minmax(N), result_sol(M);

        std::vector<std::vector<float>> A;
        std::vector<float> b;
        std::vector<float> x;
        std::vector<float> flat_A;
        std::vector<float> tmp;
        float val;

        // Populate A with random values
        for(int i = 0; i < N; ++i){
            tmp.clear();
            for(int j = 0; j < M; ++j){
                val = static_cast<float>(rand()) / RAND_MAX;
                tmp.emplace_back(val);
                flat_A.emplace_back(val);
            }
            A.emplace_back(tmp);
            b.emplace_back(static_cast<float>(rand()) / RAND_MAX);
        }

        for(int j = 0; j < M; ++j){
            x.emplace_back(static_cast<float>(rand()) / RAND_MAX);
        }

        
        //  Multi-threaded computations
        ComputeNabla(A, b, result_nabla, epsilon);
        
        // Calculate nabla_single_thread_multithreaded as the maximum value of result_nabla
        float nabla_single_thread_multithreaded = *std::max_element(result_nabla.begin(), result_nabla.end());

        ComputeMinMax(A, x, result_minmax, epsilon);
        LowestApproxSolution(A, b, result_sol, nabla_single_thread_multithreaded, epsilon);

        // CPU computations
        std::tuple<float, std::vector<float>> single_thread_results = single_thread_computeChebyshevDistance(flat_A.data(), b.data(), N, M, epsilon);
        float nabla_cpu = std::get<0>(single_thread_results);
        std::vector<float> nabla_iCPU = std::get<1>(single_thread_results);

        auto result_minmax_CPU = single_thread_computeMinMax(flat_A.data(), x.data(), N, M, epsilon);
        auto result_sol_CPU = single_thread_ComputeLowestApproxSolution(flat_A.data(), b.data(), nabla_single_thread_multithreaded, N, M, epsilon);

        // Verification
        verify_results_single_thread_computeChebyshevDistance(nabla_single_thread_multithreaded, result_nabla, nabla_cpu, nabla_iCPU, epsilon);
        verify_results_computeMinMax(result_minmax, result_minmax_CPU, epsilon);
        verify_results_ComputeLowestApproxSolution(result_sol, result_sol_CPU, epsilon);

        std::cout << "Iteration " << k << " finished." << std::endl;
    }

    return 0;
}
