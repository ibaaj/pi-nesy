//
//  test.cpp
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <iostream>
#include <limits>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include "Computations/ComputeNabla.hpp"
#include "Computations/ComputeMinMax.hpp"
#include "Computations/ComputePotentialMinSolution.hpp"
#include "Computations/NablaChunk.hpp"


int main(int argc, const char * argv[]) {
    
    
    const float epsilon = std::numeric_limits<float>::epsilon();


    int i = 0; 
    while(i < 1000000){
        int N = 200;
        int M = 300;

        float* A = new float[N*M];
        float* b = new float[N];
        float* x = new float[M];


        for(unsigned long i = 0; i < N; ++i){
            for(unsigned long j = 0; j < M; ++j){
                A[i*M + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        for(unsigned long j = 0; j < M; ++j){
            x[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        
        ComputeNabla* nabla = new ComputeNabla();
        nabla->init_with_device(device);
        
        nabla->N = N;
        nabla->M = M;
        nabla->epsilon = epsilon;
        nabla->debug = true;
        
        nabla->prepare_data(A,b);
        nabla->send_compute_command();

        float nabla_result = nabla->nabla;

        std::cout << "Execution ComputeNabla finished." << std::endl;
        p_pool->release();
        delete nabla;

        p_pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();

        ComputeMinMax* minmax = new ComputeMinMax();
        minmax->init_with_device(device);
        
        minmax->N = N;
        minmax->M = M;
        minmax->epsilon = epsilon;
        minmax->debug = true;
        
        minmax->prepare_data(A,x);
        minmax->send_compute_command();

        std::cout << "Execution ComputeMinMax finished." << std::endl;
        p_pool->release();
        delete minmax;

        p_pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();

        ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
        sol->init_with_device(device);
        
        sol->N = N;
        sol->M = M;
        sol->epsilon = epsilon;
        sol->debug = true;
        
        sol->prepare_data(A,b,nabla_result);
        sol->send_compute_command();

        std::cout << "Execution ComputeMinSol finished." << std::endl;
        p_pool->release();
        delete sol;

         nabla_chunk_verify_results(A, b, N,  M,  epsilon);

        delete[] A;
        delete[] b;
        delete[] x;

    i++;
    }
    
    
    
    return 0;
}
