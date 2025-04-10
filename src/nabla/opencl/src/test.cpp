//
//  test.cpp
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <iostream>
#include <limits>
#include "Computations/ComputeNabla.hpp"
#include "Computations/ComputeMinMax.hpp"
#include "Computations/ComputePotentialMinSolution.hpp"



int main(int argc, const char * argv[]) {
    
    const unsigned int N = 200;
    const unsigned int M = 30;
    const float epsilon = std::numeric_limits<float>::epsilon();


    int i = 0; 
    while(i < 100){

        std::vector<std::vector<float>> A;
        std::vector<float> b;
        std::vector<float> x;

        std::vector<float> tmp;

        for(unsigned long i = 0; i < N; ++i){
            tmp.clear();
            for(unsigned long j = 0; j < M; ++j){
                tmp.emplace_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            A.emplace_back(tmp);
            b.emplace_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }

        for(unsigned long j = 0; j < M; ++j){
            x.emplace_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }

        
        
        ComputeNabla* nabla = new ComputeNabla();
        
        nabla->N = N;
        nabla->M = M;
        nabla->epsilon = epsilon;
        nabla->debug = true;
        
        nabla->init_with_device();
        nabla->prepareData(A,b);
        nabla->sendComputeCommand();

        float nabla_result = nabla->nabla;
        std::vector<float> res = nabla->nablaI;

        std::cout << "Execution ComputeNabla finished." << std::endl;
        
        delete nabla;

        ComputeMinMax* minmax = new ComputeMinMax();
        minmax->init_with_device();
        
        minmax->N = N;
        minmax->M = M;
        minmax->epsilon = epsilon;
        minmax->debug = true;
        
        minmax->prepareData(A,x);
        minmax->sendComputeCommand();

        std::cout << "Execution ComputeMinMax finished." << std::endl;

        delete minmax;
        

        ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
        sol->init_with_device();
        
        sol->N = N;
        sol->M = M;
        sol->epsilon = epsilon;
        sol->debug = true;
        
        sol->prepareData(A,b,nabla_result);
        sol->sendComputeCommand();

        std::cout << "Execution ComputeMinSol finished." << std::endl;
        
        delete sol;

        

    i++;
    }
    
    
    
    return 0;
}
