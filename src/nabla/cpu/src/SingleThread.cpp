//
// single_thread.cpp
// 
#include <vector>
#include <limits>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <tuple>

using namespace std;


std::vector<float> single_thread_computeMinMax(const float* A, const float* x, int N, int M, float epsilon)
{    

    std::vector<float> output_single_thread;
    float row;
    float maxTerm;
    for (int i = 0; i < N; ++i){
        row = FLT_MAX;
        for (int j = 0; j < M; ++j){
            maxTerm = std::fmax(A[i*M+j], x[j]);
            row = std::fmin(maxTerm, row);
            if (row < epsilon)
                break;
        }
        output_single_thread.emplace_back(row);
    }
    return output_single_thread;
}


void verify_results_computeMinMax(std::vector<float> output, std::vector<float> output_single_thread, float epsilon)
{

    bool diffFound = false;
    for(int i = 0; i < output.size(); ++i){

        std::cout << " i =" << i << " CPU_multithread :::: " << output[i] <<  " vs  single_thread :::: " << output_single_thread[i] << std::endl;

        if(abs(output[i] - output_single_thread[i]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }


    }
    if(!diffFound){
        std::cout << "Compute results as expected\n";
    }
    else
    {
        std::cerr << "something is wrong." << std::endl;
        std::exit(-1);
    }
}


std::tuple<float,std::vector<float>> single_thread_computeChebyshevDistance(const float* A, const float* b, int N, int M, float epsilon)
{
    

    float nablasingle_thread = 0.0f; 
    std::vector<float> nabla_isingle_thread;
    float nablai;
    float nablaij;

    
    
    for (int i = 0; i < N; ++i){
        nablai = 1.0f;
        for (int j = 0; j < M; ++j){
            nablaij = std::fmax(A[i*M+j] - b[i], 0.0f);
            for (int k = 0; k < N; ++k){
                nablaij = std::fmax(nablaij, std::fmin( std::fmax(b[k] - b[i], 0)/2.0f, std::fmax(b[k] - A[k*M +j], 0)     ) );
                if(nablaij > 1.0 - epsilon)
                    break;
            }
            nablai = fmin(nablai, nablaij);

            if (nablai < epsilon)
            break;
        }
        nablasingle_thread = fmax(nablasingle_thread, nablai);
        nabla_isingle_thread.emplace_back(nablai);
    }
    return std::make_tuple(nablasingle_thread, nabla_isingle_thread);
}



void verify_results_single_thread_computeChebyshevDistance(float nabla, std::vector<float> nabla_i, float nablasingle_thread, std::vector<float> nabla_isingle_thread, float epsilon){

    std::cout <<"Nabla single_thread:" << nablasingle_thread << std::endl;
    std::cout <<"Nabla CPU_multithread:" << nabla << std::endl;

    bool diffFound = false;
    for(int i = 0; i < nabla_i.size(); ++i){

        std::cout << "nabla CPU_multithread i =" << i << " :::: " << nabla_i[i] <<  " vs nabla single_thread i =" << i << " :::: " << nabla_isingle_thread[i] << std::endl;

        if(abs(nabla_i[i] - nabla_isingle_thread[i]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }


    }
   if(!diffFound){
        std::cout << "Compute results as expected\n";
    }
    else
    {
        std::cerr << "something is wrong." << std::endl;
        std::exit(-1);
    }
}


std::vector<float> single_thread_ComputeLowestApproxSolution(float* A, float* b, float nabla, int N, int M, float epsilon){

   
   float* data_ptr_buffer_transA = new float[N*M];
    float* data_ptr_buffer_b = new float[N];
    

    for (unsigned long i = 0; i < N; ++i) {
    for (unsigned long j = 0; j < M; ++j) {
        data_ptr_buffer_transA[j*N + i] = A[i*M + j];
        }
    }

     
    for(unsigned long i = 0; i < N; ++i)
    {
            data_ptr_buffer_b[i] = std::fmax(b[i] - nabla, 0);
    }


    

    std::vector<float> output_single_thread;
    float col;
    float epsilonTerm;
    for (int j = 0; j < M; ++j){
        col = -FLT_MAX;
        for (int i = 0; i < N; ++i){

                float tmp1 = std::fmax(std::fabs(data_ptr_buffer_transA[j*N +i]), std::fabs(data_ptr_buffer_b[i]));
                tmp1 = tmp1 * epsilon;
                tmp1 = std::fmax(tmp1, 1e-9);
                float epsilonTerm;

          


                if (fabs(data_ptr_buffer_transA[j*N +i] - data_ptr_buffer_b[i]) < tmp1) {
                    epsilonTerm =  0.0; // Return 0 if they are approximately equal
                }
                else 
                {
                    epsilonTerm =  data_ptr_buffer_transA[j*N +i] < data_ptr_buffer_b[i] ? data_ptr_buffer_b[i] : 0.0;
                }

                col = std::fmax(epsilonTerm, col);

                if(col > 1.0 - epsilon)
                    break;


        }
        output_single_thread.emplace_back(col);
    }

    delete[] data_ptr_buffer_transA;
    delete[] data_ptr_buffer_b;

    return output_single_thread;
}



void verify_results_ComputeLowestApproxSolution(std::vector<float> output, std::vector<float> output_single_thread, float epsilon){
    bool diffFound = false;
    for(int j = 0; j < output.size(); ++j){

        std::cout << " j =" << j << " CPU_multithread :::: " << output[j] <<  " vs  single_thread :::: " << output_single_thread[j] << std::endl;

        if(abs(output[j] - output_single_thread[j]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }


    }
    if(!diffFound){
        std::cout << "Compute results as expected\n";
    }
    else
    {
        std::cerr << "something is wrong but it's CPU_multithread float vs single_thread float comparison..." << std::endl;
        std::exit(-1);
    }
}
