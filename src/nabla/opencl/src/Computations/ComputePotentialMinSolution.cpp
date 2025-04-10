//
//  ComputePotentialMinSolution.cpp
//
//

#include <iostream>
#include "ComputePotentialMinSolution.hpp"



const char* kernelSourceComputePotentialMinSolution = R"kernel(
__kernel void ComputePotentialMinSolution_j(
    __global const float* A, 
    __global const float* b, 
    __global float* sol,
    const int N,
    const int M,
    const float epsilon) {
    int j = get_global_id(0);
    float col = -FLT_MAX;
    float epsilonTerm;
    float tmp1;
    
    for (int i = 0; i < N; ++i){
        tmp1 = fmax(fabs(A[j*N +i]), fabs(b[i]));
        tmp1 = tmp1*epsilon;
        tmp1 = fmax(tmp1, 1e-9);
        
            if (fabs(A[j*N +i] - b[i]) <= tmp1) {
                epsilonTerm =  0.0; // Return 0 if they are approximately equal
            }
            else 
            {   
                if(A[j*N +i] < b[i])
                {
                    epsilonTerm = b[i];
                }
                else{
                    epsilonTerm = 0.0;
                }
            }

            col = fmax(epsilonTerm, col);

        if(col > 1.0 - epsilon)
            break;
    }
    sol[j] = col;
}
)kernel";


ComputePotentialMinSolution::~ComputePotentialMinSolution() {
    context = nullptr;
    kernel = nullptr;
    queue = nullptr;
    bufferA = nullptr;
    bufferB = nullptr;
    bufferResult = nullptr;
}

void ComputePotentialMinSolution::init_with_device(){
    // Setup OpenCL environment
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL GPU devices found, checking for CPU devices.");
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found.");
        }
    }

    auto device = devices.front();
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    
    cl::Program::Sources sources;
    sources.push_back({kernelSourceComputePotentialMinSolution, strlen(kernelSourceComputePotentialMinSolution)});
    cl::Program program(context, sources);
    program.build({device});

    kernel = cl::Kernel(program, "ComputePotentialMinSolution_j");
    
};

void ComputePotentialMinSolution::prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& b_in,  float nabla_in) {
    if(N == 0 || M == 0) {
        std::cerr << "Error: N and M are not set." << std::endl;
        return;
    }
     for (unsigned long j = 0; j < M; ++j) {
        for (unsigned long i = 0; i < N; ++i) {
            flatTransA.emplace_back(A_in[i][j]);
        }
    }
    b = b_in;
    nabla = nabla_in;

    for(int i = 0; i < N; ++i){
        b[i] = std::fmax(b[i] - nabla, 0);
    }

    const size_t bufferSizeA = N * M * sizeof(float);
    const size_t bufferSizeB = N * sizeof(float);
    const size_t bufferSizeResult = M * sizeof(float);

    bufferA = nullptr;
    bufferB = nullptr;
    bufferResult = nullptr;

    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeA);
    bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeB);
    bufferResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSizeResult);

    
    queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, bufferSizeA, flatTransA.data());
    queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, bufferSizeB, b.data());
}

void ComputePotentialMinSolution::sendComputeCommand() {
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, N);
    kernel.setArg(4, M);
    kernel.setArg(5, epsilon);
    
   
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M), cl::NDRange(1));

    getResults();
    if(debug) {
        verifyResults();
       // showResults();
    }
}

std::vector<float> ComputePotentialMinSolution::getResults() {
    std::vector<float> results(M);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, M * sizeof(float), results.data());
    for(int i = 0; i < M; ++i) {
        output.emplace_back(results.at(i));
    }
    
    return results;
}

void ComputePotentialMinSolution::showResults(){
    std::cout <<"Vector Output" << std::endl;
    for(int i = 0; i < M; ++i){
        std::cout << "i =" << i << " :::: " << output.at(i) << std::endl;
    }
}



std::vector<float> ComputePotentialMinSolution::cpu_ComputePotentialMinSolution(){

    std::vector<float> output_CPU;
    float col;
    float epsilonTerm;
    for (int j = 0; j < M; ++j){
        col = -FLT_MAX;
        for (int i = 0; i < N; ++i){


                 float tmp1 = std::fmax(std::fabs(flatTransA[j*N +i]), std::fabs(b[i]));
                tmp1 = tmp1 * epsilon;
                tmp1 = std::fmax(tmp1, epsilon);
                float epsilonTerm;
            
                if (fabs(flatTransA[j*N +i] - b[i]) < tmp1) {
                    epsilonTerm =  0.0; 
                }
                else 
                {
                    epsilonTerm =  flatTransA[j*N +i] < b[i] ? b[i] : 0.0;
                }

                col = std::fmax(epsilonTerm, col);

            if(col > 1.0 - epsilon)
                break;

        }
        output_CPU.emplace_back(col);
    }
    return output_CPU;
}



void ComputePotentialMinSolution::verifyResults(){

    if(!debug){
        std::cerr << "can't use verify results when debug is off." << std::endl;
        return;
    }


    std::vector<float> output_CPU =  ComputePotentialMinSolution::cpu_ComputePotentialMinSolution();


    bool diffFound = false;
    for(int j = 0; j < M; ++j){

        std::cout << " j =" << j << " GPU :::: " << output[j] <<  " vs  CPU :::: " << output_CPU[j] << std::endl;

        if(abs(output[j] - output_CPU[j]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }


    }
    if(!diffFound){
        std::cout << "Compute results as expected\n";
    }
    else
    {
        std::cerr << "something is wrong but it's GPU float vs CPU float comparison..." << std::endl;
        //std::exit(-1);
    }
}
