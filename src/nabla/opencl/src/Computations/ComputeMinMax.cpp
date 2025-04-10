//
//  ComputeMinMax.cpp
//
//

#include <iostream>
#include "ComputeMinMax.hpp"





const char* kernelSourceComputeMinMax = R"kernel(
__kernel void ComputeMinMax_i(
    __global const float* A, 
    __global const float* x, 
    __global float* minMaxResults,
    const int N,
    const int M,
    const float epsilon) {
    int i = get_global_id(0);
    float maxTerm;
    float res = 1.0; 
    for (int j = 0; j < M; j++){
        maxTerm = fmax(A[i*M+j], x[j]);
        res = fmin(res, maxTerm);
        if (res < epsilon) 
            break;
    }
    minMaxResults[i] = res;
}
)kernel";


ComputeMinMax::~ComputeMinMax() {
    
    context = nullptr;
    kernel = nullptr;
    queue = nullptr;
    bufferA = nullptr;
    bufferX = nullptr;
    bufferResult = nullptr;
}

void ComputeMinMax::init_with_device(){
    
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
    sources.push_back({kernelSourceComputeMinMax, strlen(kernelSourceComputeMinMax)});
    cl::Program program(context, sources);
    try {
    program.build({device});
    } catch (const cl::BuildError& e) {
    std::cerr << "Build Error for device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    for (const auto& log : e.getBuildLog()) {
        std::cerr << "Build log:" << std::endl << log.second << std::endl;
    }
    throw; 
    }


    kernel = cl::Kernel(program, "ComputeMinMax_i");
    
};

void ComputeMinMax::prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& x_in) {
    if(N == 0 || M == 0) {
        std::cerr << "Error: N and M are not set." << std::endl;
        return;
    }
    A = A_in;
    x = x_in;

    std::vector<float> flattenedA; for (auto& subVec : A) flattenedA.insert(flattenedA.end(), subVec.begin(), subVec.end());

    const size_t bufferSizeA = N * M * sizeof(float);
    const size_t bufferSizeX = M * sizeof(float);
    const size_t bufferSizeResult = N * sizeof(float);

    bufferA = nullptr;
    bufferX = nullptr;
    bufferResult = nullptr;

    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeA);
    bufferX = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeX);
    bufferResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSizeResult);

    
    queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, bufferSizeA, flattenedA.data());
    queue.enqueueWriteBuffer(bufferX, CL_FALSE, 0, bufferSizeX, x.data());
}

void ComputeMinMax::sendComputeCommand() {
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferX);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, N);
    kernel.setArg(4, M);
    kernel.setArg(5, epsilon);
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(1));

    getResults();
    if(debug) {
        verifyResults();
       // showResults();
    }
}

std::vector<float> ComputeMinMax::getResults() {
    std::vector<float> results(N);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, N * sizeof(float), results.data());
    for(int i = 0; i < N; ++i) {
        output.emplace_back(results.at(i));
    }
    return results;
}

void ComputeMinMax::showResults(){
    std::cout <<"Vector Output" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << "i =" << i << " :::: " << output.at(i) << std::endl;
    }
}


std::vector<float> ComputeMinMax::cpu_computeMinMax(){
    

    std::vector<float> output_CPU;
    float row;
    float maxTerm;
    for (int i = 0; i < N; ++i){
        row = FLT_MAX;
        for (int j = 0; j < M; ++j){
            maxTerm = std::fmax(A[i][j], x[j]);
            row = std::fmin(maxTerm, row);
            if(row < epsilon)
                break;
        }
        output_CPU.emplace_back(row);
    }
    return output_CPU;
}



void ComputeMinMax::verifyResults(){

    if(!debug){
        std::cerr << "can't use verify results when debug is off." << std::endl;
        return;
    }


    std::vector<float> output_CPU =  ComputeMinMax::cpu_computeMinMax();


    bool diffFound = false;
    for(int i = 0; i < N; ++i){

        std::cout << " i =" << i << " GPU :::: " << output[i] <<  " vs  CPU :::: " << output_CPU[i] << std::endl;

        if(abs(output[i] - output_CPU[i]) > epsilon){
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
