//
//  ComputeNabla.cpp
//
//

#include <iostream>
#include "ComputeNabla.hpp"





const char* kernelSourceComputeNabla = R"kernel(
    __kernel void ComputeNabla_i(
        __global const float* A, 
        __global const float* b, 
        __global float* nablaResults,
        const int N,
        const int M,
        const float epsilon
    ) {
        int i = get_global_id(0);
        float nablaMin = 1.0;
        float temp;
        float nabla_ij;
        float diff;
        float sigmaEpsilon;
        
        for(int j = 0; j < M; ++j) {
            nabla_ij = 0.0;
            for(int k = 0; k < N; ++k) {
                diff = fmax(0, A[i*M+j] - b[i]);
                sigmaEpsilon = fmin(fmax(0, (b[k] - b[i]) / 2.0), fmax(0, b[k] - A[k*M+j]));
                temp = fmax(diff, sigmaEpsilon);
                nabla_ij = fmax(temp, nabla_ij);

                if(nabla_ij > 1.0 - epsilon)
                    break;

            }
            nablaMin = fmin(nablaMin, nabla_ij);
            if (nablaMin < epsilon)
                break;
        }

        nablaResults[i] = nablaMin;
    }
    )kernel";



ComputeNabla::~ComputeNabla() {
    context = nullptr;
    kernel = nullptr;
    queue = nullptr;
    bufferA = nullptr;
    bufferB = nullptr;
    bufferResult = nullptr;
    
}

void ComputeNabla::init_with_device(){
    
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
    sources.push_back({kernelSourceComputeNabla, strlen(kernelSourceComputeNabla)});
   
    cl::Program program(context, sources);
    try {
        program.build({device});
    } catch (const cl::Error&) {
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Error during kernel compilation:\n" << buildLog << std::endl;
        throw;
    }

    kernel = cl::Kernel(program, "ComputeNabla_i");
    
};

void ComputeNabla::prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& b_in) {
    if(N == 0 || M == 0) {
        std::cerr << "Error: N and M are not set." << std::endl;
        return;
    }
    A = A_in;
    b = b_in;

    std::vector<float> flattenedA; for (auto& subVec : A) flattenedA.insert(flattenedA.end(), subVec.begin(), subVec.end());

    const size_t bufferSizeA = N * M * sizeof(float);
    const size_t bufferSizeB = N * sizeof(float);
    const size_t bufferSizeResult = N * sizeof(float);

    bufferA = nullptr;
    bufferB = nullptr;
    bufferResult = nullptr;

    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeA);
    bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSizeB);
    bufferResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSizeResult);

    
    queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, bufferSizeA, flattenedA.data());
    queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, bufferSizeB, b.data());
}

void ComputeNabla::sendComputeCommand() {
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, N);
    kernel.setArg(4, M);
    kernel.setArg(5, epsilon);
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(1));

    getResults();
    if(debug) {
        verifyResults();
        //showResults();
    }
}

std::vector<float> ComputeNabla::getResults() {
    std::vector<float> results(N);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, N * sizeof(float), results.data());

    nabla = 0.0f;
    for(int i = 0; i < N; ++i) {
        nablaI.push_back(results[i]);
        nabla = std::max(results[i], nabla);
    }
    return results;
}

void ComputeNabla::showResults() {
    std::cout << "Nabla GPU: " << nabla << std::endl;
    for(int i = 0; i < N; ++i) {
        std::cout << "nabla GPU i = " << i << " :::: " << nablaI[i] << std::endl;
    }
}


std::tuple<float,std::vector<float>> ComputeNabla::cpu_computeChebyshevDistance(){

    float nablaCPU = 0.0f; 
    std::vector<float> nabla_iCPU;
    float nablai;
    float nablaij;
    
    for (int i = 0; i < N; ++i){
        nablai = 1.0f;
        for (int j = 0; j < M; ++j){
            nablaij = std::fmax(A[i][j] - b[i], 0.0f);
            for (int k = 0; k < N; ++k){
                nablaij = std::fmax(nablaij, std::fmin( std::fmax(b[k] - b[i], 0)/2.0f, std::fmax(b[k] - A[k][j], 0)     ) );
                if(nablaij > 1.0 - epsilon)
                break;
            }
            nablai = fmin(nablai, nablaij);
            if(nablai < epsilon)
                break;
        }
        nablaCPU = fmax(nabla, nablai);
        nabla_iCPU.emplace_back(nablai);
    }

    

    return std::make_tuple(nablaCPU, nabla_iCPU);
}



void ComputeNabla::verifyResults(){

    if(!debug){
        std::cerr << "can't use verify results when debug is off." << std::endl;
        return;
    }

    float nablaCPU = 0.0f; 
    std::vector<float> nabla_iCPU;

    std::tie(nablaCPU, nabla_iCPU) =  ComputeNabla::cpu_computeChebyshevDistance();

    std::cout <<"Nabla CPU:" << nablaCPU << std::endl;
    std::cout <<"Nabla GPU:" << nabla << std::endl;

    bool diffFound = false;
    for(int i = 0; i < N; ++i){

        std::cout << "nabla GPU i =" << i << " :::: " << nablaI[i] <<  " vs nabla CPU i =" << i << " :::: " << nabla_iCPU[i] << std::endl;

        if(abs(nablaI[i] - nabla_iCPU[i]) > epsilon){
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

