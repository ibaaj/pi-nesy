//
//  ComputeNabla.hpp
//


#include <thread>
#include <vector>
#include <tuple>
#include <cfloat>

#define CL_HPP_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define MAX_SOURCE_SIZE (0x100000)
#include <CL/opencl.hpp>

class ComputePotentialMinSolution {
public:
    ~ComputePotentialMinSolution();
    void init_with_device();
    void prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& b_in, float nabla_in);
    void sendComputeCommand();
    std::vector<float> getResults();
    void showResults();
    void verifyResults();
    bool debug = false;
    
    unsigned int N = 0;
    unsigned int M = 0;
    float epsilon = 0.0;
    std::vector<float> output;
    float nabla = 0.0;
    std::vector<float> flatTransA;
    std::vector<float> b;

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Buffer bufferA, bufferB, bufferResult;

    std::vector<float> cpu_ComputePotentialMinSolution();
};

