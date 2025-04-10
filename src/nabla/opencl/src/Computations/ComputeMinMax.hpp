//
//  ComputeMinMax.hpp
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

class ComputeMinMax {
public:
    ~ComputeMinMax();
    void init_with_device();
    void prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& x_in);
    void sendComputeCommand();
    std::vector<float> getResults();
    void showResults();
    void verifyResults();
    bool debug = false;
    int N, M;
    std::vector<std::vector<float>> A;
    std::vector<float> x;
    std::vector<float> output;
    float epsilon;

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Buffer bufferA, bufferX, bufferResult;

    std::vector<float> cpu_computeMinMax();
};

