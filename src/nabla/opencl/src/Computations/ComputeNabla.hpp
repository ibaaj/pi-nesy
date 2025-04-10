//
//  ComputeNabla.hpp
//


#include <thread>
#include <vector>
#include <tuple>


#define CL_HPP_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define MAX_SOURCE_SIZE (0x100000)
#include <CL/opencl.hpp>

class ComputeNabla {
public:
    ~ComputeNabla();
    void init_with_device();
    void prepareData(const std::vector<std::vector<float>>& A_in, const std::vector<float>& b_in);
    void sendComputeCommand();
    std::vector<float> getResults();
    void showResults();
    void verifyResults();
    bool debug = false;
    int N, M;
    float nabla;
    std::vector<float> nablaI;
    std::vector<std::vector<float>> A;
    std::vector<float> b;
    float epsilon;

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Buffer bufferA, bufferB, bufferResult;

    std::tuple<float, std::vector<float>> cpu_computeChebyshevDistance();
};

