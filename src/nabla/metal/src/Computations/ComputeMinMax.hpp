//
//  ComputeMinMax.hpp
//

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <thread>
#include <vector>
#include <tuple>
#include <iostream>

#include "Tools.hpp"

class ComputeMinMax{
public:
    ~ComputeMinMax(); 

    bool debug = false;

    unsigned int N = 0;
    unsigned int M = 0;
    float epsilon = 0.0;
    std::vector<float> output;

    MTL::Device* m_device = nullptr;
    MTL::ComputePipelineState *m_computeMinMax_function_pso = nullptr;
    MTL::CommandQueue *m_command_queue = nullptr;
    
    MTL::Buffer *m_buffer_A = nullptr;
    MTL::Buffer *m_buffer_x  = nullptr;
    MTL::Buffer *m_buffer_result = nullptr;
    
    void init_with_device(MTL::Device*);
    void prepare_data(float* A, float* b);
    void send_compute_command();

    void getResults();
    void showResults();
    void verify_results();
   

private:
    void encode_command(MTL::ComputeCommandEncoder* compute_encoder);
    std::vector<float> cpu_computeMinMax();
    MTL::Fence* m_fence = nullptr; 
    
    
};
