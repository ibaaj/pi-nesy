//
//  NablaChunk.hpp
//

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <thread>
#include <vector>
#include <tuple>

#include "Tools.hpp"
#include "SingleThread.hpp"

class NablaChunk{
public:
    ~NablaChunk(); 

    bool debug = false;

    unsigned int N = 0;
    unsigned int M = 0;
    float epsilon = 0.0;
    float nabla;
    std::vector<float> res;

    MTL::Device* m_device =  nullptr;
    MTL::ComputePipelineState *m_computeNablaForFixedIJWithChunk_function_pso =  nullptr;
    MTL::CommandQueue *m_command_queue =  nullptr;
    
    MTL::Buffer *m_buffer_A_chunk =  nullptr;
    MTL::Buffer *m_buffer_b =  nullptr;
    MTL::Buffer *m_buffer_result =  nullptr;
    
    void init_with_device(MTL::Device*);
    void prepare_data(float* A, float* b);
    void send_compute_command();

   
    void getResults();
    

private:
    void encode_command(MTL::ComputeCommandEncoder* compute_encoder);
    MTL::Fence* m_fence = nullptr; 

    

};



float cpu_sigma_epsilon(float u, float v, float w);
std::tuple<size_t, float> cpu_computeNablaForFixedIJ(const std::vector<float>& colA_j, const std::vector<float>& b, float A_ij, float b_i, size_t j, const float epsilon) ;

void cpu_ComputeNablaFixedIChunk(const std::vector<std::vector<float>>& A_chunk, 
                             const std::vector<float>& b, 
                             std::vector<float>& resultsWithIndicesChunk, 
                             float b_i, size_t i, 
                             const std::vector<size_t>& chunkColumnIndices, 
                             float epsilon);


void ComputeNablaByChunk( float* A_flat, 
                         float* b, 
                        size_t N, 
                        size_t M, 
                        std::vector<float>& nablaResults,
                        float epsilon, 
                        bool isCPU) ;
void nabla_chunk_verify_results( float* A_flat, 
                         float* b, 
                        size_t N, 
                        size_t M, 
                        float epsilon
                        );