//
//  NablaChunk.cpp
//
//


#include "NablaChunk.hpp"


NablaChunk::~NablaChunk() {
    if (m_buffer_A_chunk) m_buffer_A_chunk->release();
    if (m_buffer_b) m_buffer_b->release();
    if (m_buffer_result) m_buffer_result->release();
    if (m_command_queue) m_command_queue->release();
    if (m_computeNablaForFixedIJWithChunk_function_pso) m_computeNablaForFixedIJWithChunk_function_pso->release();
    if (m_device) m_device->release();
}


void NablaChunk::init_with_device(MTL::Device* device){
    m_device = device;
    NS::Error* error;
    NS::String* filePath = NS::String::string("./shaders/NablaChunk.metallib", NS::UTF8StringEncoding);
    auto default_library = m_device->newLibrary(filePath, &error);
    
    if(!default_library){
        std::cerr << "Failed to load default library NablaChunk.";
        return;
    }
    
    auto function_name = NS::String::string("computeNablaForFixedIJWithChunk", NS::ASCIIStringEncoding);
    auto computeNablaForFixedIJWithChunk_function = default_library->newFunction(function_name);
    
    if(!computeNablaForFixedIJWithChunk_function){
        std::cerr << "Failed to find the computeNablaForFixedIJWithChunk function.";
        return;
    }

    m_fence = createFenceWithRetries(m_device, MAX_RETRIES); 

    if (!m_fence) {
        std::cerr << "Failed to create the fence.\n";
        return;
    }
    
    m_computeNablaForFixedIJWithChunk_function_pso = m_device->newComputePipelineState(computeNablaForFixedIJWithChunk_function, &error);

    if (!m_computeNablaForFixedIJWithChunk_function_pso) {
        std::cerr << "Failed to create PSO in computeNablaForFixedIJWithChunk." << std::endl;
        return;
    }

    m_command_queue = m_device->newCommandQueue();

    error = nullptr;
    filePath = nullptr;
    
}







void NablaChunk::prepare_data(float* A, float* b){
    if(N == 0 or M == 0){
        std::cerr << " error: N and M are not set." << std::endl;
        return;
    }

    const unsigned int buffer_size_A = N*M*sizeof(float);
    const unsigned int buffer_size_b = N*sizeof(float);
    const unsigned int buffer_size_result = N*sizeof(float);

    
    m_buffer_A_chunk = nullptr;
    m_buffer_b = nullptr;
    m_buffer_result = nullptr;

    
    m_buffer_A_chunk = createBufferWithRetries(m_device, buffer_size_A, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_b = createBufferWithRetries(m_device, buffer_size_b, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_result = createBufferWithRetries(m_device,buffer_size_result, MTL::ResourceStorageModeShared, MAX_RETRIES);
    

    if (!m_buffer_A_chunk or !m_buffer_b or !m_buffer_result) {
    std::cerr << "Failed to allocate buffer(s)." << std::endl;
        return;
    }

    float* data_ptr_buffer_A = (float*)m_buffer_A_chunk->contents();
    float* data_ptr_buffer_b = (float*)m_buffer_b->contents();
    
    

    for(unsigned long i = 0; i < N; ++i){
        for(unsigned long j = 0; j < M; ++j){
            data_ptr_buffer_A[i*M + j] = A[i*M +j];
        }
        data_ptr_buffer_b[i] = b[i];

    }
    
}





void NablaChunk::send_compute_command(){

     MTL::CommandBuffer* command_buffer = createCommandBufferWithRetries(m_command_queue, MAX_RETRIES);

    if (!command_buffer) {
        std::cerr << "Failed to create command buffer after 50 retries.\n";
        return;
    }


    MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
    // Check if compute_encoder is valid
    if (!compute_encoder) {
        std::cerr << "Failed to create compute encoder.\n";
        return; 
    }

    
    // Use the fence for synchronization
    compute_encoder->waitForFence(m_fence);
    encode_command(compute_encoder);
    compute_encoder->endEncoding();
    compute_encoder->updateFence(m_fence); 
    // Signal the fence once the work is encoded

    MTL::CommandBufferStatus status = command_buffer->status();
    if(debug)
    {
        std::cout << "CommandBuffer GPU status:" << status << std::endl;
    }
    
    command_buffer->commit();
    
    command_buffer->waitUntilCompleted();
 
    getResults();
    
  
    
}

void NablaChunk::encode_command(MTL::ComputeCommandEncoder* compute_encoder){
    
    

    compute_encoder->setComputePipelineState(m_computeNablaForFixedIJWithChunk_function_pso);
    compute_encoder->setBuffer(m_buffer_A_chunk, 0, 0);
    compute_encoder->setBuffer(m_buffer_b, 0, 1);
    compute_encoder->setBuffer(m_buffer_result, 0, 2);
    compute_encoder->setBytes(&N, sizeof(uint), 3);
    compute_encoder->setBytes(&M, sizeof(uint), 4);
    compute_encoder->setBytes(&epsilon, sizeof(float), 5);

    
    MTL::Size grid_size = MTL::Size(M, 1, 1);
    
    NS::UInteger thread_group_size_ = m_computeNablaForFixedIJWithChunk_function_pso->maxTotalThreadsPerThreadgroup();
    if(thread_group_size_ > M){
        thread_group_size_ = M;
    }
    
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);

    

   
}

void NablaChunk::getResults(){
    
    auto result = (float*) m_buffer_result->contents();

    for(unsigned long i = 0; i < N; ++i){
        res.push_back(result[i]);
    }    
}



// Function to compute sigma_epsilon
float cpu_sigma_epsilon(float u, float v, float w) {
    return std::fmin(std::fmax(w - u, 0.0f) / 2.0, std::fmax(w - v, 0.0f));
}

// Function to compute nabla_ijk for a fixed i and j, iterating over all k
std::tuple<size_t, float> cpu_computeNablaForFixedIJ(const std::vector<float>& colA_j, const std::vector<float>& b, float A_ij, float b_i, size_t j, const float epsilon) {
    float maxNabla_ij = 0.0f;
    for (size_t k = 0; k < b.size(); ++k) {
        float tmp = std::fmax(0.0f, A_ij - b_i);
        float nabla_ijk = std::fmax(tmp, cpu_sigma_epsilon(b_i, colA_j[k], b[k]));
        maxNabla_ij = std::fmax(maxNabla_ij, nabla_ijk);
        if (maxNabla_ij > 1.0 - epsilon)
            break;
    }
    return {j, maxNabla_ij}; // Return both j and nabla_ij
}

void cpu_ComputeNablaFixedIChunk(const std::vector<std::vector<float>>& A_chunk, 
                             const std::vector<float>& b, 
                             std::vector<float>& resultsWithIndicesChunk, 
                             float b_i, size_t i, 
                             const std::vector<size_t>& chunkColumnIndices, 
                             float epsilon) {
    for (size_t idx = 0; idx < chunkColumnIndices.size(); ++idx) {
        size_t j = chunkColumnIndices[idx];
        const auto& colA_j = A_chunk[idx];
        auto [index, value] = cpu_computeNablaForFixedIJ(colA_j, b, A_chunk[idx][i], b_i, j, epsilon);
        resultsWithIndicesChunk[idx] = value; // Note: Using idx as we're indexing within the chunk
    }
}





void ComputeNablaByChunk( float* A_flat, 
                        float* b, 
                        size_t N, 
                        size_t M, 
                        std::vector<float>& nablaResults,
                        float epsilon, 
                        bool isCPU = true) {
    const size_t chunkSize = 3; // Or any other size that suits your needs

    nablaResults.resize(N, std::numeric_limits<float>::max());

    std::vector<float> b_vec;
    for(int i = 0; i < N; ++i)
    {
        b_vec.emplace_back(b[i]);
    }

    for (size_t i = 0; i < N; ++i) {
        std::vector<float> resultsWithIndices(M, std::numeric_limits<float>::max()); // For the final aggregation

        for (size_t startCol = 0; startCol < M; startCol += chunkSize) {
            size_t endCol = std::min(startCol + chunkSize, M);
            std::vector<float> flat_A_chunk; 
            std::vector<std::vector<float>> A_chunk;
            std::vector<size_t> chunkColumnIndices;
            std::vector<float> tmp;
            
            for (size_t j = startCol; j < endCol; ++j) {
                chunkColumnIndices.push_back(j);
                tmp.clear();
                for (size_t k = 0; k < N; ++k) {
                    // Push back each element in the column `j` for the chunk
                    flat_A_chunk.push_back(A_flat[k * M + j]);
                    tmp.emplace_back(A_flat[k * M + j]);

                }
                A_chunk.emplace_back(tmp);
            }
            

            std::vector<float> resultsWithIndicesChunk(chunkColumnIndices.size(), std::numeric_limits<float>::max()); // For the chunk results
            if (isCPU){
                cpu_ComputeNablaFixedIChunk(A_chunk, b_vec, resultsWithIndicesChunk, b[i], i, chunkColumnIndices, epsilon);
            }
            else {
                NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
                MTL::Device* device = MTL::CreateSystemDefaultDevice();
                NablaChunk* nabla_chunk = new NablaChunk();
                nabla_chunk->init_with_device(device);
                nabla_chunk->N = N;
                nabla_chunk->M = endCol - startCol;
                nabla_chunk->epsilon = epsilon;
                nabla_chunk->debug = true;
                
                nabla_chunk->prepare_data(flat_A_chunk.data(),b);
                nabla_chunk->send_compute_command();

                resultsWithIndicesChunk = nabla_chunk->res;

                
                p_pool->release();
                delete nabla_chunk;
            }
                

            // Combine chunk results back into the main results vector
            for (size_t idx = 0; idx < chunkColumnIndices.size(); ++idx) {
                size_t j = chunkColumnIndices[idx];
                resultsWithIndices[j] = resultsWithIndicesChunk[idx];
            }
        }

        // Compute the minimum nabla_i for the current row i
        float nabla_i = *std::min_element(resultsWithIndices.begin(), resultsWithIndices.end());
        nablaResults[i] = nabla_i;
    }
}





void nabla_chunk_verify_results(float* A_flat, 
                        float* b, 
                        size_t N, 
                        size_t M, 
                        float epsilon
                        )
                        {

    

    std::vector<float> nabla_iGPU;
    std::vector<float> nabla_iCPU;
    

    ComputeNablaByChunk(A_flat, b, 
                        N, 
                        M, 
                        nabla_iGPU,
                        epsilon,
                         true);

    ComputeNablaByChunk(A_flat, b, 
                        N, 
                        M, 
                        nabla_iCPU,
                        epsilon,
                         true);
    
    float nablaGPU = *std::max_element(nabla_iGPU.begin(), nabla_iGPU.end());
    float nablaCPU = *std::max_element(nabla_iCPU.begin(), nabla_iCPU.end());



    std::cout <<"Nabla CPU:" << nablaCPU << std::endl;
    std::cout <<"Nabla GPU:" << nablaGPU << std::endl;

    bool diffFound = false;
    for(int i = 0; i < N; ++i){

        std::cout << "nabla GPU i =" << i << " :::: " << nabla_iGPU[i] <<  " vs nabla CPU i =" << i << " :::: " << nabla_iCPU[i] << std::endl;

        if(abs(nabla_iGPU[i] - nabla_iCPU[i]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }
    }
   if(!diffFound){
        std::cout << "Compute results as expected GPU vs CPU\n";
    }
    else
    {
        std::cerr << "something is wrong." << std::endl;
        std::exit(-1);
    }


   float nabla_CPU2;
   std::vector<float> nablai_CPU2;
    std::tie(nabla_CPU2, nablai_CPU2) = single_thread_computeChebyshevDistance(A_flat, b, N, M, epsilon);
    

    std::cout <<"Nabla CPU by chunk:" << nablaCPU << std::endl;
    std::cout <<"Nabla CPU monothread:" <<nabla_CPU2 << std::endl;


     diffFound = false;
    for(int i = 0; i < N; ++i){

        std::cout << "nabla CPU chunk i =" << i << " :::: " << nabla_iCPU[i] <<  " vs nabla monothread i =" << i << " :::: " << nablai_CPU2[i] << std::endl;

        if(abs(nabla_iCPU[i] - nablai_CPU2[i]) > epsilon){
            std::cout << "big difference..." << std::endl;
            diffFound = true;
        }
    }
   if(!diffFound){
        std::cout << "Compute results as expected CPU by chunk vs CPU monothread\n";
    }
    else
    {
        std::cerr << "something is wrong." << std::endl;
        std::exit(-1);
    }

}


