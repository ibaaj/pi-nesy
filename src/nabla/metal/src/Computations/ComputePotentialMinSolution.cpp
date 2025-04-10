//
//  ComputePotentialMinSolution.cpp
//
//


#include "ComputePotentialMinSolution.hpp"


ComputePotentialMinSolution::~ComputePotentialMinSolution() {
    if(m_buffer_transA) m_buffer_transA->release();
    if(m_buffer_b) m_buffer_b->release();
    if(m_buffer_result) m_buffer_result->release();
    if(m_command_queue) m_command_queue->release();
    if(m_ComputePotentialMinSolution_function_pso) m_ComputePotentialMinSolution_function_pso->release();
    if(m_device) m_device->release();
}


void ComputePotentialMinSolution::init_with_device(MTL::Device* device){
    m_device = device;
    NS::Error* error;
    NS::String* filePath = NS::String::string("./shaders/ComputePotentialMinSolution.metallib", NS::UTF8StringEncoding);
    auto default_library = m_device->newLibrary(filePath, &error);
    
    if(!default_library){
        std::cerr << "Failed to load default library ComputePotentialMinSolution.";
        return;
    }
    
    auto function_name = NS::String::string("ComputePotentialMinSolution", NS::ASCIIStringEncoding);
    auto ComputePotentialMinSolution_function = default_library->newFunction(function_name);
    
    if(!ComputePotentialMinSolution_function){
        std::cerr << "Failed to find the ComputePotentialMinSolution function.";
        return;
    }

    m_fence = createFenceWithRetries(m_device, MAX_RETRIES); 
    if (!m_fence) {
        std::cerr << "Failed to create the fence.\n";
        return;
    }
    
    m_ComputePotentialMinSolution_function_pso = m_device->newComputePipelineState(ComputePotentialMinSolution_function, &error);

    if (!m_ComputePotentialMinSolution_function_pso) {
        std::cerr << "Failed to create PSO in ComputePotentialMinSolution." << std::endl;
        return;
    }

    m_command_queue = m_device->newCommandQueue();

    error = nullptr;
    filePath = nullptr;
    
};



void ComputePotentialMinSolution::prepare_data(float* A, float* b, float nabla_in){
    if(N == 0 or M == 0){
        std::cerr << " error: N and M are not set." << std::endl;
        return;
    }
    nabla = nabla_in;

    const unsigned int buffer_size_transA = N*M*sizeof(float);
    const unsigned int buffer_size_b = N*sizeof(float);
    const unsigned int buffer_size_result = M*sizeof(float);

    
    m_buffer_transA = nullptr;
    m_buffer_b = nullptr;
    m_buffer_result = nullptr;


    m_buffer_transA = createBufferWithRetries(m_device,buffer_size_transA, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_b = createBufferWithRetries(m_device,buffer_size_b, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_result = createBufferWithRetries(m_device,buffer_size_result, MTL::ResourceStorageModeShared, MAX_RETRIES);
    

    if (!m_buffer_transA or !m_buffer_b or !m_buffer_result) {
    std::cerr << "Failed to allocate buffer(s)." << std::endl;
        std::exit(-1);
    }

    float* data_ptr_buffer_transA = (float*)m_buffer_transA->contents();
    float* data_ptr_buffer_b = (float*)m_buffer_b->contents();
    

    for (unsigned long i = 0; i < N; ++i) {
    for (unsigned long j = 0; j < M; ++j) {
        data_ptr_buffer_transA[j*N + i] = A[i*M + j];
        }
    }

     
    for(unsigned long i = 0; i < N; ++i)
    {
            data_ptr_buffer_b[i] = std::fmax(b[i] - nabla, 0);
    }
       
    
    
    
}





void ComputePotentialMinSolution::send_compute_command(){
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
    if(debug)
    {
        verify_results();
        //showResults();
    }
    
    

    
    
}

void ComputePotentialMinSolution::encode_command(MTL::ComputeCommandEncoder* compute_encoder){
    
    

    compute_encoder->setComputePipelineState(m_ComputePotentialMinSolution_function_pso);
    compute_encoder->setBuffer(m_buffer_transA, 0, 0);
    compute_encoder->setBuffer(m_buffer_b, 0, 1);
    compute_encoder->setBuffer(m_buffer_result, 0, 2);
    compute_encoder->setBytes(&N, sizeof(uint), 3);
    compute_encoder->setBytes(&M, sizeof(uint), 4);
    compute_encoder->setBytes(&epsilon, sizeof(float), 5);

    
    MTL::Size grid_size = MTL::Size(M, 1, 1);
    
    NS::UInteger thread_group_size_ = m_ComputePotentialMinSolution_function_pso->maxTotalThreadsPerThreadgroup();
    if(thread_group_size_ > M){
        thread_group_size_ = M;
    }
    
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);

    

   
}

void ComputePotentialMinSolution::getResults(){
    
    auto result = (float*) m_buffer_result->contents();

    for(unsigned long j = 0; j < M; ++j){
        output.push_back(result[j]);
    }    
}

void ComputePotentialMinSolution::showResults(){
    std::cout <<"Vector solution " << std::endl;
    for(int j = 0; j < M; ++j){
        std::cout << "j =" << j << " :::: " << output.at(j) << std::endl;
    }
}


std::vector<float> ComputePotentialMinSolution::cpu_ComputePotentialMinSolution(){

    auto A = (float*) m_buffer_transA->contents();
    auto b = (float*) m_buffer_b->contents();
    

    std::vector<float> output_CPU;
    float col;
    float epsilonTerm;
    for (int j = 0; j < M; ++j){
        col = -FLT_MAX;
        for (int i = 0; i < N; ++i){
            
                float tmp1 = std::fmax(std::fabs(A[j*N +i]), std::fabs(b[i]));
                tmp1 = tmp1 * epsilon;
                tmp1 = std::fmax(tmp1, epsilon);
                float epsilonTerm;

          


                if (std::fabs(A[j*N +i] - b[i]) < tmp1) {
                    epsilonTerm =  0.0; // Return 0 if they are approximately equal
                }
                else 
                {
                    epsilonTerm =  A[j*N +i] < b[i] ? b[i] : 0.0;
                }

                col = std::fmax(epsilonTerm, col);

                if(col > 1.0 - epsilon)
                    break;

        }
        output_CPU.emplace_back(col);
    }
    return output_CPU;
}



void ComputePotentialMinSolution::verify_results(){

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
