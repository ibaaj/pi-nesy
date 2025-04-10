//
//  ComputeMinMax.cpp
//
//

#include "ComputeMinMax.hpp"


ComputeMinMax::~ComputeMinMax() {
    if(m_buffer_A) m_buffer_A->release();
    if(m_buffer_x) m_buffer_x->release();
    if(m_buffer_result) m_buffer_result->release();
    if(m_command_queue) m_command_queue->release();
    if(m_computeMinMax_function_pso) m_computeMinMax_function_pso->release();
    if(m_device) m_device->release();
}


void ComputeMinMax::init_with_device(MTL::Device* device){
    m_device = device;
    NS::Error* error;
    NS::String* filePath = NS::String::string("./shaders/ComputeMinMax.metallib", NS::UTF8StringEncoding);
    auto default_library = m_device->newLibrary(filePath, &error);
    
    if(!default_library){
        std::cerr << "Failed to load default library ComputeMinMax.";
        return;
    }
    
    auto function_name = NS::String::string("computeMinMax", NS::ASCIIStringEncoding);
    auto computeMinMax_function = default_library->newFunction(function_name);
    
    if(!computeMinMax_function){
        std::cerr << "Failed to find the computeMinMax function.";
        return;
    }

    m_fence = createFenceWithRetries(m_device, MAX_RETRIES); ;
    if (!m_fence) {
        std::cerr << "Failed to create the fence.\n";
        return;
    }
    
    m_computeMinMax_function_pso = m_device->newComputePipelineState(computeMinMax_function, &error);

    if (!m_computeMinMax_function_pso) {
        std::cerr << "Failed to create PSO in ComputeMinMax." << std::endl;
        return;
    }

    m_command_queue = m_device->newCommandQueue();

    error = nullptr;
    filePath = nullptr;
    
};



void ComputeMinMax::prepare_data(float* A, float* x){
    if(N == 0 or M == 0){
        std::cerr << " error: N and M are not set." << std::endl;
        return;
    }

    const unsigned int buffer_size_A = N*M*sizeof(float);
    const unsigned int buffer_size_x = M*sizeof(float);
    const unsigned int buffer_size_result = N*sizeof(float);

    
    m_buffer_A = nullptr;
    m_buffer_x = nullptr;
    m_buffer_result = nullptr;

    
    m_buffer_A = createBufferWithRetries(m_device, buffer_size_A, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_x = createBufferWithRetries(m_device,buffer_size_x, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_result =createBufferWithRetries(m_device,buffer_size_result, MTL::ResourceStorageModeShared, MAX_RETRIES);

    if (!m_buffer_A or !m_buffer_x or !m_buffer_result) {
    std::cerr << "Failed to allocate buffer(s)." << std::endl;
        std::exit(-1);
    }

    float* data_ptr_buffer_A = (float*)m_buffer_A->contents();
    float* data_ptr_buffer_x = (float*)m_buffer_x->contents();
    
    

    for(unsigned long i = 0; i < N; ++i){
        for(unsigned long j = 0; j < M; ++j){
            data_ptr_buffer_A[i*M + j] = A[i*M +j];
        }
    }

    for(unsigned long j = 0; j < M; ++j){
        data_ptr_buffer_x[j] = x[j];
    }
    
}





void ComputeMinMax::send_compute_command(){
     MTL::CommandBuffer* command_buffer = createCommandBufferWithRetries(m_command_queue, MAX_RETRIES);

    if (!command_buffer) {
        std::cerr << "Failed to create command buffer after 50 retries.\n";
        return;
    }

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

void ComputeMinMax::encode_command(MTL::ComputeCommandEncoder* compute_encoder){
   

    compute_encoder->setComputePipelineState(m_computeMinMax_function_pso);
    compute_encoder->setBuffer(m_buffer_A, 0, 0);
    compute_encoder->setBuffer(m_buffer_x, 0, 1);
    compute_encoder->setBuffer(m_buffer_result, 0, 2);
    compute_encoder->setBytes(&M, sizeof(uint), 3);
    compute_encoder->setBytes(&epsilon, sizeof(float), 4);

    
    MTL::Size grid_size = MTL::Size(N, 1, 1);
    
    NS::UInteger thread_group_size_ = m_computeMinMax_function_pso->maxTotalThreadsPerThreadgroup();
    if(thread_group_size_ > N){
        thread_group_size_ = N;
    }
    
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);

    

   
}

void ComputeMinMax::getResults(){
    
    auto result = (float*) m_buffer_result->contents();

    for(unsigned long i = 0; i < N; ++i){
        output.push_back(result[i]);
    }    
}

void ComputeMinMax::showResults(){
    std::cout <<"Vector Output" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << "i =" << i << " :::: " << output.at(i) << std::endl;
    }
}


std::vector<float> ComputeMinMax::cpu_computeMinMax(){

    auto A = (float*) m_buffer_A->contents();
    auto x = (float*) m_buffer_x->contents();
    

    std::vector<float> output_CPU;
    float row;
    float maxTerm;
    for (int i = 0; i < N; ++i){
        row = FLT_MAX;
        for (int j = 0; j < M; ++j){
            maxTerm = std::fmax(A[i*M+j], x[j]);
            row = std::fmin(maxTerm, row);
            if (row < epsilon)
                break;
        }
        output_CPU.emplace_back(row);
    }
    return output_CPU;
}



void ComputeMinMax::verify_results(){

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
