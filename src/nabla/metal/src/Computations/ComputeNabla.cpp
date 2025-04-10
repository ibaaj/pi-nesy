//
//  ComputeNabla.cpp
//
//


#include "ComputeNabla.hpp"


ComputeNabla::~ComputeNabla() {
    if (m_buffer_A) m_buffer_A->release();
    if (m_buffer_b) m_buffer_b->release();
    if (m_buffer_result) m_buffer_result->release();
    if (m_command_queue) m_command_queue->release();
    if (m_computeChebyshevDistanceNabla_function_pso) m_computeChebyshevDistanceNabla_function_pso->release();
    if (m_device) m_device->release();
}


void ComputeNabla::init_with_device(MTL::Device* device){
    m_device = device;
    NS::Error* error;
    NS::String* filePath = NS::String::string("./shaders/ComputeNabla.metallib", NS::UTF8StringEncoding);
    auto default_library = m_device->newLibrary(filePath, &error);
    
    if(!default_library){
        std::cerr << "Failed to load default library ComputeNabla.";
        return;
    }
    
    auto function_name = NS::String::string("computeChebyshevDistanceNabla", NS::ASCIIStringEncoding);
    auto computeChebyshevDistanceNabla_function = default_library->newFunction(function_name);
    
    if(!computeChebyshevDistanceNabla_function){
        std::cerr << "Failed to find the computeChebyshevDistanceNabla function.";
        return;
    }

    m_fence = createFenceWithRetries(m_device, MAX_RETRIES); 

    if (!m_fence) {
        std::cerr << "Failed to create the fence.\n";
        return;
    }
    
    m_computeChebyshevDistanceNabla_function_pso = m_device->newComputePipelineState(computeChebyshevDistanceNabla_function, &error);

    if (!m_computeChebyshevDistanceNabla_function_pso) {
        std::cerr << "Failed to create PSO in ComputeChebyshevDistanceNabla." << std::endl;
        return;
    }

    m_command_queue = m_device->newCommandQueue();

    error = nullptr;
    filePath = nullptr;
    
}







void ComputeNabla::prepare_data(float* A, float* b){
    if(N == 0 or M == 0){
        std::cerr << " error: N and M are not set." << std::endl;
        return;
    }

    const unsigned int buffer_size_A = N*M*sizeof(float);
    const unsigned int buffer_size_b = N*sizeof(float);
    const unsigned int buffer_size_result = N*sizeof(float);

    
    m_buffer_A = nullptr;
    m_buffer_b = nullptr;
    m_buffer_result = nullptr;

    
    m_buffer_A = createBufferWithRetries(m_device, buffer_size_A, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_b = createBufferWithRetries(m_device, buffer_size_b, MTL::ResourceStorageModeShared, MAX_RETRIES);
    m_buffer_result = createBufferWithRetries(m_device,buffer_size_result, MTL::ResourceStorageModeShared, MAX_RETRIES);
    

    if (!m_buffer_A or !m_buffer_b or !m_buffer_result) {
    std::cerr << "Failed to allocate buffer(s)." << std::endl;
        return;
    }

    float* data_ptr_buffer_A = (float*)m_buffer_A->contents();
    float* data_ptr_buffer_b = (float*)m_buffer_b->contents();
    
    

    for(unsigned long i = 0; i < N; ++i){
        for(unsigned long j = 0; j < M; ++j){
            data_ptr_buffer_A[i*M + j] = A[i*M +j];
        }
        data_ptr_buffer_b[i] = b[i];

    }
    
}





void ComputeNabla::send_compute_command(){

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

void ComputeNabla::encode_command(MTL::ComputeCommandEncoder* compute_encoder){
    
    

    compute_encoder->setComputePipelineState(m_computeChebyshevDistanceNabla_function_pso);
    compute_encoder->setBuffer(m_buffer_A, 0, 0);
    compute_encoder->setBuffer(m_buffer_b, 0, 1);
    compute_encoder->setBuffer(m_buffer_result, 0, 2);
    compute_encoder->setBytes(&N, sizeof(uint), 3);
    compute_encoder->setBytes(&M, sizeof(uint), 4);
    compute_encoder->setBytes(&epsilon, sizeof(float), 5);

    
    MTL::Size grid_size = MTL::Size(N, 1, 1);
    
    NS::UInteger thread_group_size_ = m_computeChebyshevDistanceNabla_function_pso->maxTotalThreadsPerThreadgroup();
    if(thread_group_size_ > N){
        thread_group_size_ = N;
    }
    
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);

    

   
}

void ComputeNabla::getResults(){
    auto result = (float*) m_buffer_result->contents();

    nabla = 0.0f;
    for(unsigned long i = 0; i < N; ++i){
        nabla_i.push_back(result[i]);
        nabla = std::fmax(result[i], nabla);
    }    
}

void ComputeNabla::showResults(){
    std::cout <<"Nabla GPU:" << nabla << std::endl;
    for(int i = 0; i < N; ++i){

        std::cout << "nabla GPU i =" << i << " :::: " << nabla_i[i] << std::endl;
        
    }
}


std::tuple<float,std::vector<float>> ComputeNabla::cpu_computeChebyshevDistance(){

    auto A = (float*) m_buffer_A->contents();
    auto b = (float*) m_buffer_b->contents();
    

    float nablaCPU = 0.0f; 
    std::vector<float> nabla_iCPU;
    float nablai;
    float nablaij;
    
    for (int i = 0; i < N; ++i){
        nablai = 1.0f;
        for (int j = 0; j < M; ++j){
            nablaij = std::fmax(A[i*M+j] - b[i], 0.0f);
            for (int k = 0; k < N; ++k){
                nablaij = std::fmax(nablaij, std::fmin( std::fmax(b[k] - b[i], 0)/2.0f, std::fmax(b[k] - A[k*M +j], 0)     ) );

                 if(nablaij > 1.0 - epsilon)
                    break;
            }
            nablai = fmin(nablai, nablaij);

            if (nablai < epsilon)
            break;
        }
        nablaCPU = fmax(nabla, nablai);
        nabla_iCPU.emplace_back(nablai);
    }

    

    return std::make_tuple(nablaCPU, nabla_iCPU);
}



void ComputeNabla::verify_results(){

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

        std::cout << "nabla GPU i =" << i << " :::: " << nabla_i[i] <<  " vs nabla CPU i =" << i << " :::: " << nabla_iCPU[i] << std::endl;

        if(abs(nabla_i[i] - nabla_iCPU[i]) > epsilon){
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
