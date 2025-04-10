#include "Tools.hpp"






MTL::Fence* createFenceWithRetries(MTL::Device* device, int maxRetries) {
    MTL::Fence* fence = nullptr;
    int retries = 0;

    while (retries < maxRetries) {
        fence = device->newFence();
        if (fence) {
            return fence; // Fence created successfully
        } else {
            std::cerr << "Failed to create fence. Retrying...\n";
            // Implement backoff or wait strategy here if needed
            std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
            retries++;
        }
    }

    // Log an error if max retries reached without success
    if (retries == maxRetries) {
        std::cerr << "Failed to create fence after " << maxRetries << " retries.\n";
    }

    return nullptr; // Indicate failure
}

MTL::Buffer* createBufferWithRetries(MTL::Device* device, unsigned int size, MTL::ResourceOptions options, int maxRetries) {
    MTL::Buffer* buffer = nullptr;
    int retries = 0;

    while (retries < maxRetries) {
        buffer = device->newBuffer(size, options);
        if (buffer) {
            return buffer; // Buffer created successfully
        } else {
            std::cerr << "Failed to create buffer. Retrying...\n";
            // Implement backoff or wait strategy here if needed
            std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
            retries++;
        }
    }

    // Log an error if max retries reached without success
    if (retries == maxRetries) {
        std::cerr << "Failed to create buffer after " << maxRetries << " retries.\n";
    }

    return nullptr; // Indicate failure
}

MTL::CommandBuffer* createCommandBufferWithRetries(MTL::CommandQueue* commandQueue, int maxRetries) {
    MTL::CommandBuffer* commandBuffer = nullptr;
    int retries = 0;

    while (retries < maxRetries) {
        commandBuffer = commandQueue->commandBuffer();
        if (commandBuffer) {
            return commandBuffer; // Command buffer created successfully
        } else {
            std::cerr << "Failed to create command buffer. Retrying...\n";
            // Implement backoff or wait strategy here if needed
            std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
            retries++;
        }
    }

    // Log an error if max retries reached without success
    if (retries == maxRetries) {
        std::cerr << "Failed to create command buffer after " << maxRetries << " retries.\n";
    }

    return nullptr; // Indicate failure
}