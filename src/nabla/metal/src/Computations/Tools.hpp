#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <thread>
#include <vector>
#include <tuple>

#include <iostream>
#include <chrono>
#include <thread>

#define MAX_RETRIES 50
#define WAIT_TIME 500

MTL::Fence* createFenceWithRetries(MTL::Device* device, int maxRetries);
MTL::Buffer* createBufferWithRetries(MTL::Device* device, unsigned int size, MTL::ResourceOptions options, int maxRetries);
MTL::CommandBuffer* createCommandBufferWithRetries(MTL::CommandQueue* commandQueue, int maxRetries);

#endif // TOOLS_HPP
