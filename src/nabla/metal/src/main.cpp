//
//  main.cpp
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION


#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstring>

#include "Computations/ComputeNabla.hpp"
#include "Computations/ComputeMinMax.hpp"
#include "Computations/ComputePotentialMinSolution.hpp"
#include "Computations/NablaChunk.hpp"


namespace py = pybind11;

const py::float_ DEFAULT_EPSILON = std::numeric_limits<float>::epsilon();





py::tuple nabla(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    float* A = new float[N*M];
    float* b = new float[N];
    std::vector<float> res;
    py::list respy;

    float nabla_result;

    int index = 0;
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        std::vector<float> rowVector;

        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            A[index] = value;
            index++;
        }
    }
    index = 0;
    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b[index] = value;
        index++;
    }
    

    
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device." << std::endl;
        std::exit(-1); 
    }

    ComputeNabla* nabla = new ComputeNabla();
    nabla->init_with_device(device);
    
    nabla->N = N;
    nabla->M = M;
    nabla->epsilon = epsilon;
    
    nabla->prepare_data(A,b);
    nabla->send_compute_command();

    nabla_result = nabla->nabla;
    res = nabla->nabla_i;
    p_pool->release();
    delete nabla;
    

    delete[] A;
    delete[] b;

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return py::make_tuple(nabla_result, respy);   
}


py::list minMax(py::list matrix, py::list x_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    float* A = new float[N*M];
    float* x = new float[M];
    std::vector<float> res;
    py::list respy;

    int index = 0;
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        std::vector<float> rowVector;

        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            A[index] = value;
            index++;
        }
    }
    index = 0;
    for(py::handle valueHandle : py::cast<py::list>(x_in)){
        float value = py::cast<float>(valueHandle);
        x[index] = value;
        index++;
    }
    

    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Failed to create Metal device." << std::endl;
        std::exit(-1);
    }

    ComputeMinMax* minmax = new ComputeMinMax();
    minmax->init_with_device(device);
    
    minmax->N = N;
    minmax->M = M;
    minmax->epsilon = epsilon;
    
    minmax->prepare_data(A,x);
    minmax->send_compute_command();

    res = minmax->output;

    p_pool->release();
    delete minmax;

        
        
    

    delete[] A;
    delete[] x;

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}

py::list potentialMinSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;
    
    float* A = new float[N*M];
    float* b = new float[N];
    std::vector<float> res;
    py::list respy;

    int index = 0;
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        std::vector<float> rowVector;

        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            A[index] = value;
            index++;
        }
    }
    index = 0;
    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b[index] = value;
        index++;
    }
    

     NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Failed to create Metal device." << std::endl;
        std::exit(-1);
    }

    ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
    sol->init_with_device(device);
    
    sol->N = N;
    sol->M = M;
    sol->epsilon = epsilon;

    

    
    
    sol->prepare_data(A,b, 0.0f);
    sol->send_compute_command();

    res = sol->output;
    p_pool->release();
    delete sol;
    
        
       


    delete[] A;
    delete[] b;

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}

py::list lowestApproxSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ nabla_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float nabla = nabla_in;
    float epsilon = epsilon_in;

    float* A = new float[N*M];
    float* b = new float[N];
    std::vector<float> res;
    py::list respy;

    int index = 0;
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        std::vector<float> rowVector;

        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            A[index] = value;
            index++;
        }
    }
    index = 0;
    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b[index] = value;
        index++;
    }
    
    nabla = nabla_in;

    
        NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
        MTL::Device* device = MTL::CreateSystemDefaultDevice();

        if (!device) {
            std::cerr << "Failed to create Metal device." << std::endl;
            std::exit(-1);
        }

        ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
        sol->init_with_device(device);

        
        
        sol->N = N;
        sol->M = M;
        sol->epsilon = epsilon;
        
        
        sol->prepare_data(A,b,nabla);
        sol->send_compute_command();

        res = sol->output;
        p_pool->release();
        delete sol;

        
    

    delete[] A;
    delete[] b;
    

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}






py::list nablaChunk(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    float* A = new float[N*M];
    float* b = new float[N];
    std::vector<float> res;
    py::list respy;

    int index = 0;
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        std::vector<float> rowVector;

        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            A[index] = value;
            index++;
        }
    }
    index = 0;
    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b[index] = value;
        index++;
    }
    
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Failed to create Metal device." << std::endl;
        std::exit(-1);
    }

    std::vector<float> resultsWithIndicesChunk(M, std::numeric_limits<float>::max()); // For the chunk results
    
    NablaChunk* nabla_chunk = new NablaChunk();
    nabla_chunk->init_with_device(device);
    nabla_chunk->N = N;
    nabla_chunk->M = M;
    nabla_chunk->epsilon = epsilon;
    nabla_chunk->debug = false;
    
    nabla_chunk->prepare_data(A,b);
    nabla_chunk->send_compute_command();

    resultsWithIndicesChunk = nabla_chunk->res;

    
    p_pool->release();

    delete nabla_chunk;
    
    delete[] A;
    delete[] b;

    for(int i = 0; i < resultsWithIndicesChunk.size(); i++){
        respy.append(resultsWithIndicesChunk.at(i));
    }
    
    return respy;   
    
}




PYBIND11_MODULE(metal_computation_py, m) {
    m.def("nabla", &nabla, "Compute Nabla");
    m.def("min_max", &minMax, "Compute MinMax");
    m.def("potential_min_solution", &potentialMinSolution, "Compute Potential Min Solution");
    m.def("lowest_approx_solution", &lowestApproxSolution, "Compute Lowest Approx Solution");
    m.def("nabla_chunk", &nablaChunk, "Compute chunk of Nabla");
}

