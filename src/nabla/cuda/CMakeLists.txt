cmake_minimum_required(VERSION 3.14) 
project(cuda_computation_py)

set(CMAKE_CXX_STANDARD 14) 
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_75 -Xcompiler -fPIC")


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})


set(CUDA_SOURCES
    src/PotentialLowestSolution.cu
    src/ChebyshevDistanceNabla.cu
    src/MinMax.cu
)


set(CPP_SOURCES
    src/main.cpp
    src/LowestApproxSolution.cpp 
)


cuda_add_library(cuda_computation_py_cuda ${CUDA_SOURCES})


pybind11_add_module(cuda_computation_py ${CPP_SOURCES})
target_link_libraries(cuda_computation_py PRIVATE cuda_computation_py_cuda)


include_directories(.)


add_executable(test_program src/test.cpp src/LowestApproxSolution.cpp src/SingleThread.cpp)
target_link_libraries(test_program PRIVATE cuda_computation_py_cuda ${CUDA_LIBRARIES})
if(WIN32)
    target_link_libraries(test_program PRIVATE ${PYTHON_LIBRARIES})
endif()
