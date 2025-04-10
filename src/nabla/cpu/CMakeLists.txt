cmake_minimum_required(VERSION 3.29)
project(cpu_computation_py_pi_nesy)


set(CMAKE_CXX_STANDARD 17)



find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")


pybind11_add_module(cpu_computation_py ${SOURCE_DIR}/main.cpp ${SOURCE_DIR}/Computations.cpp)
target_include_directories(cpu_computation_py PRIVATE ${SOURCE_DIR})


add_executable(test_program ${SOURCE_DIR}/test.cpp ${SOURCE_DIR}/Computations.cpp ${SOURCE_DIR}/SingleThread.cpp)
target_include_directories(test_program PRIVATE ${SOURCE_DIR})

