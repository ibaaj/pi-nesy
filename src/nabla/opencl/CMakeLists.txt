cmake_minimum_required(VERSION 3.21)

project(opencl_computation_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(${PROJECT_SOURCE_DIR}/src)


set(OPENCL_CPP_HEADERS_PATH "/opt/homebrew/opt/opencl-clhpp-headers/include")


if(NOT EXISTS ${OPENCL_CPP_HEADERS_PATH})
    message(FATAL_ERROR "OpenCL C++ headers directory not found at ${OPENCL_CPP_HEADERS_PATH}. Please modify the CMakeLists.txt to include the correct path.")
else()
    include_directories(${OPENCL_CPP_HEADERS_PATH})
endif()


include_directories(${PROJECT_SOURCE_DIR}/external/OpenCL)


file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/Computations/*.cpp")


find_package(OpenCL REQUIRED)


add_executable(opencl_computation_test ./src/test.cpp ${SOURCES})
target_link_libraries(opencl_computation_test OpenCL::OpenCL)
message(STATUS "Processing opencl_computation_test")


project(opencl_computation_py)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Pybind11 module
pybind11_add_module(opencl_computation_py MODULE ./src/main.cpp ${SOURCES})

# Link against the OpenCL and Python libraries
target_link_libraries(opencl_computation_py PRIVATE OpenCL::OpenCL Python3::Python)

# Compiler options for optimization and all warnings
target_compile_options(opencl_computation_py PRIVATE -O3 -Wall)

# If on MacOS, set the dynamic_lookup linker flag
if(APPLE)
    set_target_properties(opencl_computation_py PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

message(STATUS "Processing opencl_computation_py")

