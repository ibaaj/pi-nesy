cmake_minimum_required(VERSION 3.29)

project(metal_computation_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")



include_directories(${PROJECT_SOURCE_DIR}/src)


file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/Computations/*.cpp")


set(SHADER_DIR "${PROJECT_SOURCE_DIR}/shaders")

add_custom_command(
    OUTPUT ${SHADER_DIR}/ComputePotentialMinSolution.metallib ${SHADER_DIR}/ComputeNabla.metallib
    COMMAND xcrun -sdk macosx metal -fcikernel ${SHADER_DIR}/ComputePotentialMinSolution.metal -c -o ${SHADER_DIR}/ComputePotentialMinSolution.ir && echo "Success: Compiled ComputePotentialMinSolution.metal" || echo "Error: Failed to compile ComputePotentialMinSolution.metal"
    COMMAND xcrun -sdk macosx metallib -o ${SHADER_DIR}/ComputePotentialMinSolution.metallib ${SHADER_DIR}/ComputePotentialMinSolution.ir && echo "Success: Created ComputePotentialMinSolution.metallib" || echo "Error: Failed to create ComputePotentialMinSolution.metallib"
    
    COMMAND xcrun -sdk macosx metal -fcikernel ${SHADER_DIR}/ComputeNabla.metal -c -o ${SHADER_DIR}/ComputeNabla.ir && echo "Success: Compiled ComputeNabla.metal" || echo "Error: Failed to compile ComputeNabla.metal"
    COMMAND xcrun -sdk macosx metallib -o ${SHADER_DIR}/ComputeNabla.metallib ${SHADER_DIR}/ComputeNabla.ir && echo "Success: Created ComputeNabla.metallib" || echo "Error: Failed to create ComputeNabla.metallib"
    
    COMMAND xcrun -sdk macosx metal -fcikernel ${SHADER_DIR}/ComputeMinMax.metal -c -o ${SHADER_DIR}/ComputeMinMax.ir && echo "Success: Compiled ComputeMinMax.metal" || echo "Error: Failed to compile ComputeMinMax.metal"
    COMMAND xcrun -sdk macosx metallib -o ${SHADER_DIR}/ComputeMinMax.metallib ${SHADER_DIR}/ComputeMinMax.ir && echo "Success: Created ComputeMinMax.metallib" || echo "Error: Failed to create ComputeMinMax.metallib"
    
    COMMAND xcrun -sdk macosx metal -fcikernel ${SHADER_DIR}/NablaChunk.metal -c -o ${SHADER_DIR}/NablaChunk.ir && echo "Success: Compiled NablaChunk.metal" || echo "Error: Failed to compile NablaChunk.metal"
    COMMAND xcrun -sdk macosx metallib -o ${SHADER_DIR}/NablaChunk.metallib ${SHADER_DIR}/NablaChunk.ir && echo "Success: Created NablaChunk.metallib" || echo "Error: Failed to create NablaChunk.metallib"
    
    COMMENT "Compiling Metal shaders"
)


add_custom_target(CompileShaders ALL DEPENDS ${SHADER_DIR}/ComputePotentialMinSolution.metallib ${SHADER_DIR}/ComputeNabla.metallib ${SHADER_DIR}/ComputeMinMax.metallib ${SHADER_DIR}/NablaChunk.metallib)

set(DIR_LIB_SHADERS "${CMAKE_CURRENT_LIST_DIR}/../../../shaders")


add_custom_command(
    OUTPUT "${DIR_LIB_SHADERS}/.dummy"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIR_LIB_SHADERS}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_DIR}" "${DIR_LIB_SHADERS}"
    COMMAND ${CMAKE_COMMAND} -E touch "${DIR_LIB_SHADERS}/.dummy" 
    COMMENT "Copying shaders to ${DIR_LIB_SHADERS}"
)
add_custom_target(CopyShadersToLibDir ALL DEPENDS "${DIR_LIB_SHADERS}/.dummy")


set(DIR_BUILD_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/shaders")


add_custom_command(
    OUTPUT "${DIR_BUILD_SHADERS}/.dummy" 
    COMMAND ${CMAKE_COMMAND} -E echo "Copying shaders to build directory..."
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIR_BUILD_SHADERS}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_DIR}" "${DIR_BUILD_SHADERS}"
    COMMAND ${CMAKE_COMMAND} -E touch "${DIR_BUILD_SHADERS}/.dummy"
    COMMENT "Copying shaders to ${DIR_BUILD_SHADERS}"
)

add_custom_target(CopyShadersToBuildDir ALL DEPENDS "${DIR_BUILD_SHADERS}/.dummy")






add_subdirectory(lib/metal-cmake)  


add_executable(metal_computation_test ./src/test.cpp ${SOURCES})


add_dependencies(metal_computation_test CompileShaders)
add_dependencies(metal_computation_test CopyShadersToBuildDir)

target_link_libraries(metal_computation_test METAL_CPP)
message(STATUS "Processing metal_computation_test")


project(metal_computation_py)





find_package(Python3 COMPONENTS Interpreter QUIET)


if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.version_info[0], sys.version_info[1], sep='.')"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    
    if(PYTHON_VERSION)
        find_package(Python3 ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)
    endif()
endif()
find_package(pybind11 REQUIRED)


pybind11_add_module(metal_computation_py MODULE ./src/main.cpp ${SOURCES})

add_dependencies(metal_computation_py CompileShaders)
add_dependencies(metal_computation_py CopyShadersToLibDir)


target_link_libraries(metal_computation_py PRIVATE METAL_CPP Python3::Python)


target_compile_options(metal_computation_py PRIVATE -O3 -Wall)


if(APPLE)
    set_target_properties(metal_computation_py PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

message(STATUS "Processing metal_computation_py")


