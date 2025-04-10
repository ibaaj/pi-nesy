//
//  main.cpp
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include "Computations.hpp"


namespace py = pybind11;

const py::float_ DEFAULT_EPSILON = std::numeric_limits<float>::epsilon();


py::tuple nabla(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    py::list respy;
    

    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    std::vector<std::vector<float>> A;
    std::vector<float> b;
    std::vector<float> res;
    std::vector<float> tmp;
    

    float nabla_result = 0.0f;

    
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        tmp.clear();
        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            tmp.emplace_back(value);
            
        }
        A.emplace_back(tmp);
    }

    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b.emplace_back(value);
    }
    ComputeNabla(A, b, res, epsilon);
    

    for(int i = 0; i < N; ++i)
    {
         nabla_result = std::fmax(res.at(i),  nabla_result);
          respy.append(res.at(i));
    }

    
    
    return py::make_tuple(nabla_result, respy);   
}


py::list minMax(py::list matrix, py::list x_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    std::vector<std::vector<float>> A;
    std::vector<float> x;
    std::vector<float> res;
    std::vector<float> tmp;
    py::list respy;

    
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        tmp.clear();
        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            tmp.emplace_back(value);
            
        }
        A.emplace_back(tmp);
    }

    for(py::handle valueHandle : py::cast<py::list>(x_in)){
        float value = py::cast<float>(valueHandle);
        x.emplace_back(value);
    }
    
    

     ComputeMinMax(A, x, res, epsilon);
   
    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}

py::list potentialMinSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;
    
    std::vector<std::vector<float>> A;
    std::vector<float> b;
    std::vector<float> res;
    std::vector<float> tmp;
    py::list respy;

    
    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        tmp.clear();
        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            tmp.emplace_back(value);
            
        }
        A.emplace_back(tmp);
    }

    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b.emplace_back(value);
    }


   

    
        ComputePotentialMinSolution(A, b, res, epsilon);
  

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    return respy;   
}

py::list lowestApproxSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ nabla_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;
    
    std::vector<std::vector<float>> A;
    std::vector<float> b;
    std::vector<float> res;
    std::vector<float> tmp;
    py::list respy;
    
    

    for (py::handle rowHandle : matrix) {
        py::list rowList = py::cast<py::list>(rowHandle);
        tmp.clear();
        for (py::handle valueHandle : rowList) {
            float value = py::cast<float>(valueHandle);
            tmp.emplace_back(value);
            
        }
        A.emplace_back(tmp);
    }

    for(py::handle valueHandle : py::cast<py::list>(b_in)){
        float value = py::cast<float>(valueHandle);
        b.emplace_back(value);
    }
    
    float nabla = nabla_in;

    
        LowestApproxSolution(A, b, res, nabla, epsilon);
    

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}

PYBIND11_MODULE(cpu_computation_py, m) {
    m.def("nabla", &nabla, "Compute Nabla");
    m.def("min_max", &minMax, "Compute MinMax");
    m.def("potential_min_solution", &potentialMinSolution, "Compute Potential Min Solution");
    m.def("lowest_approx_solution", &lowestApproxSolution, "Compute Lowest Approx Solution");
}

