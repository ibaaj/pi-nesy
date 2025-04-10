//
//  main.cpp
//

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



namespace py = pybind11;

const py::float_ DEFAULT_EPSILON = std::numeric_limits<float>::epsilon();


// redirect the std::out and std::err to python
// from: https://stackoverflow.com/questions/58758429/pybind11-redirect-python-sys-stdout-to-c-from-print 
class PyStdErrOutStreamRedirect {
    py::object _stdout;
    py::object _stderr;
    py::object _stdout_buffer;
    py::object _stderr_buffer;
public:
    PyStdErrOutStreamRedirect() {
        auto sysm = py::module::import("sys");
        _stdout = sysm.attr("stdout");
        _stderr = sysm.attr("stderr");
        auto stringio = py::module::import("io").attr("StringIO");
        _stdout_buffer = stringio();  
        _stderr_buffer = stringio();
        sysm.attr("stdout") = _stdout_buffer;
        sysm.attr("stderr") = _stderr_buffer;
    }
    std::string stdoutString() {
        _stdout_buffer.attr("seek")(0);
        return py::str(_stdout_buffer.attr("read")());
    }
    std::string stderrString() {
        _stderr_buffer.attr("seek")(0);
        return py::str(_stderr_buffer.attr("read")());
    }
    ~PyStdErrOutStreamRedirect() {
        auto sysm = py::module::import("sys");
        sysm.attr("stdout") = _stdout;
        sysm.attr("stderr") = _stderr;
    }
};




auto nabla(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    int N = N_in;
    int M = M_in;
    
    float epsilon = epsilon_in;

    std::vector<std::vector<float>> A;
    std::vector<float> b;
    std::vector<float> res;
    std::vector<float> tmp;
    py::list respy;

    float nabla_result;

    
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
    

    {
        
        PyStdErrOutStreamRedirect pyOutputRedirect{};

        ComputeNabla* nabla = new ComputeNabla();
        
        nabla->N = N;
        nabla->M = M;
        nabla->epsilon = epsilon;
        
        nabla->init_with_device();
        nabla->prepareData(A,b);
        nabla->sendComputeCommand();

        nabla_result = nabla->nabla;
        res = nabla->nablaI;
        delete nabla;
    }

    for(int i = 0; i < res.size(); i++){
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
    

    {
        PyStdErrOutStreamRedirect pyOutputRedirect{};


        ComputeMinMax* minmax = new ComputeMinMax();

        
        minmax->N = N;
        minmax->M = M;
        minmax->epsilon = epsilon;
        
        minmax->init_with_device();
        minmax->prepareData(A,x);
        minmax->sendComputeCommand();

        res = minmax->output;
        delete minmax;
    }

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
    

    {
        PyStdErrOutStreamRedirect pyOutputRedirect{};

        ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
        sol->init_with_device();
        
        sol->N = N;
        sol->M = M;
        sol->epsilon = epsilon;

        sol->init_with_device();
        sol->prepareData(A,b, 0.0f);
        sol->sendComputeCommand();

        res = sol->output;
        delete sol;
    }


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

    {
        PyStdErrOutStreamRedirect pyOutputRedirect{};

        ComputePotentialMinSolution* sol = new ComputePotentialMinSolution();
        sol->init_with_device();
        
        sol->N = N;
        sol->M = M;
        sol->epsilon = epsilon;
        
        
        sol->prepareData(A,b,nabla);
        sol->sendComputeCommand();

        res = sol->output;
        delete sol;
    }
    

    for(int i = 0; i < res.size(); i++){
        respy.append(res.at(i));
    }
    
    return respy;   
}

PYBIND11_MODULE(opencl_computation_py, m) {
    m.def("nabla", &nabla, "Compute Nabla");
    m.def("min_max", &minMax, "Compute MinMax");
    m.def("potential_min_solution", &potentialMinSolution, "Compute Potential Min Solution");
    m.def("lowest_approx_solution", &lowestApproxSolution, "Compute Lowest Approx Solution");
}

