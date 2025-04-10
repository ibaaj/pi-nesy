#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cudaChebyshevDistanceNabla.h"
#include "cudaPotentialLowestSolution.h"
#include "cudaMinMax.h"
#include "LowestApproxSolution.hpp"



namespace py = pybind11;

const py::float_ DEFAULT_EPSILON = std::numeric_limits<float>::epsilon();


py::tuple nabla(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
         py::list respy;

        int N = N_in;
         int M = M_in;
    
        float epsilon = static_cast<float>(epsilon_in);
       
        
        
        float* A_ptr = new float [N*M];
        float* b_ptr = new float[N];
        float* res_ptr = new float[N];

        float nabla_result = 0.0f;

        int i = 0;
        for (auto row : matrix) {
            for (auto val : row.cast<py::list>()) {
                A_ptr[i] = val.cast<float>();
                i++;
            }
        }
        i = 0;
        for (auto val : b_in) {
             b_ptr[i] = val.cast<float>();
            i++;
        }

        
        ChebyshevDistanceNabla(A_ptr, b_ptr, res_ptr, N, M, epsilon);
    
        float nabla = 0.0f;

        for(int i = 0; i < N; ++i){
            respy.append(res_ptr[i]);

            nabla = std::fmax(nabla, res_ptr[i]);

        }
        delete[] res_ptr;
        delete[] A_ptr;
        delete[] b_ptr;

        
        return py::make_tuple(nabla, respy);

}




py::list minMax(py::list matrix, py::list x_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
        py::list respy;

        int N = N_in;
        int M = M_in;
    
       float epsilon = static_cast<float>(epsilon_in);

       
        
        
        float* A_ptr = new float [N*M];
        float* x_ptr = new float[M];
        float* res_ptr = new float[N];

        float nabla_result = 0.0f;

        int i = 0;
        for (auto row : matrix) {
            for (auto val : row.cast<py::list>()) {
                A_ptr[i] = val.cast<float>();
                i++;
            }
        }
        i = 0;
        for (auto val : x_in) {
             x_ptr[i] = val.cast<float>();
            i++;
        }


        MinMax(A_ptr, x_ptr, res_ptr, N, M, epsilon);

    
        for(int i = 0; i < N; i++){
        respy.append(res_ptr[i]);
        }

        delete[] res_ptr;
        delete[] A_ptr;
        delete[] x_ptr;


        return respy;

}


py::list potentialMinSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    py::list respy;

    int N = N_in;
    int M = M_in;

    float epsilon = static_cast<float>(epsilon_in);


    
    
    float* A_ptr = new float [N*M];
    float* b_ptr = new float[N];
    float* res_ptr = new float[M];

    float nabla_result = 0.0f;

    int i = 0;
    for (auto row : matrix) {
        for (auto val : row.cast<py::list>()) {
            A_ptr[i] = val.cast<float>();
            i++;
        }
    }
    i = 0;
    for (auto val : b_in) {
        b_ptr[i] = val.cast<float>();
        i++;
    }
        

    PotentialLowestSolution(A_ptr, b_ptr, res_ptr, N, M, epsilon);


    for(int i = 0; i < M; i++){
    respy.append(res_ptr[i]);
    }

    delete[] res_ptr;
    delete[] A_ptr;
    delete[] b_ptr;


    return respy;
}

py::list lowestApproxSolution(py::list matrix, py::list b_in, py::int_ N_in, py::int_ M_in, py::float_ nabla_in, py::float_ epsilon_in = DEFAULT_EPSILON){
    py::list respy;

    int N = N_in;
    int M = M_in;

    float epsilon = static_cast<float>(epsilon_in);

    float nabla = nabla_in;

    
    
    float* A_ptr = new float [N*M];
    float* b_ptr = new float[N];
    float* res_ptr = new float[M];

    float nabla_result = 0.0f;

    int i = 0;
    for (auto row : matrix) {
        for (auto val : row.cast<py::list>()) {
            A_ptr[i] = val.cast<float>();
            i++;
        }
    }
    i = 0;
    for (auto val : b_in) {
        b_ptr[i] = val.cast<float>();
        i++;
    }


    LowestApproxSolution(A_ptr, b_ptr, res_ptr, nabla, N, M, epsilon);


    for(int i = 0; i < M; i++){
    respy.append(res_ptr[i]);
    }

    delete[] res_ptr;
    delete[] A_ptr;
    delete[] b_ptr;


    return respy;
}





PYBIND11_MODULE(cuda_computation_py, m) {
    m.def("nabla", &nabla, "Compute Nabla");
    m.def("min_max", &minMax, "Compute MinMax");
    m.def("potential_min_solution", &potentialMinSolution, "Compute Potential Min Solution");
    m.def("lowest_approx_solution", &lowestApproxSolution, "Compute Lowest Approx Solution");
}


