import numpy as np
import time


from collections import deque
import os
import sys


current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config


parent_dir = os.path.join(current_dir, '../../', 'lib')
sys.path.append(os.path.abspath(parent_dir))

if load_config.config["library"] == "Cuda":
    from cuda_computation_py import min_max, lowest_approx_solution, potential_min_solution, nabla
if load_config.config["library"] == "Apple Metal":
    from metal_computation_py import min_max, lowest_approx_solution, potential_min_solution, nabla, nabla_chunk
if load_config.config["library"] == "OpenCL":
    from opencl_computation_py import min_max, lowest_approx_solution, potential_min_solution, nabla
if load_config.config["library"] == "CPU":
    from cpu_computation_py import min_max, lowest_approx_solution, potential_min_solution, nabla



EPSILON = load_config.config["PiNeSy_epsilon"]



def worker(task):
    function_name, input_data = task
    try:
        if function_name == 'min_max':
            result = min_max(*input_data)
        elif function_name == 'lowest_approx_solution':
            result = lowest_approx_solution(*input_data)
        elif function_name == 'potential_min_solution':
            result = potential_min_solution(*input_data)
        elif function_name == 'nabla':
            result = nabla(*input_data)
        else:
            raise ValueError("Unknown function")
        return (True, result)
    except Exception as e:
        return (False, str(e))

def execute_task_with_retries(task):
    """ in the end, the isolation in a subprocess was avoided ... """
    return worker(task)[1]
    

def is_close(a, b, rel_tol=EPSILON, abs_tol=0.0):
    # https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python 
    """Check if two values are close to each other within a tolerance."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_vector_equal(x, y):
    """Check if two vectors are equal within a tolerance of EPSILON."""
    return all(abs(a - b) <= EPSILON for a, b in zip(x, y))

def epsilon_product(a, b):
    """perform the epsilon product."""
    if is_close(a, b, EPSILON):
        return 0.0
    return b if a < b else 0.0

def minmax_product(matrix, input_vector):
    """Compute the min-max product using  the gpu computation library."""
    task = ('min_max', (matrix, input_vector, len(matrix), len(matrix[0]), EPSILON))
    try:
        result = execute_task_with_retries(task)
        return result
    except Exception as e:
        print(f"Task min_max failed: {e}")

def build_matrix_learning_system(Lambda, inputs_vector):
    """Build a matrix from lambda values and input vectors."""
    lambdas_deg, rhos_deg = inputs_vector[::2], inputs_vector[1::2]
    matrix = []
    for row_lambda in Lambda:
        row_matrix = []
        for el in row_lambda:
            if el > 0:
                row_matrix.extend([lambdas_deg[el - 1], 1])
            else:
                row_matrix.extend([1, rhos_deg[-el - 1]])
        matrix.append(row_matrix)
    return matrix

def get_transpose(matrix):
    """Get the transpose of a matrix."""
    return [list(i) for i in zip(*matrix)]

def compute_pseudo_greatest_solution(lambda_values, b):
    """Compute the pseudo greatest solution."""
    """ this solution is greater than the lowest solution..."""
    n, m = len(b), len(lambda_values[0])
    sol = []
    for j in range(m):
        lambda_j_top = [i for i in range(n) if lambda_values[i][j] > 0]
        lambda_j_bot = [i for i in range(n) if lambda_values[i][j] < 0]
        sol.extend([
            max((b[i] for i in lambda_j_top), default=0),
            max((b[i] for i in lambda_j_bot), default=0)
        ])
    return sol

def lowest_approximation_solution(matrix, b, nabla):
    """Find the lowest approximation solution."""
    task = ('lowest_approx_solution', (matrix, b, len(matrix), len(matrix[0]), float(nabla), EPSILON))
    try:
        result = execute_task_with_retries(task)
        return result
    except Exception as e:
        print(f"Task failed lowest_approx_solution: {e}")

def check_consistency_system(matrix, b):
    """Check if a min-max system is consistent."""
    sol = potential_min_sol(matrix, b)
    min_max_v = minmax_product(matrix, sol)
    return check_vector_equal(b, min_max_v)

def potential_min_sol(matrix, b):
    """Compute the potential lowest solution for a given matrix and a vector 'b' via GPU"""
    task = ('potential_min_solution', (matrix, b, len(matrix), len(matrix[0]), EPSILON))
    try:
        result = execute_task_with_retries(task)
        return result
    except Exception as e:
        print(f"Task failed potential_min_solution: {e}")
    # In Python
    """
    Compute the potential lowest solution for a given matrix and a vector 'b' via CPU python
    
    Args:
        matrix: A list of lists representing the matrix.
        b: A list representing the vector 'b'.
    
    Returns:
        A list representing the potential lowest solution vector.
    """
    transposed_matrix = get_transpose(matrix)
    m, n = len(transposed_matrix), len(transposed_matrix[0])
    result_vector = []

    for j in range(m):
        max_val = 0
        for i in range(n):
            max_val = max(max_val, epsilon_product(transposed_matrix[j][i], b[i]))
        result_vector.append(max_val)

    return result_vector

def lowest_chebyshev_approximation(matrix, initial_b, nabla):
    """
    Compute the lowest Chebyshev approximation of vector b, for a given matrix, vector 'b', and nabla.
    
    Args:
        matrix: A list of lists representing the matrix.
        prev_b: A list representing the initial vector 'b'.
        nabla: A float representing the nabla value.
    
    Returns:
        A list representing the lowest Chebyshev approximation.
    """
    adjusted_b = [max(bi - nabla,0) for bi in initial_b]
    try:
        task1 = ('potential_min_solution', (matrix, adjusted_b, len(matrix), len(matrix[0]), EPSILON))
        potential_min_sol = execute_task_with_retries(task1)
        task2 = ('min_max', (matrix, potential_min_sol, len(matrix), len(matrix[0]), EPSILON))
        min_max_v = execute_task_with_retries(task2)
        return min_max_v
    except Exception as e:
        print(f"Task failed potential_min_solution or min_max: {e}")

def chebyshev_distance(matrix, b, threshold):
    """
    Compute the Chebyshev distance,
            the set NC of indices of the nabla_i strictly lower than the threshold,
            the max of the nabla_i within NC.
    
    Args:
        matrix: A list of lists representing the matrix.
        b: A list representing the vector 'b'.
        threshold: A float representing the threshold .
    
    Returns:
        A tuple containing:
            - nabla: The Chebyshev distance.
            - NC: A list of indices that such that nabla_i < threshold.
            - nablaNC: the max of the nabla_i within NC.
    """
    nabla_res = 0.0
    nabla_i_array = []

    task = ('nabla', (matrix, b, len(matrix), len(matrix[0]), EPSILON))
    try:
        nabla_res, nabla_i_array = execute_task_with_retries(task)
        NC = [i for i, delta in enumerate(nabla_i_array) if delta < threshold]
        nablaNC = max(nabla_i_array[i] for i in NC) if NC else 0
        return nabla_res, NC, nablaNC
    except Exception as e:
        print(f"Task failed nabla: {e}")


def query_Gij(partition, examples, sequence_kept_examples, possibilistic_training_data, i, j):
    N_total = len(sequence_kept_examples)*len(partition)
    a_ij = 0.0
    example_id = i // len(partition)
    real_example_id = sequence_kept_examples[example_id]
    inputs_vector, output_vector = possibilistic_training_data[real_example_id]
    lambdas_deg, rhos_deg = inputs_vector[::2], inputs_vector[1::2]
    j_to_check = j // 2 + 1
    k_to_check = i - (i // len(partition))*len(partition)
    k = 0
    if j_to_check in partition[k_to_check]:
        if j % 2 == 0:
            a_ij = lambdas_deg[j_to_check - 1]
        else:
            a_ij = 1
    if -j_to_check in partition[k_to_check]:
        if j % 2 == 0:
            a_ij = 1
        else:
            a_ij = rhos_deg[j_to_check - 1]
    return a_ij

def chebyshev_distance_low_memory_use(partition, examples, sequence_kept_examples, possibilistic_training_data, M, big_y_output_vector, threshold):
    if load_config.config["library"] != "Apple Metal":
        print("this function only works with Apple Metal currently.")
        sys.exit(-1)

    N = len(sequence_kept_examples)*len(partition)
    b = big_y_output_vector
    assert(len(b) == N)

    chunkSize = 20  
    
    nablaResults = [float('inf')] * N  

    for i in range(N):
        resultsWithIndices = [float('inf')] * M
        
        for startCol in range(0, M, chunkSize):
            endCol = min(startCol + chunkSize, M)
            A_chunk = []  
            chunkColumnIndices = []

            for j in range(startCol, endCol):
                chunkColumnIndices.append(j)
            
            for k in range(N): 
                rowForChunk = []
                for j in range(startCol, endCol):
                    rowForChunk.append(query_Gij(partition, examples, sequence_kept_examples, possibilistic_training_data, k, j))
                A_chunk.append(rowForChunk)

            resultsWithIndicesChunk = nabla_chunk(A_chunk, b, N, endCol-startCol, EPSILON)



            
            for idx, j in enumerate(chunkColumnIndices):
                resultsWithIndices[j] = resultsWithIndicesChunk[idx]
            

        nabla_i = min(resultsWithIndices)
        nablaResults[i] = nabla_i
    
    nabla_res = max(nablaResults)
    NC = [i for i, delta in enumerate(nablaResults) if delta < threshold]
    nablaNC = max(nablaResults[i] for i in NC) if NC else 0
    return nabla_res, NC, nablaNC

