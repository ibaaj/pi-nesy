import time
import sys
import os
import pprint
import numpy as np

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config


from learning import build_matrix_learning_system, chebyshev_distance, lowest_chebyshev_approximation, \
    lowest_approximation_solution, chebyshev_distance_low_memory_use



def custom_range(n=load_config.config["PiNeSy_custom_range_n"], strengh=load_config.config["PiNeSy_custom_range_strengh"]):
    epsilon = load_config.config["PiNeSy_epsilon"]

    """
    end_value = 1 + epsilon
    
    # Generate n evenly spaced values in [0, 1]
    t = np.linspace(0, 1, n)

    values = (t**strengh) * end_value
    """

    end_value = 1 + epsilon

    # Generate i from 1 to n
    i = np.arange(1, n+1)

    # Compute t_i = (i / n)
    t = i / n

    # Compute the values for the set
    values = (t**strengh) * end_value
    

    sorted_list = list(values)

    
    
    return sorted_list

def retry_learning_step(func, *args, max_attempts=50, **kwargs):
    """
    Retry wrapper for GPU function calls.
    """
    for attempt in range(max_attempts):
        try:
            result = func(*args, **kwargs)
            # Apply specific check based on the function name
            if func.__name__ == 'lowest_chebyshev_approximation' or func.__name__ == 'lowest_approximation_solution':
                if len(result) == 0:
                    raise ValueError("Function returned an empty list, which is invalid.")
            elif func.__name__ == 'chebyshev_distance':
                if result[0] < 0.0 or result[0] > 1.0 or len(result[1]) == 0 or result[2] < 0.0 or result[2] > 1.0:
                    raise ValueError("Function returned an error, which is invalid.")
            
            # If result passes the check, return it
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
            time.sleep(0.5)  # Wait for 500 milliseconds before the next attempt


def get_threshold_map(thresholds_range, examples, possibilistic_training_data, partition):
    chebyshev_distance_list = {}
    threshold_dict = {threshold: 0 for threshold in thresholds_range}
    final_threshold = -1
    for example_index, example in enumerate(examples):
        input_vector, output_vector = possibilistic_training_data[example_index]
        learning_matrix = build_matrix_learning_system(partition, input_vector)

        chebyshev_distance_result, nc_indices, reduced_chebyshev_distance = retry_learning_step(
            chebyshev_distance, learning_matrix, output_vector, 1.0)
        chebyshev_distance_list[example_index] = chebyshev_distance_result

        for threshold in thresholds_range:
            if chebyshev_distance_result < threshold:
                threshold_dict[threshold] += 1
    return chebyshev_distance_list, threshold_dict

def build_equation_system_learning(partition, examples, chebyshev_distance_list, possibilistic_training_data, final_threshold):
    gamma_matrix = []
    big_y_vector = []
    keep_examples_count = 0
    sequence_kept_examples = []
    M = 0
    for example_index, example in enumerate(examples):
        if chebyshev_distance_list[example_index] < final_threshold:
            sequence_kept_examples.append(example_index)
            keep_examples_count += 1
            input_vector, output_vector = possibilistic_training_data[example_index]
            learning_matrix = build_matrix_learning_system(partition, input_vector)
            M = len(learning_matrix[0])
            
            output_vector = retry_learning_step(lowest_chebyshev_approximation, learning_matrix, output_vector,
                                                           chebyshev_distance_list[example_index])

            for row_index, row in enumerate(learning_matrix):
                gamma_matrix.append(row)
                big_y_vector.append(output_vector[row_index])
    
    return gamma_matrix, big_y_vector, sequence_kept_examples, M, keep_examples_count

def RuleParametersLearning(examples, possibilistic_training_data, partition, chebyshev_distance_list, threshold):
    perform_possibilistic_learning = load_config.config["PiNeSy_perform_possibilistic_learning"]
    if perform_possibilistic_learning is False:
        print("possibilistic learning is not performed (by config).")
        input_vector, output_vector = possibilistic_training_data[0]
        m_size = len(input_vector)
        s_parameters = [0.0 for index in range(0, m_size, 2)]
        r_parameters = [0.0 for index in range(1, m_size, 2)]

        print(f's_parameters not learned but fixed to zero:{s_parameters}')
        print(f'r_parameters not learned but fixed to zero:{r_parameters}')

        return s_parameters, r_parameters, [], 1


    gamma_matrix, big_y_vector, sequence_kept_examples, M, keep_examples_count = build_equation_system_learning(partition, examples, chebyshev_distance_list, possibilistic_training_data, threshold)    

    reduced_chebyshev_distance, nc_indices_final, final_reduced_chebyshev_distance = retry_learning_step(
            chebyshev_distance, gamma_matrix, big_y_vector, threshold)
    
    reduced_big_y = big_y_vector
    reduced_gamma = gamma_matrix

    lowest_approx_solution = retry_learning_step(lowest_approximation_solution, reduced_gamma, reduced_big_y,
                                                           final_reduced_chebyshev_distance)
    


    s_parameters = [lowest_approx_solution[index] for index in range(0, len(lowest_approx_solution), 2)]
    r_parameters = [lowest_approx_solution[index] for index in range(1, len(lowest_approx_solution), 2)]

    print(f's_parameters learned:{s_parameters}')
    print(f'r_parameters learned:{r_parameters}')

    return s_parameters, r_parameters, keep_examples_count, final_reduced_chebyshev_distance



