import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import math
import pprint

from nn import run_mnist_experiment


from mnistsudoku_rulebase import generate_rule_base_sudoku
from mnistsudoku_parser_files import parse_puzzles, load_txt_files_into_dict
from mnistsudoku_constraints import sudoku_build_set_constraints
from mnistsudoku_export import exportResultsSudoku
from mnistsudoku_inference import prepareInference, InferencePuzzles
from mnistsudoku_learning import prepare_learning_sudoku_1, prepare_learning_sudoku_2, possibilistic_learning_first_sets_of_rules, LearningParametersOfSecondset_of_rules


from mnistsudoku_generate_examples import make_examples_sudoku

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config
from rocauc import ROC_AUC

parent_dir = os.path.join(current_dir, '../../', 'possibilistc_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from partitions import build_partition, partition_tuples, build_attribute_values_for_rule_set
from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference_first_set_of_rules, build_matrix_inference_second_set_of_rules
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range
from possprob_transformations import antipignistic_transformation, minimum_specificity_principle_transformation



BASE_FILE_PATH_RESULTS = load_config.config["results_sudoku_directory"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


possible_threshold_values = custom_range()


def find_min_thresholds_and_learning(constraints,
        first_sets_of_rules,
        second_set_of_rules,
        puzzles_map_train,
        label_train_puzzle,
        puzzles_map_valid,
        label_validation_puzzle,
        probposs_transformation_method, 
        test_thresholds = []):
    
    use_thresholds = load_config.config["PiNeSy_use_thresholds"]
    matrices = {}
    threshold_first = 0
    isFHandled = []
    first_setting = True
    
    Parameters = {}
    logging.info("Number of sets f : " + str(len(constraints)))
    v = 0

    threshold_first = -1
    for f in constraints:
        Parameters[f] = []
    
    if use_thresholds is False:
        possible_threshold_values = [1.01] # just one possible thresholds > 1.
    else:
        possible_threshold_values = custom_range()


    logging.info("Preparing possibilistic learning step 1..")

    lambda_array, rho_array, partition_array, target_output_vector_array = prepare_learning_sudoku_1(constraints,
                                                                                                     puzzles_map_train,
                                                                                                         first_sets_of_rules,
                                                                                                     probposs_transformation_method)
    keep_examples_count_all = {}
    for threshold_1 in possible_threshold_values:
        if len(test_thresholds) != 0:
            if threshold_1 < test_thresholds[0]:
                continue
        for function in constraints:
            if function in isFHandled and first_setting:
                continue
            training_data_for_first_set = []
            index_set__examples = []
            for item in lambda_array[function]:
                lambda_degrees = lambda_array[function][item]
                rho_degrees = rho_array[function][item]
                input_vector = []
                for key in lambda_degrees:
                    input_vector.append(lambda_degrees[key])
                    input_vector.append(rho_degrees[key])
                training_data_for_first_set.append((input_vector, target_output_vector_array[function][item]))
                partition_1 = partition_array[function][item]
                index_set__examples.append(item)
            chebyshev_distances, threshold_to_errors = get_threshold_map(
                possible_threshold_values, index_set__examples, training_data_for_first_set, partition_1)
            if threshold_to_errors[threshold_1] == 0:
                continue
            s_params, r_params, kept_example_count, nabla = RuleParametersLearning(
                index_set__examples, training_data_for_first_set, partition_1, chebyshev_distances, threshold_1)
            keep_examples_count_all["f_" + str(function)] = kept_example_count
            Parameters[function].extend([s_params, r_params])
            threshold_first = max(threshold_first, threshold_1)
            isFHandled.append(function)
        if len(isFHandled) == len(constraints):
            first_setting = False
            logging.info(f"The threshold of the first sets of rules is {threshold_first}")
            break
    logging.info("Beginning possibilistic learning for the second set of rules...")
    partition_2, lambda_array_2, rho_array_2, target_output_vector_array_2 = prepare_learning_sudoku_2(
        constraints, puzzles_map_train, label_train_puzzle, first_sets_of_rules, lambda_array, rho_array, partition_array, Parameters,
        second_set_of_rules, probposs_transformation_method)
    Parameters2 = []
    training_data_for_second_set = []
    index_set__examples = []
    for key in lambda_array_2:
        input_vector = [lambda_array_2[key], rho_array_2[key]]
        output_vector = target_output_vector_array_2[key]
        training_data_for_second_set.append((input_vector, output_vector))
        index_set__examples.append(key)
    
    chebyshev_distances_2, threshold_to_errors_2 = get_threshold_map(
        possible_threshold_values, index_set__examples, training_data_for_second_set, partition_2)
    
    for threshold_2 in possible_threshold_values:
        if threshold_to_errors_2[threshold_2] == 0:
            continue

        if len(test_thresholds) != 0:
            if threshold_2 < test_thresholds[1]:
                continue
        s_params_2, r_params_2, kept_example_count_2, nabla_2 = RuleParametersLearning(
            index_set__examples, training_data_for_second_set, partition_2, chebyshev_distances_2, threshold_2)
        keep_examples_count_all["final_output_attr"] = kept_example_count_2
        Parameters2.extend([s_params_2, r_params_2])
        tuples_partition_first_sets_of_rules, matrices_first_set_of_rules, tuples_partition_second_set_of_rules, matrix_second_set_of_rules = prepareInference(
            constraints, first_sets_of_rules, Parameters, partition_array,
            second_set_of_rules, Parameters2, partition_2)
        break
    logging.info(f'thresholds_found: {threshold_first} and {threshold_2}')


    if len(test_thresholds) > 0:
        print("+++++++++++++++++++++++++++++++++++++++")
        print("Rule parameter values found by possibilistic learning")
        print("From Sets F:  (set F) -> [[s_params], [r_params]]")
        pprint.pprint(Parameters)
        print("From the output attribute for the decision: s_param, r_param")
        pprint.pprint(Parameters2)
        print("Keeped examples for learning | stats")
        pprint.pprint(keep_examples_count_all)
        print("+++++++++++++++++++++++++++++++++++++++")
        




    return [threshold_first, threshold_2],  matrices_first_set_of_rules, matrix_second_set_of_rules,  tuples_partition_first_sets_of_rules, tuples_partition_second_set_of_rules, keep_examples_count_all
    

def finding_hyperparameters_thresholds(constraints,
                                    first_sets_of_rules,
                                    second_set_of_rules,
                                    puzzles_map_train,
                                    label_train_puzzle,
                                    puzzles_map_valid,
                                    label_validation_puzzle,
                                    probposs_transformation_method):
    min_improvement = load_config.config["PiNeSy_min_improvement"]
    stagnation = load_config.config["PiNeSy_stagnation"]
    use_thresholds = load_config.config["PiNeSy_use_thresholds"]

    min_thresholds, matrices_first_set_of_rules, matrix_second_set_of_rules,  tuples_partition_first_sets_of_rules, tuples_partition_second_set_of_rules,_  = find_min_thresholds_and_learning(constraints,
                                                        first_sets_of_rules,
                                                        second_set_of_rules,
                                                        puzzles_map_train,
                                                        label_train_puzzle,
                                                        puzzles_map_valid,
                                                        label_validation_puzzle,
                                                        probposs_transformation_method)
    
    best_thresholds = min_thresholds.copy()

    if use_thresholds is False:
        return best_thresholds
    
    possible_threshold_values = custom_range()



    statistics_valid, inference_time = InferencePuzzles(2, puzzles_map_valid, label_validation_puzzle,
                                                                    constraints, probposs_transformation_method,
                                                                    first_sets_of_rules, second_set_of_rules,
                                                                    matrices_first_set_of_rules,
                                                                    tuples_partition_first_sets_of_rules,
                                                                    matrix_second_set_of_rules,
                                                                    tuples_partition_second_set_of_rules)
    best_accuracy = statistics_valid["accuracy_valid"]
    search_ranges = [sorted([val for val in possible_threshold_values if val > min_thresh]) for min_thresh in min_thresholds]
    stagnation_counters = [0] * 2


    for i in range(2): 
        logging.info(f"Pertubing the {i+1}-th threshold...")
        
        for current_threshold in search_ranges[i]:
            if stagnation_counters[i] >= stagnation: 
                if i < 1:
                    logging.info("Stagnation of accuracy. Next threshold.")
                else:
                    logging.info("Stagnation of accuracy. Exit processing.")
                break  
            
            test_thresholds = best_thresholds.copy()
            test_thresholds[i] = current_threshold
            
            test_thresholds, matrices_first_set_of_rules, matrix_second_set_of_rules,  tuples_partition_first_sets_of_rules, tuples_partition_second_set_of_rules,_  = find_min_thresholds_and_learning(constraints,
                                                        first_sets_of_rules,
                                                        second_set_of_rules,
                                                        puzzles_map_train,
                                                        label_train_puzzle,
                                                        puzzles_map_valid,
                                                        label_validation_puzzle,
                                                        probposs_transformation_method, 
                                                        test_thresholds)
            statistics_valid, inference_time = InferencePuzzles(2, puzzles_map_valid, label_validation_puzzle,
                                                                    constraints, probposs_transformation_method,
                                                                    first_sets_of_rules, second_set_of_rules,
                                                                    matrices_first_set_of_rules,
                                                                    tuples_partition_first_sets_of_rules,
                                                                    matrix_second_set_of_rules,
                                                                    tuples_partition_second_set_of_rules)
            current_accuracy = statistics_valid["accuracy_valid"]
            
            if current_accuracy > best_accuracy + min_improvement:
                best_accuracy = current_accuracy
                best_thresholds = test_thresholds
                logging.info(f"New best with thresholds {test_thresholds} at accuracy {best_accuracy}")
                stagnation_counters[i] = 0  
            else:
                stagnation_counters[i] += 1
    logging.info(f"Found these thresholds for the task: {best_thresholds}")
    return best_thresholds
