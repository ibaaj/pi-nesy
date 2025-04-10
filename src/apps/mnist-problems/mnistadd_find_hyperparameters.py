import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import pprint
import gc


from mnistadd_c_tuples import (
    possibilistic_training_data_c_tuples,
    build_input_vector_c_tuples,
    compute_min_threshold_c_tuples,
    matrix_and_data_learning_c_tuples,
    inference_with_training_data_c_tuples
)

from mnistadd_carry import (
    build_input_vector_carry,
    possibilistic_training_data_carry,
    inference_with_training_data_carry,
    matrix_and_data_learning_carry,
    compute_min_threshold_carry
)


from mnistadd_y_outputs import (
    build_input_vector_y_outputs,
    possibilistic_training_data_y_outputs,
    compute_min_threshold_y_outputs,
    matrix_and_data_learning_y_outputs,
    inference_with_training_data_y_outputs
)

from mnistadd_evaluation import evaluate_model


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))


from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range



parent_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

possible_threshold_values = sorted(custom_range())


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



def get_minimal_thresholds(k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, probposs_transformation_method):
    matrices = {}
    matrices["c_tuples"] = {}
    matrices["carry"] = {}
    min_threshold_c_tuple = 0
    min_threshold_carry = 0
    output_possibility_distributions_c_tuples = {}
    logging.info(f"Generating Possibilistic training data for c_tuple : c_{k}")
    training_data_carry_tuples_c_k = possibilistic_training_data_c_tuples(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples_train, 
                                                                        probposs_transformation_method)
    
    logging.info(f"Computing minimal threshold for c_tuple : c_{k}")
    min_threshold_c_tuple_c_k, chebyshev_distance_list_c_tuple_c_k = compute_min_threshold_c_tuples(k, possible_threshold_values, training_data_carry_tuples_c_k, examples_train, partitions)
    
    logging.info(f"Minimal threshold for c_tuple : c_{k} : {min_threshold_c_tuple_c_k}")

    logging.info(f"Computing relational matrices for c_tuple : c_{k}")
    data_c_tuple_c_k_learning, matrices_c_tuple_c_k_learning, _ = matrix_and_data_learning_c_tuples(k, examples_train, training_data_carry_tuples_c_k, partitions, chebyshev_distance_list_c_tuple_c_k, min_threshold_c_tuple_c_k)

    logging.info(f"Inferring training examples for c_tuple : c_{k}")
    output_possibility_distributions_c_tuple_c_k = inference_with_training_data_c_tuples(k, k,  tps, examples_train,  matrices_c_tuple_c_k_learning,  training_data_carry_tuples_c_k)
    
    matrices["c_tuples"][f"c_{k}"] = matrices_c_tuple_c_k_learning[f"c_{k}"]

    min_threshold_c_tuple = max(min_threshold_c_tuple, min_threshold_c_tuple_c_k)

    

    for example_index, example in enumerate(examples_train):
        output_possibility_distributions_c_tuples[example_index] = {}
        output_possibility_distributions_c_tuples[example_index][f"c_{k}"] = output_possibility_distributions_c_tuple_c_k[example_index]

    
    
    logging.info(f"Generating Possibilistic training data for carry : w_{k}")
    training_data_carry_r_k = possibilistic_training_data_carry(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples_train, 
                                                                        output_possibility_distributions_c_tuple_c_k)

    logging.info(f"Computing minimal threshold for carry : w_{k}")
    min_threshold_carry_r_k, chebyshev_distance_list_carry_r_k = compute_min_threshold_carry(k, possible_threshold_values, training_data_carry_r_k, examples_train, partitions)

    logging.info(f"Minimal threshold for carry : w_{k} : {min_threshold_carry_r_k}")
    

    logging.info(f"Computing relational matrices for carry : w_{k}")
    data_carry_r_k_learning, matrices_carry_r_k_learning, _ = matrix_and_data_learning_carry(k, examples_train, training_data_carry_r_k, partitions, chebyshev_distance_list_carry_r_k, min_threshold_carry_r_k)

    logging.info(f"Inferring training examples for carry : w_{k}")
    output_possibility_distributions_carry_r_k = inference_with_training_data_carry(k, k,  tps, examples_train,  matrices_carry_r_k_learning,  training_data_carry_r_k)

    matrices["carry"][f"w_{k}"] = matrices_carry_r_k_learning[f"w_{k}"]
    min_threshold_carry = max(min_threshold_carry, min_threshold_carry_r_k)

    
    
    last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_k
    for j in range(k-1, 0, -1):
        logging.info(f"Generating Possibilistic training data for c_tuple : c_{j}")
        training_data_carry_tuples_c_j = possibilistic_training_data_c_tuples(j,
                                                                            k,
                                                                            rule_sets,
                                                                            partitions,
                                                                            tps,
                                                                            examples_train,
                                                                            probposs_transformation_method,
                                                                            last_output_possibility_distribution_carry_r_j
                                                                            )

        logging.info(f"Computing minimal threshold for c_tuple : c_{j}")
        min_threshold_c_tuple_c_j, chebyshev_distance_list_c_tuple_c_j = compute_min_threshold_c_tuples(j, possible_threshold_values, training_data_carry_tuples_c_j, examples_train, partitions)

        logging.info(f"Minimal threshold for c_tuple : c_{j} : {min_threshold_c_tuple_c_j}")

        logging.info(f"Computing relational matrices for c_tuple : c_{j}")
        data_c_tuple_c_j_learning, matrices_c_tuple_c_j_learning, _  = matrix_and_data_learning_c_tuples(j, examples_train, training_data_carry_tuples_c_j, partitions, chebyshev_distance_list_c_tuple_c_j, min_threshold_c_tuple_c_j)

        logging.info(f"Inferring training examples for c_tuple : c_{j}")
        output_possibility_distributions_c_tuple_c_j = inference_with_training_data_c_tuples(j, k,  tps, examples_train,  matrices_c_tuple_c_j_learning,  training_data_carry_tuples_c_j)

        matrices["c_tuples"][f"c_{j}"] = matrices_c_tuple_c_j_learning[f"c_{j}"].copy()
        min_threshold_c_tuple = max(min_threshold_c_tuple, min_threshold_c_tuple_c_j)

        for example_index, example in enumerate(examples_train):
            output_possibility_distributions_c_tuples[example_index][f"c_{j}"] = output_possibility_distributions_c_tuple_c_j[example_index]

        logging.info(f"Generating Possibilistic training data for carry : w_{j}")
        training_data_carry_r_j = possibilistic_training_data_carry(j,
                                                                                k, 
                                                                                rule_sets, 
                                                                                partitions, 
                                                                                tps, 
                                                                                examples_train, 
                                                                                output_possibility_distributions_c_tuple_c_j)

        logging.info(f"Computing minimal threshold for carry : w_{j}")
        min_threshold_carry_r_j, chebyshev_distance_list_carry_r_j = compute_min_threshold_carry(j, possible_threshold_values, training_data_carry_r_j, examples_train, partitions)


        logging.info(f"Minimal threshold for carry : w_{j} : {min_threshold_carry_r_j}")


        logging.info(f"Computing relational matrices for carry : w_{j}")
        data_carry_r_j_learning, matrices_carry_r_j_learning,_ = matrix_and_data_learning_carry(j, examples_train, training_data_carry_r_j, partitions, chebyshev_distance_list_carry_r_j, min_threshold_carry_r_j)

        logging.info(f"Inferring training examples for carry : w_{j}")
        output_possibility_distributions_carry_r_j = inference_with_training_data_carry(j, k,  tps, examples_train,  matrices_carry_r_j_learning,  training_data_carry_r_j)

        matrices["carry"][f"w_{j}"] = matrices_carry_r_j_learning[f"w_{j}"]

        min_threshold_carry = max(min_threshold_carry, min_threshold_carry_r_j)
        last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_j
    

    logging.info("Generating Possibilistic training data for y outputs...")
    training_data_y_outputs = possibilistic_training_data_y_outputs(k, rule_sets, partitions, tps, examples_train, output_possibility_distributions_c_tuples, last_output_possibility_distribution_carry_r_j)
    
    logging.info("Computing minimal threshold for y outputs...")
    min_threshold_y_outputs, chebyshev_distance_list_y_outputs = compute_min_threshold_y_outputs(possible_threshold_values, training_data_y_outputs, examples_train, partitions)
    logging.info(f"min threshold y_outputs: {min_threshold_y_outputs}")


    logging.info("Computing relational matrices for y outputs...")
    data_y_outputs_learning, matrices_y_outputs_learning,_ = matrix_and_data_learning_y_outputs(examples_train, training_data_y_outputs, partitions, chebyshev_distance_list_y_outputs,  min_threshold_y_outputs)
    
    matrices["y_outputs"] = matrices_y_outputs_learning
    
    del training_data_y_outputs, chebyshev_distance_list_y_outputs, data_y_outputs_learning
    gc.collect()
    
    return [min_threshold_c_tuple, min_threshold_carry, min_threshold_y_outputs], matrices

def compute_matrices_with_fixed_thresholds(thresholds, k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, probposs_transformation_method, check_above_min_threshold = True):
    use_thresholds = load_config.config["PiNeSy_use_thresholds"]
    matrices = {}
    matrices["c_tuples"] = {}
    matrices["carry"] = {}
    output_possibility_distributions_c_tuples = {}

    logging.info(f"New threshold for c_tuples: {thresholds[0]}...")
    logging.info(f"Generating Possibilistic training data for c_tuple : c_{k}")
    training_data_carry_tuples_c_k = possibilistic_training_data_c_tuples(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples_train, 
                                                                        probposs_transformation_method)
    
    min_threshold_c_tuple_c_k, chebyshev_distance_list_c_tuple_c_k = compute_min_threshold_c_tuples(k, possible_threshold_values, training_data_carry_tuples_c_k, examples_train, partitions)
    
    logging.info(f"Minimal threshold for c_tuple : c_{k} : {min_threshold_c_tuple_c_k}")

    if check_above_min_threshold:
        logging.info(f"Checking if given threshold {thresholds[0]} for c_k is strictly lower the min treshold for c_k : {min_threshold_c_tuple_c_k}...")
        if min_threshold_c_tuple_c_k > thresholds[0]:
            thresholds[0] = min_threshold_c_tuple_c_k
            logging.info(f"Not above, so we set it to the min treshold for c_k...")
        else:
            logging.info("No.")
    
    

    logging.info(f"Computing relational matrices for c_tuple : c_{k}")
    data_c_tuple_c_k_learning, matrices_c_tuple_c_k_learning, keep_examples_count_c_tuples_c_k_learning = matrix_and_data_learning_c_tuples(k, examples_train, training_data_carry_tuples_c_k, partitions, chebyshev_distance_list_c_tuple_c_k, thresholds[0])

    logging.info(f"Inferring training examples for c_tuple : c_{k}")
    output_possibility_distributions_c_tuple_c_k = inference_with_training_data_c_tuples(k, k,  tps, examples_train,  matrices_c_tuple_c_k_learning,  training_data_carry_tuples_c_k)
    
    matrices["c_tuples"][f"c_{k}"] = matrices_c_tuple_c_k_learning[f"c_{k}"]

    del training_data_carry_tuples_c_k, data_c_tuple_c_k_learning
    gc.collect()

    


    

    for example_index, example in enumerate(examples_train):
        output_possibility_distributions_c_tuples[example_index] = {}
        output_possibility_distributions_c_tuples[example_index][f"c_{k}"] = output_possibility_distributions_c_tuple_c_k[example_index]

    logging.info("Checking if the new threshold for c_k changes something for r_k")
    logging.info(f"Generating Possibilistic training data for carry : w_{k}")
    training_data_carry_r_k = possibilistic_training_data_carry(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples_train, 
                                                                        output_possibility_distributions_c_tuple_c_k)

    logging.info(f"Computing minimal threshold for carry : w_{k}")
    min_threshold_carry_r_k, chebyshev_distance_list_carry_r_k = compute_min_threshold_carry(k, possible_threshold_values, training_data_carry_r_k, examples_train, partitions)
    logging.info(f"Minimal threshold for carry : w_{k} : {min_threshold_carry_r_k}")
    

    

    if check_above_min_threshold:
        logging.info(f"Checking if given threshold {thresholds[1]} for r_* is strictly lower the min treshold for carry r_* : {min_threshold_carry_r_k}...")
        if min_threshold_carry_r_k > thresholds[1]:
            thresholds[1] = min_threshold_carry_r_k
            logging.info(f"Not above, so we set it to the min treshold for carry r_k...")
        else:
            logging.info("No.")

    logging.info(f"Computing relational matrices for carry : w_{k}")
    data_carry_r_k_learning, matrices_carry_r_k_learning, keep_examples_count_r_k_learning = matrix_and_data_learning_carry(k, examples_train, training_data_carry_r_k, partitions, chebyshev_distance_list_carry_r_k, thresholds[1])

    logging.info(f"Inferring training examples for carry : w_{k}")
    output_possibility_distributions_carry_r_k = inference_with_training_data_carry(k, k,  tps, examples_train,  matrices_carry_r_k_learning,  training_data_carry_r_k)

    matrices["carry"][f"w_{k}"] = matrices_carry_r_k_learning[f"w_{k}"]

    
    
    
    
    last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_k

    keep_examples_count_c_j_arr = {}
    keep_examples_count_r_j_arr = {}

    for j in range(k-1, 0, -1):
        logging.info(f"Generating Possibilistic training data for c_tuple : c_{j}")
        training_data_carry_tuples_c_j = possibilistic_training_data_c_tuples(j, 
                                                                            k, 
                                                                            rule_sets, 
                                                                            partitions, 
                                                                            tps, 
                                                                            examples_train, 
                                                                            probposs_transformation_method, 
                                                                            last_output_possibility_distribution_carry_r_j
                                                                            )
        
        min_threshold_c_tuple_c_j, chebyshev_distance_list_c_tuple_c_j = compute_min_threshold_c_tuples(j, possible_threshold_values, training_data_carry_tuples_c_j, examples_train, partitions)

        logging.info(f"Minimal threshold for c_tuple : c_{j} : {min_threshold_c_tuple_c_j}")

        if check_above_min_threshold:
            logging.info(f"Checking if given threshold {thresholds[0]} for c_j is strictly lower the min treshold for c_j : {min_threshold_c_tuple_c_j}...")
            if min_threshold_c_tuple_c_j > thresholds[0]:
                thresholds[0] = min_threshold_c_tuple_c_j
                logging.info(f"Not above, so we set it to the min treshold for c_j...")
            else:
                logging.info("No.")

        logging.info(f"Computing relational matrices for c_tuple : c_{j}")
        data_c_tuple_c_j_learning, matrices_c_tuple_c_j_learning, keep_examples_count_c_tuples_c_j_learning = matrix_and_data_learning_c_tuples(j, examples_train, training_data_carry_tuples_c_j, partitions, chebyshev_distance_list_c_tuple_c_j, thresholds[0])

        keep_examples_count_c_j_arr[j] = keep_examples_count_c_tuples_c_j_learning

        logging.info(f"Inferring training examples for c_tuple : c_{j}")
        output_possibility_distributions_c_tuple_c_j = inference_with_training_data_c_tuples(j, k,  tps, examples_train,  matrices_c_tuple_c_j_learning,  training_data_carry_tuples_c_j)
        
        matrices["c_tuples"][f"c_{j}"] = matrices_c_tuple_c_j_learning[f"c_{j}"].copy()
        
        for example_index, example in enumerate(examples_train):
            output_possibility_distributions_c_tuples[example_index][f"c_{j}"] = output_possibility_distributions_c_tuple_c_j[example_index]

        logging.info(f"Generating Possibilistic training data for carry : w_{j}")
        training_data_carry_r_j = possibilistic_training_data_carry(j,
                                                                            k,
                                                                            rule_sets,
                                                                            partitions,
                                                                            tps,
                                                                            examples_train,
                                                                            output_possibility_distributions_c_tuple_c_j)

        logging.info(f"Computing minimal threshold for carry : w_{j}")
        min_threshold_carry_r_j, chebyshev_distance_list_carry_r_j = compute_min_threshold_carry(j, possible_threshold_values, training_data_carry_r_j, examples_train, partitions)

        logging.info(f"Minimal threshold for carry : w_{j} : {min_threshold_carry_r_j}")


        if check_above_min_threshold:
            logging.info(f"Checking if given threshold {thresholds[1]} for r_* is strictly lower the min treshold for carry r_* : {min_threshold_carry_r_j}...")
            if min_threshold_carry_r_j > thresholds[1]:
                thresholds[1] = min_threshold_carry_r_j
                logging.info(f"Not above, so we set it to the min treshold for carry r_k...")
            else:
                logging.info("No.")

        logging.info(f"Computing relational matrices for carry : w_{j}")
        data_carry_r_j_learning, matrices_carry_r_j_learning, keep_examples_count_r_j_learning = matrix_and_data_learning_carry(j, examples_train, training_data_carry_r_j, partitions, chebyshev_distance_list_carry_r_j, thresholds[1])


        keep_examples_count_r_j_arr[j] = keep_examples_count_r_j_learning

        logging.info(f"Inferring training examples for carry : w_{j}")
        output_possibility_distributions_carry_r_j = inference_with_training_data_carry(j, k,  tps, examples_train,  matrices_carry_r_j_learning,  training_data_carry_r_j)

        matrices["carry"][f"w_{j}"] = matrices_carry_r_j_learning[f"w_{j}"]

        last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_j

        


    logging.info("Checking if the new threshold for  c_tuples and r_* changes something for y_outputs")
    logging.info("Generating Possibilistic training data for y outputs...")
    training_data_y_outputs = possibilistic_training_data_y_outputs(k, rule_sets, partitions, tps, examples_train, output_possibility_distributions_c_tuples, last_output_possibility_distribution_carry_r_j)
    
    logging.info("Computing minimal threshold for y outputs...")
    min_threshold_y_outputs, chebyshev_distance_list_y_outputs = compute_min_threshold_y_outputs(possible_threshold_values, training_data_y_outputs, examples_train, partitions)
    logging.info(f"min threshold y_outputs: {min_threshold_y_outputs}")
    if check_above_min_threshold:
        logging.info(f"Checking if given threshold {thresholds[2]} for y_outputs is strictly lower the min treshold for y_outputs: {min_threshold_y_outputs}...")
        if min_threshold_y_outputs > thresholds[2]:
            logging.info(f"Not above, so we set it to the min treshold for y_outputs...")
            thresholds[2] = min_threshold_y_outputs
        else:
            logging.info("No.")

    logging.info("Computing relational matrices for y outputs...")
    data_y_outputs_learning, matrices_y_outputs_learning, keep_examples_count_y_outputs = matrix_and_data_learning_y_outputs(examples_train, training_data_y_outputs, partitions, chebyshev_distance_list_y_outputs,  thresholds[2])
    
    matrices["y_outputs"] = matrices_y_outputs_learning

    

    keep_examples_count_all = {}

    keep_examples_count_all[f"c_{k}"] = keep_examples_count_c_tuples_c_k_learning
    keep_examples_count_all[f"w_{k}"] =  keep_examples_count_r_k_learning
    for j in range(k-1, 0, -1):
        keep_examples_count_all[f"w_{j}"] = keep_examples_count_r_j_arr[j]
        keep_examples_count_all[f"c_{j}"] = keep_examples_count_c_j_arr[j]
    for j in  training_data_y_outputs.keys():
        keep_examples_count_all[f"{j}"] = keep_examples_count_y_outputs[j]
    
    del training_data_y_outputs, min_threshold_y_outputs, chebyshev_distance_list_y_outputs, data_y_outputs_learning
    gc.collect()

    return thresholds, matrices, keep_examples_count_all
    

def finding_hyperparameters_thresholds(k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, examples_validation, probposs_transformation_method):
    min_improvement = load_config.config["PiNeSy_min_improvement"]
    stagnation = load_config.config["PiNeSy_stagnation"]

    use_thresholds = load_config.config["PiNeSy_use_thresholds"]

    if use_thresholds is False:
        return [1.01, 1.01, 1.01], [] # if we do not use thresholds, fix them to 1.01 (above 1.0)

    min_thresholds, matrices = get_minimal_thresholds(k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, probposs_transformation_method)
    
    best_thresholds = min_thresholds.copy()
    
    best_accuracy = evaluate_model(k,"Validation", examples_validation, matrices, rule_sets,  partitions, tps, probposs_transformation_method)["Accuracy_Validation"]
    search_ranges = [sorted([val for val in possible_threshold_values if val > min_thresh]) for min_thresh in min_thresholds]
    stagnation_counters = [0] * 3


    for i in range(3): 
        logging.info(f"Pertubing the {i+1}-th threshold...")
        
        for current_threshold in search_ranges[i]:
            if stagnation_counters[i] >= stagnation: 
                if i < 2:
                    logging.info("Stagnation of accuracy. Next threshold.")
                else:
                    logging.info("Stagnation of accuracy. Exit processing.")
                break  
            
            test_thresholds = best_thresholds.copy()
            test_thresholds[i] = current_threshold
            
            test_thresholds, matrices, keep_examples_count_all = compute_matrices_with_fixed_thresholds(test_thresholds, k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, probposs_transformation_method)
            current_accuracy = evaluate_model(k, "Validation", examples_validation, matrices, rule_sets,  partitions, tps, probposs_transformation_method)["Accuracy_Validation"]
            
            if current_accuracy > best_accuracy + min_improvement:
                best_accuracy = current_accuracy
                best_thresholds = test_thresholds
                print(f"New best with thresholds {test_thresholds} at accuracy {best_accuracy}")
                stagnation_counters[i] = 0  
            else:
                stagnation_counters[i] += 1
    logging.info(f"Found these thresholds for the task: {best_thresholds}")
    return best_thresholds, keep_examples_count_all

