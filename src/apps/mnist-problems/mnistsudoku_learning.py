import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import math
import logging

from nn import run_mnist_experiment



from mnistsudoku_rulebase import generate_rule_base_sudoku
from mnistsudoku_parser_files import parse_puzzles, load_txt_files_into_dict
from mnistsudoku_constraints import sudoku_build_set_constraints
from mnistsudoku_export import exportResultsSudoku
from mnistsudoku_inference import prepareInference, InferencePuzzles
from mnistsudoku_generate_examples import make_examples_sudoku

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config


parent_dir = os.path.join(current_dir, '../../', 'possibilistc_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))
from partitions import build_partition, partition_tuples, build_attribute_values_for_rule_set
from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference_first_set_of_rules, build_matrix_inference_second_set_of_rules
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range
from possprob_transformations import antipignistic_transformation, minimum_specificity_principle_transformation



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



def possibilistic_learning_first_sets_of_rules(constraints, lambda_array, rho_array, partition_array,
                                               target_output_vector_array,threshold_firsts):
    logging.info("Learning the parameters of the first sets of rules...")
    Parameters = {}
    logging.info("number of sets f : " + str(len(constraints)))
    v = 0
    for f in constraints:
        v += 1
        logging.info("set f: " + str(f) + " which is id=" + str(v) + " on " + str(len(constraints)))
        Parameters[f] = []
        training_data_first_set_rules = []
        index_set_examples_array = []
        for i in lambda_array[f].keys():
            lambda_degs = lambda_array[f][i]
            rho_degs = rho_array[f][i]

            iv = []
            for k in lambda_degs.keys():
                iv.append(lambda_degs[k])
                iv.append(rho_degs[k])
            training_data_first_set_rules.append((iv, target_output_vector_array[f][i]))
            partition_1 = partition_array[f][i]
            index_set_examples_array.append(i)


        chebyshev_distance_list, threshold_dict = get_threshold_map(possible_threshold_values, index_set_examples_array,
                                                                        training_data_first_set_rules, partition_1)
        
        print("calling RuleParametersLearning for f = " + str(f))

        s_params_first_set, r_params_first_set, keep_examples_count1, nabla1 = RuleParametersLearning(
                index_set_examples_array, training_data_first_set_rules,
                partition_1,
                chebyshev_distance_list,
                threshold_firsts
                )
        Parameters[f].append(s_params_first_set)
        Parameters[f].append(r_params_first_set)
    return Parameters

def prepare_learning_sudoku_1(constraints, puzzles_map_train, first_set_of_rules, ProbPossTransform):
    lambda_array = {}
    rho_array = {}
    partition_array = {}
    target_output_vector_array = {}
    for k in puzzles_map_train.keys():
        puzzle = puzzles_map_train[k]

        for f in constraints:
            i, j, ip, jp = f
            set_of_rules = first_set_of_rules[f]
            partition = build_partition(set_of_rules, "b_" + str(f))

            if f not in lambda_array:
                lambda_array[f] = {}
                rho_array[f] = {}
                partition_array[f] = {}
                target_output_vector_array[f] = {}

            partition_array[f][k] = partition
            lambda_array[f][k] = {}
            rho_array[f][k] = {}
            target_output_vector_array[f][k] = []

            for l in range(0, len(set_of_rules)):
                rule = set_of_rules[l]
                input_attribute_in_premise = next(iter(rule["premise"].keys())).split("_")
                input_attribute_in_premise_index_i = int(input_attribute_in_premise[1])
                input_attribute_in_premise_index_j = int(input_attribute_in_premise[2])
                target_singleton = rule["premise"][next(iter(rule["premise"].keys()))][0]

                if ProbPossTransform == 1:
                    poss = antipignistic_transformation(
                        puzzle[input_attribute_in_premise_index_i][input_attribute_in_premise_index_j]["probdist"])
                else:
                    poss = minimum_specificity_principle_transformation(
                        puzzle[input_attribute_in_premise_index_i][input_attribute_in_premise_index_j]["probdist"])

                lambda_array[f][k][l] = poss[target_singleton]
                rho_array[f][k][l] = max([poss[degid] for degid in range(0, len(poss)) if degid != target_singleton])

            for w in partition:
                q = list(build_attribute_values_for_rule_set(w, set_of_rules, "b_" + str(f)))[0]
                if q[0] == puzzle[i][j]["label"] and q[1] == puzzle[ip][jp]["label"]:
                    target_output_vector_array[f][k].append(1)
                else:
                    target_output_vector_array[f][k].append(0)

    return lambda_array, rho_array, partition_array, target_output_vector_array


def prepare_learning_sudoku_2(constraints, puzzles_map_train, label_puzzles_train, first_set_of_rules, lambda_array, rho_array,
                              partition_array,parameters_first_sets_of_rules, second_set_of_rules, ProbPossTransform):
    partition_2 = build_partition(second_set_of_rules, "c")
    TP = partition_tuples(partition_2, second_set_of_rules, "c")
    
    rule = second_set_of_rules[0]
    

    lambda_array2 = {}
    rho_array2 = {}
    target_output_vector_array2 = {}

    for k in puzzles_map_train.keys():
        puzzle = puzzles_map_train[k]
        partialLambdaPremise = 1
        partialRhoPremise = 0
        for f in constraints:
            lambda_degs = lambda_array[f][k]
            rho_degs = rho_array[f][k]

            iv = []
            for u in lambda_degs.keys():
                iv.append(lambda_degs[u])
                iv.append(rho_degs[u])
            partition_1 = partition_array[f][k]
            TP = partition_tuples(partition_1, first_set_of_rules[f], "b_" + str(f))
            m = build_matrix_inference_first_set_of_rules(partition_1, parameters_first_sets_of_rules[f][0],
                                                          parameters_first_sets_of_rules[f][1])
            OutputVector = inference_minmax_product(m, iv, TP)

            currentLambdaProposition = max([OutputVector[d] for d in rule["premise"]["b_" + str(f)]])
            currentRhoProposition = max([OutputVector[d] for d in rule["premise"]["excluded_values_of_b_" + str(f)]])
            partialLambdaPremise = min(partialLambdaPremise, currentLambdaProposition)
            partialRhoPremise = max(partialRhoPremise, currentRhoProposition)
        lambda_array2[k] = partialLambdaPremise
        rho_array2[k] = partialRhoPremise
        
        target_output_vector_array2[k] = [1, 0]  if label_puzzles_train[k] == 1 else [0, 1]
    return partition_2, lambda_array2, rho_array2, target_output_vector_array2


def LearningParametersOfSecondset_of_rules(partition_2, lambda_array_2, rho_array_2,
                                                                            target_output_vector_array_2,
                                                                            threshold_second):
    logging.info("Learning the parameters of the second set of rules...")

    Parameters2 = []
    training_data_for_second_set = []
    index_set_examples = []
    for key in lambda_array_2:
        input_vector = [lambda_array_2[key], rho_array_2[key]]
        output_vector = target_output_vector_array_2[key]
        training_data_for_second_set.append((input_vector, output_vector))
        index_set_examples.append(key)
    
    chebyshev_distances_2, threshold_to_errors_2 = get_threshold_map(
        possible_threshold_values, index_set_examples, training_data_for_second_set, partition_2)
 
    
    print("calling RuleParametersLearning for second set of rules of MNIST-sudoku ")

    s_params_2, r_params_2, kept_example_count_2, nabla_2 = RuleParametersLearning(
            index_set_examples, training_data_for_second_set, partition_2, chebyshev_distances_2, threshold_second)

    Parameters2.append(s_parameters)
    Parameters2.append(r_parameters)
    return Parameters2