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
from mnistsudoku_generate_examples import make_examples_sudoku

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config
from rocauc import ROC_AUC

parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))
from partitions import build_partition, partition_tuples, build_attribute_values_for_rule_set
from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference_first_set_of_rules, build_matrix_inference_second_set_of_rules
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range
from possprob_transformations import antipignistic_transformation, minimum_specificity_principle_transformation




def prepareInference(constraints, first_set_of_rules, parameters_first_sets_of_rules, partition_array,
                     second_set_of_rules, parameters_second_set_of_rules, partition_2):
    matrices_first_set_of_rules = {}
    tuples_partition_first_sets_of_rules = {}
    for f in constraints:
        tuples_partition_first_sets_of_rules[f] = partition_tuples(partition_array[f][0], first_set_of_rules[f], 
                                                                   "b_" + str(f))
        matrices_first_set_of_rules[f] = build_matrix_inference_first_set_of_rules(partition_array[f][0],
                                                                                   parameters_first_sets_of_rules[f][0],
                                                                                   parameters_first_sets_of_rules[f][1])

    tuples_partition_second_set_of_rules = partition_tuples(partition_2, second_set_of_rules,  "c")
    matrix_second_set_of_rules = build_matrix_inference_second_set_of_rules(partition_2, parameters_second_set_of_rules[0],
                                                                            parameters_second_set_of_rules[1])

    return tuples_partition_first_sets_of_rules, matrices_first_set_of_rules, tuples_partition_second_set_of_rules, matrix_second_set_of_rules


def InferencePuzzles(Evaluation_type_id, PUZZLES_MAP, LABEL_PUZZLE, constraints, ProbPossTransform, first_set_of_rules,
                     second_set_of_rules, matrices_first_set_of_rules, tuples_partition_first_sets_of_rules,
                     matrix_second_set_of_rules, tuples_partition_second_set_of_rules):
    total = len(PUZZLES_MAP)
    total_examples = len(PUZZLES_MAP)
    size = 9 if len(constraints) == 810 else 4

    correct_predictions_count = 0
    incorrect_predictions_count = 0
    ambiguous_predictions_count = 0

    inference_runtime = []
    elapsed_time_inference_array = []
    output_possibility_distributions = []
    y_true = []

    for k in PUZZLES_MAP.keys():
        start_time_inference = time.time()
        puzzle = PUZZLES_MAP[k]
        partialLambdaPremise = 1
        partialRhoPremise = 0
        for f in constraints:
            set_of_rules = first_set_of_rules[f]

            iv = []
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
                iv.append(poss[target_singleton])
                iv.append(max([poss[degid] for degid in range(0, len(poss)) if degid != target_singleton]))

            OutputVector = inference_minmax_product(matrices_first_set_of_rules[f], iv,
                                                    tuples_partition_first_sets_of_rules[f])

            rule = second_set_of_rules[0]
            currentLambdaProposition = max([OutputVector[d] for d in rule["premise"]["b_" + str(f)]])
            currentRhoProposition = max([OutputVector[d] for d in rule["premise"]["excluded_values_of_b_" + str(f)]])
            partialLambdaPremise = min(partialLambdaPremise, currentLambdaProposition)
            partialRhoPremise = max(partialRhoPremise, currentRhoProposition)
        iv = [partialLambdaPremise, partialRhoPremise]
        outcome_second = inference_minmax_product(matrix_second_set_of_rules, iv, tuples_partition_second_set_of_rules)

        v = max(outcome_second, key=outcome_second.get)

        end_time_inference = time.time()

        elapsed_time_inference = end_time_inference - start_time_inference

        elapsed_time_inference_array.append(elapsed_time_inference)

        inference_runtime.append(end_time_inference - start_time_inference)

        if Evaluation_type_id == 3:
            example_inference_time_obj = {'problem': 'mnist-sudoku', 'size': size, 'evaluation_type': "Test", 'example_index': k, 'inference_time': elapsed_time_inference}
            print(str(example_inference_time_obj))


        output_possibility_distributions.append(outcome_second)
        y_true.append(1 if LABEL_PUZZLE[k] == 1 else 0)
        # Evaluate prediction
        predicted_result = max(outcome_second, key=outcome_second.get)
        if len([value for value in outcome_second.values() if value == outcome_second[predicted_result]]) > 1:
            print("Ambiguous prediction.")
            incorrect_predictions_count += 1
            ambiguous_predictions_count += 1
            continue
        
        
        

        if (v == 1 and LABEL_PUZZLE[k] == 1) or (v == 0 and LABEL_PUZZLE[k] == 0):
            correct_predictions_count += 1
        else:
            incorrect_predictions_count += 1
            
    average_roc_auc, roc_auc_dict = ROC_AUC(output_possibility_distributions, y_true)

    # Calculate statistics
    accuracy = correct_predictions_count / total_examples * 100
    ambiguous_rate = ambiguous_predictions_count / total_examples * 100

    average_inference_time = sum(elapsed_time_inference_array)/len(PUZZLES_MAP)

    print("size = " + str(size))

    if Evaluation_type_id == 1:
        print(
            f"Training Evaluation: Correct Predictions: {correct_predictions_count}, Incorrect Predictions: {incorrect_predictions_count}")
        print(f"Accuracy on Training Data: {accuracy}%")
    elif Evaluation_type_id == 2:
        print(
            f"Validation Evaluation: Correct Predictions: {correct_predictions_count}, Incorrect Predictions: {incorrect_predictions_count}")
        print(f"Accuracy on Validation Data: {accuracy}%")
    elif Evaluation_type_id == 3:
        print(
            f"Testing Evaluation: Correct Predictions: {correct_predictions_count}, Incorrect Predictions: {incorrect_predictions_count}")
        print(f"Accuracy on Test Data: {accuracy}%")
    
    print(f"Average inference time: {average_inference_time}")

    if Evaluation_type_id == 1:
        return {
            "number_of_train_samples": total_examples,
            "accuracy_train": accuracy,
            "ambiguous_rate_train": ambiguous_rate,
            "average_roc_auc_train": average_roc_auc,
            "roc_auc_dict_train": str(roc_auc_dict)
        }, inference_runtime
    elif Evaluation_type_id == 2:
        return {
            "number_of_valid_samples": total_examples,
            "accuracy_valid": accuracy,
            "ambiguous_rate_valid": ambiguous_rate,
            "average_roc_auc_valid": average_roc_auc,
            "roc_auc_dict_valid": str(roc_auc_dict)
        }, inference_runtime
    else:
        return {
            "number_of_test_samples": total_examples,
            "accuracy_test": accuracy,
            "ambiguous_rate_test": ambiguous_rate,
            "average_roc_auc_test": average_roc_auc,
            "roc_auc_dict_test": str(roc_auc_dict)
        }, inference_runtime
