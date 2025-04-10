import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import math
import json
import re

from nn import run_mnist_experiment

from mnistsudoku_rulebase import generate_rule_base_sudoku
from mnistsudoku_parser_files import parse_puzzles, load_txt_files_into_dict
from mnistsudoku_constraints import sudoku_build_set_constraints
from mnistsudoku_export import exportResultsSudoku
from mnistsudoku_inference import prepareInference, InferencePuzzles
from mnistsudoku_learning import prepare_learning_sudoku_1, prepare_learning_sudoku_2, possibilistic_learning_first_sets_of_rules, LearningParametersOfSecondset_of_rules
from mnistsudoku_find_hyperparameters import finding_hyperparameters_thresholds, find_min_thresholds_and_learning



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


def neural_learning(size, 
                    probposs_transformation_method,
                    puzzles_train_data,
                    puzzles_valid_data,
                    puzzles_test_data):
    out_features = 4 if size == 4 else 10

    all_train_data = []
    all_valid_data = []
    all_test_data = []
    for index_puzzle, puzzle in enumerate(puzzles_train_data):
        for i in range(size):
            for j in range(size):
                el = puzzles_train_data[index_puzzle]["grid"][i][j]
                all_train_data.append((el['image_data'], el['truth_label'], index_puzzle, i, j))
    
    for index_puzzle, puzzle in enumerate(puzzles_valid_data):
        for i in range(size):
            for j in range(size):
                el = puzzles_valid_data[index_puzzle]["grid"][i][j]
                all_valid_data.append((el['image_data'], el['truth_label'], index_puzzle, i, j))
    
    for index_puzzle, puzzle in enumerate(puzzles_test_data):
        for i in range(size):
            for j in range(size):
                el = puzzles_test_data[index_puzzle]["grid"][i][j]
                all_test_data.append((el['image_data'], el['truth_label'], index_puzzle, i, j))
    


    logging.info("neural learning...")
    file_name_train,  file_name_validation, file_name_test, accuracy_NN_test, accuracy_NN_val, file_checkpoint_model = run_mnist_experiment(0,
                                                                                                   0,
                                                                                                   0,
                                                                                                   out_features,
                                                                                                   all_train_data,
                                                                                                   all_valid_data,
                                                                                                   all_test_data)
    logging.info(f'file_name_train="{file_name_train}"')
    logging.info(f'file_name_validation="{file_name_validation}"')
    logging.info(f'file_name_test="{file_name_test}"')
    logging.info(f'accuracy_NN_test="{accuracy_NN_test}"')

    return file_name_train, file_name_validation, file_name_test,  accuracy_NN_test, accuracy_NN_val, file_checkpoint_model


def generate_examples_from_neural_data(size,
                                       probposs_transformation_method,
                                       puzzles_train_data,
                                       puzzles_valid_data,
                                       puzzles_test_data, 
                                       file_name_train, 
                                       file_name_validation, 
                                       file_name_test, 
                                       accuracy_NN_test):                                                                                      
    train_puzzle = {}
    validation_puzzle = {}
    test_puzzle = {}

    label_train_puzzle = {}
    label_validation_puzzle = {}
    label_test_puzzle = {}

    for k,v in puzzles_train_data.items():
        train_puzzle[k] = puzzles_train_data[k]
        label_train_puzzle[k] = puzzles_train_data[k]["truth"]
    
    

    for k,v in puzzles_valid_data.items():
        validation_puzzle[k] = puzzles_valid_data[k]
        label_validation_puzzle[k] = puzzles_valid_data[k]["truth"]
    
    
    
    for k,v in puzzles_test_data.items():
        test_puzzle[k] = puzzles_test_data[k]
        label_test_puzzle[k] = puzzles_test_data[k]["truth"]
    
    

    logging.info(f"loading train sudoku examples... from {file_name_train}")
    puzzles_map_train = make_examples_sudoku(file_name_train, train_puzzle)
    logging.info(f"loading validation sudoku examples...from {file_name_validation}")
    puzzles_map_valid = make_examples_sudoku(file_name_validation, validation_puzzle)
    logging.info(f"loading test sudoku examples... from {file_name_test}")
    puzzles_map_test = make_examples_sudoku(file_name_test, test_puzzle)

    return puzzles_map_train, puzzles_map_valid, puzzles_map_test, label_train_puzzle, label_validation_puzzle, label_test_puzzle



def experiment(size, split_dir, probposs_transformation_method):
    logging.info("Building the set F.")
    constraints = sudoku_build_set_constraints(size)

    logging.info("Generating the rule base...")
    attributes_set, domain_attributes_sets, first_sets_of_rules, second_set_of_rules = generate_rule_base_sudoku(
        constraints, size)
    
    if load_config.config["same_NN_as_DeepSoftLog"] is True:
        logging.info("CNN DeepSoftLog was not configured on MNIST-Sudoku.")
        sys.exit(0)




    logging.info("DATA DIRECTORY:" + str(split_dir))

    files_dict = load_txt_files_into_dict(split_dir)
    logging.info(f"Files in {split_dir}:")
    for key, value in files_dict.items():
        logging.info(f"{key}: {len(value)} characters")
    train = parse_puzzles(files_dict, size,  "train")
    valid = parse_puzzles(files_dict, size, "valid")
    test = parse_puzzles(files_dict, size, "test")

    
    
    start_time_learning = time.time()


    file_name_train, file_name_validation, file_name_test,  accuracy_NN_test, accuracy_NN_val, file_checkpoint_model = neural_learning(size, 
                                                                                                probposs_transformation_method,
                                                                                                train,
                                                                                                valid,
                                                                                                test)
    puzzles_map_train, puzzles_map_valid, puzzles_map_test, label_train_puzzle, label_validation_puzzle, label_test_puzzle = generate_examples_from_neural_data(size,
                                       probposs_transformation_method,
                                       train,
                                       valid,
                                       test, 
                                       file_name_train, 
                                       file_name_validation, 
                                       file_name_test, 
                                       accuracy_NN_test)
    
    
    thresholds = finding_hyperparameters_thresholds(constraints,
                                    first_sets_of_rules,
                                    second_set_of_rules,
                                    puzzles_map_train,
                                    label_train_puzzle,
                                    puzzles_map_valid,
                                    label_validation_puzzle,
                                    probposs_transformation_method)
    _, matrices_first_set_of_rules, matrix_second_set_of_rules,  tuples_partition_first_sets_of_rules, tuples_partition_second_set_of_rules, keep_examples_count_all  = find_min_thresholds_and_learning(constraints,
                                                        first_sets_of_rules,
                                                        second_set_of_rules,
                                                        puzzles_map_train,
                                                        label_train_puzzle,
                                                        puzzles_map_valid,
                                                        label_validation_puzzle,
                                                        probposs_transformation_method, 
                                                        thresholds)

    end_time_learning = time.time()

    elapsed_time_learning = end_time_learning - start_time_learning

    learning_time_obj = {'problem': 'mnist-sudoku', 'size': size, 'learning_time': elapsed_time_learning, 'keep_examples_count': json.dumps(keep_examples_count_all)}
    print(str(learning_time_obj))
    
    statistics_test, inference_time = InferencePuzzles(3, puzzles_map_test, label_test_puzzle,
                                                                    constraints, probposs_transformation_method,
                                                                    first_sets_of_rules, second_set_of_rules,
                                                                    matrices_first_set_of_rules,
                                                                    tuples_partition_first_sets_of_rules,
                                                                    matrix_second_set_of_rules,
                                                                    tuples_partition_second_set_of_rules)


    statistics = statistics_test
    statistics["id"] = split_dir
    statistics["accuracy_NN_test"] = accuracy_NN_test
    logging.info(statistics)
    filepath = (BASE_FILE_PATH_RESULTS +  str(probposs_transformation_method) + "/experiment::mnist-"+str(size)+"x"+str(size)+"/result-"
                    + str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + str(".csv"))

    directory = os.path.split(filepath)[0]
    os.makedirs(directory, exist_ok=True)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=statistics.keys())
        writer.writeheader()
        writer.writerow(statistics)