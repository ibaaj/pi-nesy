

import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import string
import argparse
import pickle 
import gc
import json
import re
import time


from nn import run_mnist_experiment

from mnistadd_generate_examples import generate_mnist_add_examples

from mnistadd_rulebase import generate_rule_base_mnist_add_problem


from mnistadd_c_tuples import (
        possibilistic_training_data_c_tuples, 
        inference_with_training_data_c_tuples, 
        build_input_vector_c_tuples, 
        matrix_and_data_learning_c_tuples,
        compute_min_threshold_c_tuples
)

from mnistadd_y_outputs import (
    possibilistic_training_data_y_outputs, 
    compute_min_threshold_y_outputs, 
    matrix_and_data_learning_y_outputs, 
    inference_with_training_data_y_outputs
)

from mnistadd_find_hyperparameters import finding_hyperparameters_thresholds, compute_matrices_with_fixed_thresholds

from mnistadd_evaluation import evaluate_model

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '..', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from partitions import build_partition, partition_tuples


parent_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[ logging.StreamHandler()  ])


BASE_FILE_PATH_RESULTS = load_config.config["results_mnist_addition_directory"]


def generating_mnist_add_problem(k, n_train, n_valid, n_test,  probposs_transformation_method):
    logging.info("Generating the rule base...")
    attributes_set, domain_attributes_set, rule_sets = generate_rule_base_mnist_add_problem(k)

    partitions = {}
    tps = {}

    logging.info("Building partitions for the c_tuples...")
    partitions["c_tuples"] = {}
    tps["c_tuples"] = {}
    for carry_c in rule_sets["c_tuples"].keys():
        partitions["c_tuples"][carry_c] = build_partition(rule_sets["c_tuples"][carry_c], carry_c)
        tps["c_tuples"][carry_c] = partition_tuples(partitions["c_tuples"][carry_c], rule_sets["c_tuples"][carry_c], carry_c)
    
    logging.info("Building partitions for the carry...")
    partitions["carry"] = {}
    tps["carry"] = {}
    for carry_r in rule_sets["carry"].keys():
        partitions["carry"][carry_r] = build_partition(rule_sets["carry"][carry_r], carry_r)
        tps["carry"][carry_r] = partition_tuples(partitions["carry"][carry_r], rule_sets["carry"][carry_r], carry_r)

    logging.info("Building partitions for the y_outputs...")
    
    partitions["y_outputs"] = {}
    tps["y_outputs"] = {}
    for y_i in rule_sets["y_outputs"].keys():
        partitions["y_outputs"][y_i] = build_partition(rule_sets["y_outputs"][y_i], y_i)
        tps["y_outputs"][y_i] = partition_tuples(partitions["y_outputs"][y_i], rule_sets["y_outputs"][y_i], y_i)

    

    if load_config.config["same_NN_as_DeepSoftLog"] is False:
        # PiNeSy CNN
        start_time_learning = time.time()
        logging.info("Learning the MNIST Net...")
    else:
        # DeepSoftLog CNN
        if load_config.config["DeepSoftLog_checkpointmnistaddfile"] != "":
            logging.info("The NN has already been trained. Retrieving stored model...")
        else:
            logging.info("Starting the NN training process..")
            
            best_accuracy_NN_val = -float("inf")
            best_accuracy_NN_test = -float("inf")
            best_checkpointmodel_file = ""

            for i in range(load_config.config["DeepSoftLog_CNN_training_iterations"]):
                logging.info(f"Iteration {i+1}/10: Training a new NN model...")
                file_name_train_c, file_name_valid_c, file_name_test_c, accuracy_NN_test_c, accuracy_NN_val_c, file_checkpoint_model = run_mnist_experiment(n_train, n_valid, n_test)
                logging.info(f"Training completed. Model achieved validation accuracy: {accuracy_NN_val_c:.4f}, test accuracy: {accuracy_NN_test_c:.4f}")
                
                if accuracy_NN_val_c > best_accuracy_NN_val:
                    logging.info(f"New model is better (prev validation: {best_accuracy_NN_val if best_accuracy_NN_val != -float('inf') else 'N/A'}, new: {accuracy_NN_val_c:.4f}; "
                                f"prev test: {best_accuracy_NN_test if best_accuracy_NN_test is not None else 'N/A'}, new: {accuracy_NN_test_c:.4f}). Updating best model files.")
                    best_accuracy_NN_val = accuracy_NN_val_c
                    best_accuracy_NN_test = accuracy_NN_test_c
                    best_checkpointmodel_file = file_checkpoint_model
                    
                else:
                    logging.info(f"New model is not better than the current best (best validation: {best_accuracy_NN_val:.4f}, related test accuracy: {best_accuracy_NN_test:.4f}). Keeping the best model.")
            
            logging.info(f"Training completed. Best model test accuracy: (best validation: {best_accuracy_NN_val:.4f}, related test accuracy: {best_accuracy_NN_test:.4f})")
            
            load_config.update_config("DeepSoftLog_checkpointmnistaddfile", best_checkpointmodel_file)
        start_time_learning = time.time() # Only data generation and possibilistic learning when same_NN_as_DeepSoftLog is true
    

    file_name_train, file_name_valid, file_name_test, accuracy_NN_test, accuracy_NN_val, file_checkpoint_model = run_mnist_experiment(n_train, n_valid, n_test)
    
    logging.info(f'file_name_train="{file_name_train}"')
    logging.info(f'file_name_valid="{file_name_valid}"')
    logging.info(f'file_name_test="{file_name_test}"')
    logging.info(f'accuracy_NN_test="{accuracy_NN_test}"')

    logging.info("Generating examples...")
    
    logging.info(f"Generating mnist addition train examples... from {file_name_train}")
    examples_train = generate_mnist_add_examples(k, file_name_train, n_train)
    logging.info(f"Generating mnist addition valid examples... from {file_name_valid}")
    examples_valid = generate_mnist_add_examples(k, file_name_valid, n_valid)
    logging.info(f"Generating mnist addition test examples... from {file_name_test}")
    examples_test = generate_mnist_add_examples(k, file_name_test, n_test)
    
    

    logging.info(f"TRAIN: {len(examples_train)} examples")
    logging.info(f"VALID: {len(examples_valid)} examples")
    logging.info(f"TEST: {len(examples_test)} examples")

    thresholds, _ = finding_hyperparameters_thresholds(k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, examples_valid, probposs_transformation_method)

    logging.info("Beginning possibilistic learning...")
    _, matrices, keep_examples_count_all = compute_matrices_with_fixed_thresholds(thresholds, k, attributes_set, domain_attributes_set, rule_sets, partitions, tps, examples_train, probposs_transformation_method, False)
    logging.info("End possibilistic learning.")

    end_time_learning = time.time()

    

    elapsed_time_learning = end_time_learning - start_time_learning

    learning_time_obj = {'problem': 'mnist-add', 'k': k, 'learning_time': elapsed_time_learning, 'keeped_examples_count': json.dumps(keep_examples_count_all)}
    print(str(learning_time_obj))

    logging.info(f"Evaluation on test data using the {len(examples_test)} test examples....")
    stats_test = evaluate_model(k,"Test", examples_test, matrices, rule_sets,  partitions, tps, probposs_transformation_method)



    stats_test["accuracy_NN_test"]  = accuracy_NN_test
    stats_test["mnistadd_train_examples"] = len(examples_train)
    stats_test["mnistadd_valid_examples"] = len(examples_valid)
    stats_test["mnistadd_test_examples"] = len(examples_test)
    

    return stats_test


def experiment(k, id, probposs_transformation_method):
    n_train = load_config.config["mnist_train_size_for_mnistadd"]
    n_valid = load_config.config["mnist_valid_size_for_mnistadd"]
    n_test = load_config.config["mnist_test_size_for_mnistadd"]
    

    logging.info(f"k: {k}, n_train: {n_train}, n_valid: {n_valid}, n_test: {n_test}, id: {id}, probposs_transformation_method: {probposs_transformation_method}")

    inference_runtime_total = []
    statistics_total = []
    
    statistics =  generating_mnist_add_problem(k, n_train,  n_valid, n_test, probposs_transformation_method)
    
    statistics["id"] = id
    statistics["NN_n_train"] = n_train
    statistics["NN_n_valid"] = n_valid
    statistics["NN_n_test"] = n_test
    
    logging.info(statistics)
    filepath = str(BASE_FILE_PATH_RESULTS) + "/method::pi-nesy-" + str(probposs_transformation_method) + "/" + str(k) + "/" + str(n_train) + "/result-" + str(k) + "-" + str(id) + "-" + str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + ".csv"

    directory = os.path.split(filepath)[0]
    os.makedirs(directory, exist_ok=True)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=statistics.keys())
        writer.writeheader()
        writer.writerow(statistics)


