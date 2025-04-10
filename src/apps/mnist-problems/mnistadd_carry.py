import os
import sys
import pprint
import time
from tools_input_vector import transform_probabilities


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range


def build_input_vector_carry(attributes, rules, poss_dists):
    input_vector = []
    for rule in rules: 
        for attribute in attributes:  
            if len(rule["premise"][attribute]) > 0:
                poss_values = poss_dists[attribute]
                lambda_degree = max([poss_values[id] for id in rule["premise"][attribute]])
                rho_degree = max([poss_values[id] for id in rule["premise"][f"excluded_values_of_{attribute}"]])
                input_vector.extend([lambda_degree, rho_degree])
                break
    return input_vector


def possibilistic_training_data_carry(j, k,  rule_sets, partitions, tps, examples, poss_dists):
    training_data = {}
    training_data["w_" + str(j)] = []

    for example_index, example in enumerate(examples):
            input_vector = []
            output_vector = []
            
            attributes = ["c_" + str(j)]
            input_vector = build_input_vector_carry(attributes, 
                                                    rule_sets["carry"]["w_" + str(j)],
                                                    poss_dists[example_index])
            

            for subset_index, subset in enumerate(partitions["carry"]["w_" + str(j)]):
                x_set_tuple = tps["carry"]["w_" + str(j)][subset_index]
                if example["labels"]["w_" + str(j)] == x_set_tuple:
                    
                    output_vector.append(1)
                else:
                    output_vector.append(0)
            

            training_data["w_" + str(j)].append((input_vector, output_vector))
    return training_data

def inference_with_training_data_carry(j, k,  tps, examples,  matrices_carry_learning, training_data_carry):
    output_possibility_distributions_carry = {}
    for example_index, example in enumerate(examples):
        output_possibility_distributions_carry[example_index] = {}
        output_possibility_distributions_carry[example_index]["w_" + str(j)] = inference_minmax_product(matrices_carry_learning["w_" + str(j)], training_data_carry["w_" + str(j)][example_index][0], tps["carry"]["w_" + str(j)])
    return output_possibility_distributions_carry



def matrix_and_data_learning_carry(j, examples_train, training_data_carry, partitions, chebyshev_distance_list, threshold):
    data_carry_learning = {}
    matrices_carry_learning = {}

    print("calling RuleParametersLearning for w_" + str(j))

    data_carry_learning[f"w_{j}"] = RuleParametersLearning(
                                        examples_train,
                                        training_data_carry[f"w_{j}"],
                                        partitions["carry"][f"w_{j}"],
                                        chebyshev_distance_list[f"w_{j}"],
                                        threshold
                                        ) 

    s_params, r_params, keep_examples_count, nabla = data_carry_learning[f"w_{j}"]
    matrices_carry_learning[f"w_{j}"] = build_matrix_inference(partitions["carry"][f"w_{j}"], s_params, r_params)
    
    return data_carry_learning, matrices_carry_learning, keep_examples_count

def compute_min_threshold_carry(j, possible_threshold_values, training_data_carry, examples_train, partitions):
    chebyshev_distance_list = {}
    threshold_dict = {}
    threshold_carry = 0.0
    
    cd_list, t_dict = get_threshold_map(possible_threshold_values, examples_train, training_data_carry[f"w_{j}"], partitions["carry"][f"w_{j}"])
    chebyshev_distance_list[f"w_{j}"] = cd_list
    threshold_dict[f"w_{j}"] = t_dict
    threshold_carry = min(key for key, value in t_dict.items() if value > 0)

    return (threshold_carry, chebyshev_distance_list)