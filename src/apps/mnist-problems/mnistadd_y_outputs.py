import os
import sys


from tools_input_vector import transform_probabilities


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range

def build_input_vector_y_outputs(k, attributes, rules, poss_dists):
    upper_bound = 10
    input_vector = []
    for rule in rules: 
        for attribute in attributes: 
            if len(rule["premise"][attribute]) > 0:
                pset = rule["premise"][attribute]
                pcset =  rule["premise"]["excluded_values_of_" + str(attribute)]
                
                poss_values = poss_dists[attribute]
                
                lambda_degree = max([poss_values[id] for id in pset])
                rho_degree = max([poss_values[id] for id in pcset])
                
                input_vector.extend([lambda_degree, rho_degree])
                break
    return input_vector


def possibilistic_training_data_y_outputs(k, rule_sets, partitions, tps, examples, output_possibility_distributions_c_tuples, output_possibility_distributions_carry_r):
    training_data = {}
    for j in range(0,k+1):
        training_data["y_" + str(j)] = []

    for example_index, example in enumerate(examples):
        
        for j in range(k,-1,-1):
            input_vector = []
            output_vector = []

            if j != 0:
                attributes = ["c_" + str(j)]
                input_vector = build_input_vector_y_outputs(k,attributes, 
                                                    rule_sets["y_outputs"]["y_" + str(j)],
                                                    output_possibility_distributions_c_tuples[example_index]["c_" + str(j)])
            else:
                attributes = ["w_1"]

                input_vector = build_input_vector_y_outputs(k,attributes, 
                                                    rule_sets["y_outputs"]["y_" + str(j)],
                                                    output_possibility_distributions_carry_r[example_index])

            
            
            for subset_index, subset in enumerate(partitions["y_outputs"]["y_" + str(j)]):
                
                x_set_tuple = tps["y_outputs"]["y_" + str(j)][subset_index]
                
                if example["labels"]["y_" + str(j)] == x_set_tuple:
                    output_vector.append(1)
                else:
                    output_vector.append(0)
            

            training_data["y_" + str(j)].append((input_vector, output_vector))
    return training_data

def inference_with_training_data_y_outputs(k,  tps, examples,  matrices_y_outputs_learning, training_data_y_outputs):
    output_possibility_distributions_y_outputs = {}

    for example_index, example in enumerate(examples):
        output_possibility_distributions_y_outputs[example_index] = {}
        for j in range(k,-1,-1):
            output_possibility_distributions_y_outputs[example_index]["y_" + str(j)] = inference_minmax_product(matrices_y_outputs_learning["y_" + str(j)], training_data_y_outputs["y_" + str(j)][example_index][0], tps["y_outputs"]["y_" + str(j)])
    return output_possibility_distributions_y_outputs



def matrix_and_data_learning_y_outputs(examples_train, training_data_y_outputs, partitions, chebyshev_distance_list, threshold):
    data_y_outputs_learning = {}
    matrices_y_outputs_learning = {}

    keep_examples_count_y_outputs = {}
    
    for j in  training_data_y_outputs.keys():
        print("calling RuleParametersLearning for " + str(j))
        data_y_outputs_learning[j] = RuleParametersLearning(
                                            examples_train,
                                            training_data_y_outputs[j],
                                            partitions["y_outputs"][j],
                                            chebyshev_distance_list[j],
                                            threshold)
        s_params, r_params, keep_examples_count, nabla = data_y_outputs_learning[j]
        
        matrices_y_outputs_learning[j] = build_matrix_inference(partitions["y_outputs"][j], s_params, r_params)

        keep_examples_count_y_outputs[j] = keep_examples_count
    
    return data_y_outputs_learning, matrices_y_outputs_learning, keep_examples_count_y_outputs

def compute_min_threshold_y_outputs(possible_threshold_values, training_data_y_outputs, examples_train, partitions):
    chebyshev_distance_list = {}
    threshold_dict = {}
    threshold_y_outputs = 0.0
    for j in  training_data_y_outputs.keys():
        cd_list, t_dict = get_threshold_map(possible_threshold_values, examples_train, training_data_y_outputs[j], partitions["y_outputs"][j])
        chebyshev_distance_list[j] = cd_list
        threshold_dict[j] = t_dict
        threshold_y_outputs = max(threshold_y_outputs, min(key for key, value in t_dict.items() if value > 0))

    return (threshold_y_outputs, chebyshev_distance_list)