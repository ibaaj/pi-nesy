import os
import sys
import pprint

from tools_input_vector import transform_probabilities


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from inference import  inference_minmax_product
from build_matrix import build_matrix_inference 
from possibilistic_learning import RuleParametersLearning, custom_range, get_threshold_map


def build_input_vector_c_tuples(j, k, attributes, rules, prob_dists, poss_dists, probposs_transformation_method):
    input_vector = []
    attributes_a = [att for att in attributes if att.startswith("a_")]
    poss_dict = transform_probabilities(attributes_a, prob_dists, probposs_transformation_method)
    for rule in rules:
        for attribute in attributes:
            if j == k:
                if len(rule["premise"][attribute]) > 0:
                    poss_values = poss_dict[attribute]
                    index = rule["premise"][attribute][0]
                    lambda_degree = poss_values[index]
                    rho_degree = max(poss_values[i] for i in range(0,len(poss_values)) if i != index)
                    input_vector.extend([lambda_degree, rho_degree])
                    break
            elif j != 0:
                if len(rule["premise"][attribute]) > 0:
                    if attribute.startswith("a_") is True:
                        poss_values = poss_dict[attribute]
                    else:
                        poss_values = poss_dists[attribute]
                    index = rule["premise"][attribute][0]
                    lambda_degree = poss_values[index]
                    rho_degree = max(poss_values[i] for i in range(0,len(poss_values)) if i != index)
                    input_vector.extend([lambda_degree, rho_degree])
                    break
            else:
                if len(rule["premise"][attribute]) > 0:
                    poss_values = poss_dists[attribute]
                    index = rule["premise"][attribute][0]
                    lambda_degree = poss_values[index]
                    rho_degree = max(poss_values[i] for i in [0,1] if i != index)
                    input_vector.extend([lambda_degree, rho_degree])
                    break

    return input_vector


def possibilistic_training_data_c_tuples(j, k, rule_sets, partitions, tps, examples, probposs_transformation_method, poss_dists = {}):
    training_data = {}
    training_data["c_" + str(j)] = []

    for example_index, example in enumerate(examples):
        prob_dists = {}
        
        
        input_vector = []
        output_vector = []
        if j == k:
            attributes = ["a_" + str(j), "a_" + str(k+j)]
            prob_dists = {
                "a_" + str(j): example["a_" + str(j)]["probdist"],
                "a_" + str(k+j): example["a_" + str(k+j)]["probdist"]
            }
            input_vector = build_input_vector_c_tuples(j, 
                                                    k, 
                                                    attributes, 
                                                    rule_sets["c_tuples"]["c_" + str(j)],
                                                    prob_dists,
                                                    {},
                                                    probposs_transformation_method)
            for subset_index, subset in enumerate(partitions["c_tuples"]["c_" + str(j)]):
                x_set_tuple = tps["c_tuples"]["c_" + str(j)][subset_index]
                
                if (example["labels"]["a_" + str(j)],example["labels"]["a_" + str(k+j)]) == x_set_tuple:
                    output_vector.append(1)
                else:
                    output_vector.append(0)
        elif j != 0:
            attributes = ["a_" + str(j), "a_" + str(k+j), "w_" + str(j+1)]
            prob_dists = {
                "a_" + str(j): example["a_" + str(j)]["probdist"],
                "a_" + str(k+j): example["a_" + str(k+j)]["probdist"]
            }
            
            input_vector = build_input_vector_c_tuples(j, 
                                                    k, 
                                                    attributes, 
                                                    rule_sets["c_tuples"]["c_" + str(j)],
                                                    prob_dists,
                                                    poss_dists[example_index],
                                                    probposs_transformation_method)

            for subset_index, subset in enumerate(partitions["c_tuples"]["c_" + str(j)]):
                x_set_tuple = tps["c_tuples"]["c_" + str(j)][subset_index]
                
                if (example["labels"]["a_" + str(j)],example["labels"]["a_" + str(k+j)], example["labels"]["w_" + str(j+1)]) == x_set_tuple:
                    output_vector.append(1)
                else:
                    output_vector.append(0)
        else:
            attributes = ["w_" + str(j+1)]
            prob_dists = {}
            
            input_vector = build_input_vector_c_tuples(j, 
                                                    k, 
                                                    attributes, 
                                                    rule_sets["c_tuples"]["c_" + str(j)],
                                                    prob_dists,
                                                    poss_dists[example_index],
                                                    probposs_transformation_method)
            for subset_index, subset in enumerate(partitions["c_tuples"]["c_" + str(j)]):
                x_set_tuple = tps["c_tuples"]["c_" + str(j)][subset_index]
                
                if example["labels"]["w_" + str(j+1)] == x_set_tuple:
                    output_vector.append(1)
                else:
                    output_vector.append(0)

        training_data["c_" + str(j)].append((input_vector, output_vector))
    return training_data

def inference_with_training_data_c_tuples(j, k,  tps, examples,  matrices_c_tuples_learning, training_data_c_tuples):
    output_possibility_distributions_c_tuples = {}

    for example_index, example in enumerate(examples):
        output_possibility_distributions_c_tuples[example_index] = {}
        output_possibility_distributions_c_tuples[example_index]["c_" + str(j)] = inference_minmax_product(matrices_c_tuples_learning["c_" + str(j)], training_data_c_tuples["c_" + str(j)][example_index][0], tps["c_tuples"]["c_" + str(j)])
    return output_possibility_distributions_c_tuples



def matrix_and_data_learning_c_tuples(j, examples_train, training_data_c_tuples, partitions, chebyshev_distance_list, threshold):
    data_c_tuples_learning = {}
    matrices_c_tuples_learning = {}

    print("calling RuleParametersLearning for c_" + str(j))
    
    data_c_tuples_learning[f"c_{j}"]  = RuleParametersLearning(
                                            examples_train,
                                            training_data_c_tuples[f"c_{j}"],
                                            partitions["c_tuples"][f"c_{j}"] ,
                                            chebyshev_distance_list[f"c_{j}"],
                                            threshold) 

    s_params, r_params, keep_examples_count, nabla = data_c_tuples_learning[f"c_{j}"]
    matrices_c_tuples_learning[f"c_{j}"]  = build_matrix_inference(partitions["c_tuples"][f"c_{j}"], s_params, r_params)
    
    return data_c_tuples_learning, matrices_c_tuples_learning, keep_examples_count



def compute_min_threshold_c_tuples(j, possible_threshold_values, training_data_c_tuples, examples_train, partitions):
    chebyshev_distance_list = {}
    threshold_dict = {}
    threshold_c_tuples = 0.0
    cd_list, t_dict = get_threshold_map(possible_threshold_values, examples_train, training_data_c_tuples[f"c_{j}"], partitions["c_tuples"][f"c_{j}"])
    chebyshev_distance_list[f"c_{j}"] = cd_list
    threshold_dict[f"c_{j}"] = t_dict
    threshold_c_tuples = min(key for key, value in t_dict.items() if value > 0)

    return (threshold_c_tuples, chebyshev_distance_list)