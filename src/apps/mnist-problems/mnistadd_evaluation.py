import os
import sys
import pprint
import time

from tools_input_vector import transform_probabilities




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

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))


from inference import infer_first_set_of_rules, inference_minmax_product
from build_matrix import build_matrix_inference_first_set_of_rules, build_matrix_inference_second_set_of_rules
from possibilistic_learning import RuleParametersLearning, get_threshold_map, custom_range



def possibilistic_inference(k, examples, matrices, rule_sets, partitions, tps, probposs_transformation_method):
    output_possibility_distributions_c_tuples = {}
    training_data_carry_tuples_c_k = possibilistic_training_data_c_tuples(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples, 
                                                                        probposs_transformation_method)
    
    
    output_possibility_distributions_c_tuple_c_k = inference_with_training_data_c_tuples(k, k,  tps, examples,  matrices["c_tuples"],  training_data_carry_tuples_c_k)
    

    for example_index, example in enumerate(examples):
        output_possibility_distributions_c_tuples[example_index] = {}
        output_possibility_distributions_c_tuples[example_index][f"c_{k}"] = output_possibility_distributions_c_tuple_c_k[example_index]
    
    del training_data_carry_tuples_c_k

    
    training_data_carry_r_k = possibilistic_training_data_carry(k, 
                                                                        k, 
                                                                        rule_sets, 
                                                                        partitions, 
                                                                        tps, 
                                                                        examples, 
                                                                        output_possibility_distributions_c_tuple_c_k)

    
    output_possibility_distributions_carry_r_k = inference_with_training_data_carry(k, k,  tps, examples,  matrices["carry"],  training_data_carry_r_k)

    del training_data_carry_r_k

    last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_k

    for j in range(k-1, 0, -1):
        training_data_carry_tuples_c_j = possibilistic_training_data_c_tuples(j, 
                                                                            k, 
                                                                            rule_sets, 
                                                                            partitions, 
                                                                            tps, 
                                                                            examples, 
                                                                            probposs_transformation_method, 
                                                                            last_output_possibility_distribution_carry_r_j
                                                                            )
        
        
        output_possibility_distributions_c_tuple_c_j = inference_with_training_data_c_tuples(j, k,  tps, examples, matrices["c_tuples"],  training_data_carry_tuples_c_j)


        del training_data_carry_tuples_c_j

        for example_index, example in enumerate(examples):
            output_possibility_distributions_c_tuples[example_index][f"c_{j}"] = output_possibility_distributions_c_tuple_c_j[example_index]

        training_data_carry_r_j = possibilistic_training_data_carry(j,
                                                                            k,
                                                                            rule_sets,
                                                                            partitions,
                                                                            tps,
                                                                            examples,
                                                                            output_possibility_distributions_c_tuple_c_j)


        output_possibility_distributions_carry_r_j = inference_with_training_data_carry(j, k,  tps, examples,  matrices["carry"],  training_data_carry_r_j)
        last_output_possibility_distribution_carry_r_j = output_possibility_distributions_carry_r_j

        del training_data_carry_r_j
    

    training_data_y_outputs = possibilistic_training_data_y_outputs(k, rule_sets, partitions, tps, examples, output_possibility_distributions_c_tuples, last_output_possibility_distribution_carry_r_j)

    output_possibility_distributions_y_outputs = inference_with_training_data_y_outputs(k,  tps, examples,  matrices["y_outputs"], training_data_y_outputs)

    return output_possibility_distributions_y_outputs
    





def evaluate_model(k, evaluation_type, examples, matrices, rule_sets, partitions, tps, probposs_transformation_method):
    correct_predictions_count = 0
    incorrect_predictions_count = 0
    ambiguous_predictions_count = 0

    elapsed_time_inference_array = []

    print("k = " + str(k))
    print("Evaluation type:" + str(evaluation_type))
    
    for example_index, example in enumerate(examples):
        start_time_inference = time.time()
        
        output_possibility_distributions_y_outputs = possibilistic_inference(k, [example], matrices, rule_sets, partitions,  tps, probposs_transformation_method)[0]

        end_time_inference = time.time()

        elapsed_time_inference = end_time_inference - start_time_inference

        elapsed_time_inference_array.append(elapsed_time_inference)

        if evaluation_type ==  "Test":
            example_inference_time_obj = {'problem': 'mnist-add', 'k': k, 'evaluation_type': evaluation_type, 'example_index': example_index, 'inference_time': elapsed_time_inference}
            print(str(example_inference_time_obj))

        ambiguous_prediction = False
        correct_prediction = True
        array_result = {}
        pow_10 = 0
        res = 0
        for j in range(k,-1,-1):
            predicted_result_j = max(output_possibility_distributions_y_outputs["y_" + str(j)], key=output_possibility_distributions_y_outputs["y_" + str(j)].get)
            if len([value for value in output_possibility_distributions_y_outputs["y_" + str(j)].values() if value == output_possibility_distributions_y_outputs["y_" + str(j)][predicted_result_j]]) > 1:
                print(f"Ambiguous prediction for y_{j}")
                ambiguous_prediction = True 
                break
            if predicted_result_j != example["y_" + str(j)]:
                correct_prediction = False
            
            res += predicted_result_j*(10**pow_10)
            pow_10 += 1
            
        if ambiguous_prediction is True:
            incorrect_predictions_count += 1
            ambiguous_predictions_count += 1
            continue
        

        if correct_prediction:
            correct_predictions_count += 1
        else:
            incorrect_predictions_count += 1
            

    total_examples = len(examples)
    accuracy = correct_predictions_count / total_examples * 100
    ambiguous_rate = ambiguous_predictions_count / total_examples * 100

    average_inference_time = sum(elapsed_time_inference_array)/len(examples)


    print("k = " + str(k))
    print(
        f"{evaluation_type} Evaluation: Correct Predictions: {correct_predictions_count}, Incorrect Predictions: {incorrect_predictions_count}")
    print(f"Accuracy on {evaluation_type} Data: {accuracy}%")
    print(f"Average inference time: {average_inference_time}")

    # Return collected statistics
    return {
        f"number_of_{evaluation_type}_samples": total_examples,
        f"Accuracy_{evaluation_type}": accuracy,
        f"ambiguous_rate_{evaluation_type}": ambiguous_rate,
        f"average_inference_time_{evaluation_type}": average_inference_time
    }