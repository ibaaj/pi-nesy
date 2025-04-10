from typing import List


def build_matrix_inference(partition: List[List[int]], rule_set_params_s: List[int], rule_set_params_r: List[int]) -> List[List[int]]:
    """
    Builds a matrix of the matrix relation
    for a given set of parallel possibilistic rules
    based on partition and rule set parameters.

    Parameters:
    - partition: A list of lists, where each sublist
     represents a set of x output values in the partition.
    - rule_set_params_s: A list of parameters s_i.
    - rule_set_params_r: A list of parameters r_i.

    Returns:
    - The matrix of the matrix relation - A list of lists.
    """
    matrix = []
    for x_set in partition:
        row = []
        for x_val in x_set:
            x_val_id = abs(x_val) - 1
            if x_val > 0:
                row.extend([rule_set_params_s[x_val_id], 1])
            else:
                row.extend([1, rule_set_params_r[x_val_id]])
        matrix.append(row)
    return matrix

def build_matrix_inference_first_set_of_rules(partition1: List[List[int]], first_rule_set_params_s: List[int], first_rule_set_params_r: List[int]) -> List[List[int]]:
    """
    Builds a matrix of the matrix relation
     for the first set of parallel possibilistic rules
     based on partition1 and the rule parameters of the rules in the first set of rules.

    Parameters:
    - partition1: A list of lists,
    where each sublist represents a set of x output values.
    - first_rule_set_params_s: A list of parameters s_i.
    - first_rule_set_params_r: A list of parameters r_i.

    Returns:
    - The matrix of the matrix relation  associated to the first set of rules
    A list of lists
    """
    return build_matrix_inference(partition1, first_rule_set_params_s, first_rule_set_params_r)

def build_matrix_inference_second_set_of_rules(partition2: List[List[int]], second_rule_set_params_s: List[int], second_rule_set_params_r: List[int]) -> List[List[int]]:
    """
    Builds a matrix of the matrix relation
     for the second set of parallel possibilistic rules
     based on partition2 and the rule parameters of the rules in the second set of rules.

    Parameters:
    - partition2: A list of lists,
    where each sublist represents a set of output x values
    - second_rule_set_params_s: A list of parameters s'_i.
    - second_rule_set_params_r: AA list of parameters s'_i.

    Returns:
    - The matrix of the matrix relation  associated to the second set of rules
    A list of lists
    """
    return build_matrix_inference(partition2, second_rule_set_params_s, second_rule_set_params_r)

