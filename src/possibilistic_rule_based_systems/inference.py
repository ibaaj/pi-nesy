import sys 
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config
"""
current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config
"""
from learning import minmax_product
from typing import Dict, List, Tuple, Any

epsilon = load_config.config["PiNeSy_epsilon"]



def inference_minmax_product(matrix: List[List[float]], input_vector: List[float], partition_subsets: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], float]:
    """
    Compute the min-max product of a given matrix and input vector,
    associating the result with partition subsets.

    Parameters:
    - matrix: 2D list of floats representing the matrix.
    - input_vector: List of floats representing the input vector.
    - partition_subsets: List of tuples, each representing a subset of the partition.

    Returns:
    - A dictionary mapping each partition subset to
     its corresponding value in the min-max product result.
    """

    output_vector = {}
    out_vec_min_max = minmax_product(matrix, input_vector)

    for row_index, partition_subset in enumerate(partition_subsets):
        output_vector[partition_subset] = out_vec_min_max[row_index]

    return output_vector


def infer_first_set_of_rules(input_vector: List[float], matrix: List[List[float]], partition_subsets_1: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], float]:
    # deprecated
    return inference_minmax_product(matrix, input_vector, partition_subsets_1)

def infer_second_set_of_rules(input_vector: List[float], matrix: List[List[float]], partition_subsets_2: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], float]:
    # deprecated 
    return inference_minmax_product(matrix, input_vector, partition_subsets_2)



