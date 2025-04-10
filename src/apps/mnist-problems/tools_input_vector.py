import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'possibilistic_rule_based_systems')
sys.path.append(os.path.abspath(parent_dir))

from possprob_transformations import antipignistic_transformation, minimum_specificity_principle_transformation




def transform_probabilities(attributes, prob_dists, mode):
    transformation = antipignistic_transformation if mode == 1 else minimum_specificity_principle_transformation
    poss_dict = {}
    for att in attributes:
        poss_dict[att] = transformation(prob_dists[att])
    return poss_dict


