import numpy as np
import random


def antipignistic_transformation(probdist_dict):
    """
    Transforms a probability distribution to a possibility distribution
    using the antipignistic transformation method.

    :param probdist_dict: A dictionary representing the probability distribution.
    :return: A dictionary representing the possibility distribution.
    """
    prob_sorted_keys = sorted(probdist_dict, key=probdist_dict.get, reverse=True)
    poss_dist_dict = {}
    for prob_key in prob_sorted_keys:
        poss_dist_dict[prob_key] = sum(min(probdist_dict[k], probdist_dict[prob_key]) for k in prob_sorted_keys)
    return poss_dist_dict

def minimum_specificity_principle_transformation(probdist_dict):
    """
    Transforms a probability distribution to a possibility distribution
    based on the minimum specificity principle.

    :param probdist_dict: A dictionary representing the probability distribution.
    :return: A dictionary representing the possibility distribution.
    """
    prob_sorted_keys = sorted(probdist_dict, key=probdist_dict.get, reverse=True)
    poss_dist_dict = {}
    for i, prob_key in enumerate(prob_sorted_keys):
        poss_dist_dict[prob_key] = sum(min(probdist_dict[k], probdist_dict[prob_key]) for k in prob_sorted_keys[i:])
    return poss_dist_dict




def test_probposs_transformations():
    # Test loop
    num_tests = 10000  # Number of tests
    num_events = 5  # Number of events in the probability distribution
    for test_num in range(num_tests):
        # Generate random probability distribution
        random_probs = np.random.rand(num_events)
        random_probs /= random_probs.sum()  # Normalize to sum to 1
        prob_dist_dict = {f"event_{i}": prob for i, prob in enumerate(random_probs)}
         

        # Apply both sets of functions
        antipignistic_possiblity_dist = antipignistic_transformation(prob_dist_dict)
        minimum_possiblity_dist = minimum_specificity_principle_transformation(prob_dist_dict)

        # Print the distributions 
        print(f"Test #{test_num+1}")
        print("Probability Distribution:", prob_dist_dict)
        print("Antipignistic Possibility Distribution:", antipignistic_possiblity_dist)
        print("Minimum Specificity Principle Possibility Distribution:", minimum_possiblity_dist)
        print("-" * 50)