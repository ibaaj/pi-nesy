from typing import List, Dict, Set, Any


def build_attribute_values_for_rule_set(x_set: List[int], rules: List[Dict], output_attribute_name: str = "b") -> Set[Any]:
    """
    Builds the set of output attribute values of a x_set provided (set of integers in Lambda(n)), using the set of rules.

    Parameters:
    - x_set: List of integers representing the subset of the partition.
    - rules: List of dictionaries, each representing a rule.
    - output_attribute_name: String, the name of the output attribute, default is "b".

    Returns:
    - A subset of output attribute values: a subset of D_b constructed from the Q_* of the rules
    """
    attribute_values = set()
    for x_val in x_set:
        x_val_id = abs(x_val) - 1
        target_rule = rules[x_val_id]["conclusion"][output_attribute_name] if x_val > 0 else rules[x_val_id]["conclusion"][f"excluded_values_of_{output_attribute_name}"]
        if not attribute_values:
            attribute_values.update(target_rule)
        else:
            attribute_values.intersection_update(target_rule)

    return attribute_values

def build_partition(rules: List[Dict], output_attribute_name: str = "b") -> List[List[int]]:
    """
    Builds Lambda^(n): a partition based on a set of rules and an output attribute name.

    Parameters:
    - rules: List of dictionaries representing the rules.
    - output_attribute_name: String representing the output attribute's name.

    Returns:
    - Lambda^(n) : A list representing the partition (set of x_sets, where an x_set is a subset of integers of Lambda(n)).
    """
    partition = []
    for rule_id, rule in enumerate(rules):
        if rule_id == 0:
            if rule["conclusion"][output_attribute_name]:
                partition.append([1])
            if rule["conclusion"][f"excluded_values_of_{output_attribute_name}"]:
                partition.append([-1])
            continue

        new_partitions = []
        for x_set in partition:
            x_attribute_values = build_attribute_values_for_rule_set(x_set, rules, output_attribute_name)

            # Positive and negative attribute values based on current rule
            positive_values = set(rule["conclusion"][output_attribute_name])
            negative_values = set(rule["conclusion"][f"excluded_values_of_{output_attribute_name}"])
            positive_intersection = x_attribute_values.intersection(positive_values)
            negative_intersection = x_attribute_values.intersection(negative_values)

            if positive_intersection:
                new_partitions.append(x_set + [rule_id + 1])
            if negative_intersection:
                new_partitions.append(x_set + [-(rule_id + 1)])

        partition = new_partitions or partition

    return partition

def partition_tuples(partition: List[List[int]], rules: List[Dict], output_attribute_name: str = "b") -> List[Any]:
    """
    Converts the Lambda(n) partition into tuples of output attribute values based on the set of rules. 
    i.e., we get the true partition of D_b here.

    Parameters:
    - partition: List of lists, representing the partition.
    - rules: List of dictionaries, each dictionary is a rule.
    - output_attribute_name: String representing the name of the output attribute.
    Returns:
    - List of tuples representing the partition of D_b.
    """
    tuples = []
    for x_set in partition:
        attribute_values = build_attribute_values_for_rule_set(x_set, rules, output_attribute_name)
        tuples.extend(list(attribute_values))

    return tuples