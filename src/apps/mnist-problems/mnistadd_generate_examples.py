import csv
import random
import sys
from typing import List, Dict, Any
from tools_make_examples import read_csv_file
import pprint
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



def generate_mnist_add_examples(k, file_name, N):
    records = read_csv_file(file_name)
    record_ids = list(range(len(records)))

    count_good = 0
    count_false = 0
    for id in range(len(record_ids)):
        if int(records[id]["label"]) == int(records[id]["predicted_label"]):
            count_good += 1
        else:
            count_false += 1
    logging.info("file: " + str(file_name))
    logging.info("good mnist examples:" + str(count_good) + "/" + str(len(record_ids)))
    logging.info("bad mnist examples:" + str(count_false) + "/" + str(len(record_ids)))
        
    examples_count = 0
    examples_count_good = 0
    examples_count_bad = 0
    examples = []
    while len(examples) < N and record_ids:
        if len(record_ids) < 2*k:
            break
        ids = random.sample(record_ids, 2*k)
        example = create_mnist_add_example(k, records, ids)
        examples.append(example)

        if example["IsCorrect"]:
            examples_count_good += 1
        else:
            examples_count_bad += 1
        examples_count += 1
        for id in ids:
            record_ids.remove(id)
    
    logging.info("good mnist-add examples:" + str(examples_count_good) + "/" + str(examples_count))
    logging.info("bad mnist-add examples:" + str(examples_count_bad) + "/" + str(examples_count))
    return examples


def create_mnist_add_example(k, records, ids):
    example = {}
    labels = {}
    

    for i in range(1,2*k+1):
        example["a_" + str(i)] = records[ids[i-1]]
        labels["a_" + str(i)] = records[ids[i-1]]["label"]
    c = 0
    res = 0
    pow_10 = 0

    result = 0
    carry = 0
    carries = {}
    digits = {}
    for j in range(k, 0, -1):
        current_sum = int(labels["a_" + str(j)]) + int(labels["a_" + str(k+j)]) + carry
        current_digit = current_sum % 10
        carry = current_sum // 10

        example[f'w_{j}'] = carry
        labels[f'w_{j}'] = carry
        example[f'y_{j}'] = current_digit
        labels[f'y_{j}'] = current_digit 
        
        result += current_digit * (10 ** (k-j))

    
    result += carry * (10 ** k)
    example[f'y_0'] = carry
    labels[f'y_0'] = carry
    
    example[f'w_0'] = carry
    labels[f'w_0'] = carry
    
    
    example["result"] = result
    example["labels"] = labels
    example["IsCorrect"] = all(int(records[id]["label"]) == int(records[id]["predicted_label"]) for id in ids)
    return example

