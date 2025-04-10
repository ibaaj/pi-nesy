import re
import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

log_file_path = 'out-mnist-add.txt'


learning_times = {}
inference_times = {}

ks =  list(map(int, load_config.config["mnist_addition_k"].split()))
prob_possibilities = [1, 2]

for k in ks:
    for prob_poss in prob_possibilities:
        learning_times[(k, prob_poss)] = []
        inference_times[(k, prob_poss)] = []

attempt_pattern = re.compile(r'Attempt \d+ of 50 for k=(\d+), ID=(\d+), ProbPoss=(\d+)')
learning_time_pattern = re.compile(r'\'problem\': \'(?P<problem>\w+-\w+)\', \'k\': (?P<k>\d+), \'learning_time\': (?P<learning_time>\d+\.\d+)')
inference_time_pattern = re.compile(r'\'problem\': \'(?P<problem>\w+-\w+)\', \'k\': (?P<k>\d+), \'evaluation_type\': \'(?P<evaluation_type>\w+)\', \'example_index\': (?P<example_index>\d+), \'inference_time\': (?P<inference_time>\d+\.\d+)')

with open(log_file_path, 'r') as file:
    current_k = None
    current_prob_poss = None
    for line in file:
        attempt_match = attempt_pattern.match(line)
        if attempt_match:
            current_k = int(attempt_match.group(1))
            current_id = int(attempt_match.group(2))
            current_prob_poss = int(attempt_match.group(3))
            #print(f"Found attempt: k={current_k}, ID={current_id}, ProbPoss={current_prob_poss}")
        elif learning_time_pattern.search(line):
            learning_time = float(learning_time_pattern.search(line).group('learning_time'))
            learning_times[(current_k, current_prob_poss)].append(learning_time)
            #if len(learning_times[(current_k, current_prob_poss)]) % 10 == 0: 
                #print(f"Recorded learning time: {learning_time} for k={current_k}, ProbPoss={current_prob_poss}")
        elif inference_time_pattern.search(line):
            inference_time = float(inference_time_pattern.search(line).group('inference_time'))
            inference_times[(current_k, current_prob_poss)].append(inference_time)
            #if len(inference_times[(current_k, current_prob_poss)]) % 100 == 0:  # Print every 10000th inference time for brevity
                #print(f"Recorded inference time: {inference_time} for k={current_k}, ProbPoss={current_prob_poss}, total so far: {len(inference_times[(current_k, current_prob_poss)])}")

average_learning_times = {key: np.mean(value) if value else float('nan') for key, value in learning_times.items()}
std_learning_times = {key: np.std(value) if value else float('nan') for key, value in learning_times.items()}
average_inference_times = {key: np.mean(value) if value else float('nan') for key, value in inference_times.items()}
std_inference_times = {key: np.std(value) if value else float('nan') for key, value in inference_times.items()}


result_data = []
for k in ks:
    for prob_poss in prob_possibilities:
        learning_samples = len(learning_times.get((k, prob_poss), []))
        inference_samples = len(inference_times.get((k, prob_poss), []))
        result_data.append({
            'k': k,
            'ProbPoss': prob_poss,
            'Average Learning Time': f"{average_learning_times.get((k, prob_poss), 'N/A'):.2f}",
            'Std Learning Time': f"{std_learning_times.get((k, prob_poss), 'N/A'):.2f}",
            'Average Inference Time': f"{average_inference_times.get((k, prob_poss), 'N/A'):.3f}",
            'Std Inference Time': f"{std_inference_times.get((k, prob_poss), 'N/A'):.2f}",
            'Learning Samples': learning_samples,
            'Inference Samples': inference_samples
        })

result_df = pd.DataFrame(result_data)

print(result_df)

csv_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_add_inference_and_learning_time_summary.csv")
result_df.to_csv(csv_out, index=False)


result_data = []
for k in ks:
    # Compute training and test set sizes based on k
    train_size = int(  load_config.config["mnist_train_size_for_mnistadd"] / (2*k))
    test_size = int( load_config.config["mnist_test_size_for_mnistadd"] / (2*k))
    for prob_poss in prob_possibilities:
        approach = r"$\Pi$-NeSy-" + str(prob_poss)
        avg_learn = average_learning_times.get((k, prob_poss), float('nan'))
        std_learn = std_learning_times.get((k, prob_poss), float('nan'))
        avg_inf = average_inference_times.get((k, prob_poss), float('nan'))
        std_inf = std_inference_times.get((k, prob_poss), float('nan'))
        result_data.append({
            'k': k,
            'Training set size': train_size,
            'Test set size': test_size,
            'Approach': approach,
            'Learning time': f"{avg_learn:.2f} $\pm$ {std_learn:.2f}",
            'Inference time per test sample': f"{avg_inf:.3f} $\pm$ {std_inf:.2f}"
        })

result_df = pd.DataFrame(result_data)

# When outputting LaTeX, specify the custom column format
tex_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_add_inference_and_learning_time_summary.tex")
with open(tex_out, "w", encoding="utf-8") as f:
    f.write(result_df.to_latex(index=False, column_format='@{}cccccc@{}'))

