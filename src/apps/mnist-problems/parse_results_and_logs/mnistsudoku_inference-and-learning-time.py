import re
import numpy as np
import pandas as pd
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

# Define the log file path
log_file_path = 'out-mnist-sudoku.txt'

# Initialize dictionaries to hold data
learning_times = {}
inference_times = {}

dimensions = [int(s.split("::")[1]) for s in load_config.config["mnist_sudoku_dimensions"].split()]
prob_possibilities = [1, 2]

# Initialize the data structures
for dimension in dimensions:
    for prob_poss in prob_possibilities:
        learning_times[(dimension, prob_poss)] = []
        inference_times[(dimension, prob_poss)] = []

# Regular expression patterns for parsing
attempt_pattern = re.compile(r'Attempt \d+ of 50 for Dimension=(\d+), SplitDir=.*, ProbPoss=(\d+)')
learning_time_pattern = re.compile(r'\'problem\': \'mnist-sudoku\', \'size\': (?P<size>\d+), \'learning_time\': (?P<learning_time>\d+\.\d+)')
inference_time_pattern = re.compile(r'\'problem\': \'mnist-sudoku\', \'size\': (?P<size>\d+), \'evaluation_type\': \'(?P<evaluation_type>\w+)\', \'example_index\': (?P<example_index>\d+), \'inference_time\': (?P<inference_time>\d+\.\d+)')

# Read and parse the log file
with open(log_file_path, 'r') as file:
    current_dimension = None
    current_prob_poss = None
    for line in file:
        attempt_match = attempt_pattern.match(line)
        if attempt_match:
            current_dimension = int(attempt_match.group(1))
            current_prob_poss = int(attempt_match.group(2))
            #print(f"Found attempt: Dimension={current_dimension}, ProbPoss={current_prob_poss}")
        elif learning_time_pattern.search(line):
            learning_time = float(learning_time_pattern.search(line).group('learning_time'))
            learning_times[(current_dimension, current_prob_poss)].append(learning_time)
            #print(f"Recorded learning time: {learning_time} for Dimension={current_dimension}, ProbPoss={current_prob_poss}")
        elif inference_time_pattern.search(line):
            inference_time = float(inference_time_pattern.search(line).group('inference_time'))
            inference_times[(current_dimension, current_prob_poss)].append(inference_time)
            #if len(inference_times[(current_dimension, current_prob_poss)]) % 10000 == 0:  # Print every 10000th inference time for brevity
                #print(f"Recorded inference time: {inference_time} for Dimension={current_dimension}, ProbPoss={current_prob_poss}, total so far: {len(inference_times[(current_dimension, current_prob_poss)])}")

# Calculate averages and standard deviations
average_learning_times = {key: np.mean(value) if value else float('nan') for key, value in learning_times.items()}
std_learning_times = {key: np.std(value) if value else float('nan') for key, value in learning_times.items()}
average_inference_times = {key: np.mean(value) if value else float('nan') for key, value in inference_times.items()}
std_inference_times = {key: np.std(value) if value else float('nan') for key, value in inference_times.items()}

# Prepare data for display
result_data = []
for k in dimensions:
    for prob_poss in prob_possibilities:
        result_data.append({
            'k': k,
            'ProbPoss': prob_poss,
            'Average Learning Time': f"{average_learning_times.get((k, prob_poss), 'N/A'):.2f}",
            'Std Learning Time': f"{std_learning_times.get((k, prob_poss), 'N/A'):.2f}",
            'Average Inference Time': f"{average_inference_times.get((k, prob_poss), 'N/A'):.2f}",
            'Std Inference Time': f"{std_inference_times.get((k, prob_poss), 'N/A'):.2f}",
            "Learning samples": len(learning_times[(k, prob_poss)]),
            "Inference samples": len(inference_times[(k, prob_poss)])

        })

# Convert to DataFrame for a better display
result_df = pd.DataFrame(result_data)

# Display the DataFrame
print(result_df)


csv_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_inference_and_learning_time_summary.csv")
result_df.to_csv(csv_out, index=False)


result_data = []
for dimension in dimensions:
    # Define the "Size" column as "4x4" or "9x9"
    size_str = f"{dimension}x{dimension}"
    # Use constant training and test set sizes as specified
    train_size = 2*load_config.config["mnist_train_size_for_mnistsudoku"]
    test_size = 2*load_config.config["mnist_validtest_size_for_mnistsudoku"]
    for prob_poss in prob_possibilities:
        approach = r"$\Pi$-NeSy-" + str(prob_poss)
        avg_learn = average_learning_times.get((dimension, prob_poss), float('nan'))
        std_learn = std_learning_times.get((dimension, prob_poss), float('nan'))
        avg_inf = average_inference_times.get((dimension, prob_poss), float('nan'))
        std_inf = std_inference_times.get((dimension, prob_poss), float('nan'))
        result_data.append({
            'Size': size_str,
            'Training set size': train_size,
            'Test set size': test_size,
            'Approach': approach,
            'Learning time': f"{avg_learn:.2f} $\pm$ {std_learn:.2f}",
            'Inference time per test sample': f"{avg_inf:.2f} $\pm$ {std_inf:.2f}"
        })

result_df = pd.DataFrame(result_data)

# Output the DataFrame to CSV as before
csv_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_inference_and_learning_time_summary.csv")
result_df.to_csv(csv_out, index=False)

# Output to LaTeX with the desired column format
tex_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_inference_and_learning_time_summary.tex")
with open(tex_out, "w", encoding="utf-8") as f:
    f.write(result_df.to_latex(index=False, column_format='@{}cccccc@{}'))
