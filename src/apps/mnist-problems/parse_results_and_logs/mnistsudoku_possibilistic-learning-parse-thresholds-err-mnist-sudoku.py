import re
import json
import numpy as np
from collections import defaultdict
import os
import sys
import pandas as pd

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

file_path = 'error-mnist-sudoku.err'


prob_poss = 1
output_json = []

thresholds_pattern = r"Found these thresholds for the task: \[np\.float64\(([^)]+)\), np\.float64\(([^)]+)\)\]"
dimension_split_pattern = r"dimension::(\d+).*split::(\d+)"
result_line_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - ({.*})"


with open(file_path, 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    thresholds_match = re.search(thresholds_pattern, line)
    if thresholds_match:
        
        threshold_1 = float(thresholds_match.group(1))
        threshold_2 = float(thresholds_match.group(2))

        # Search for result line in the next few lines
        result_found = False
        for j in range(i + 1, min(i + 10, len(lines))):  # Look up to 10 lines ahead
            result_match = re.search(result_line_pattern, lines[j])
            if result_match:
                raw_dict_str = result_match.group(1)
                id_match = re.search(dimension_split_pattern, raw_dict_str)
                if id_match:
                    # Extract dimension and split
                    dimension = int(id_match.group(1))
                    split = int(id_match.group(2))

                    # Create JSON object
                    json_object = {
                        "Dimension": dimension,
                        "Split": split,
                        "ProbPoss": prob_poss,
                        "Threshold_1": threshold_1,
                        "Threshold_2": threshold_2,
                    }
                    output_json.append(json_object)
                    result_found = True
                    prob_poss = 2 if prob_poss == 1 else 1
                    break  # Stop searching once result is found

        if not result_found:
            print(f"No result line found for thresholds at line {i}.")


# Group data by (Dimension, ProbPoss)
grouped_data = defaultdict(lambda: {"array_threshold_1": [], "array_threshold_2": []})

for item in output_json:
    key = (item["Dimension"], item["ProbPoss"])
    grouped_data[key]["array_threshold_1"].append(item["Threshold_1"])
    grouped_data[key]["array_threshold_2"].append(item["Threshold_2"])

# Create the new JSON structure
grouped_json = []

for (dimension, prob_poss), thresholds in grouped_data.items():
    array_threshold_1 = thresholds["array_threshold_1"]
    array_threshold_2 = thresholds["array_threshold_2"]

    json_item = {
        "Dimension": dimension,
        "ProbPoss": prob_poss,
        "array_threshold_1": array_threshold_1,
        "mean_threshold_1": np.mean(array_threshold_1),
        "std_threshold_1": np.std(array_threshold_1),
        "count_threshold_1": len(array_threshold_1),
        "array_threshold_2": array_threshold_2,
        "mean_threshold_2": np.mean(array_threshold_2),
        "std_threshold_2": np.std(array_threshold_2),
        "count_threshold_2": len(array_threshold_2)
        
    }
    grouped_json.append(json_item)



# Group configuration counts for the caption.
dim_counts = {}
for item in grouped_json:
    config = f"{item['Dimension']}x{item['Dimension']}"
    cnt1 = item['count_threshold_1']
    cnt2 = item['count_threshold_2']
    if config in dim_counts:
        existing_cnt1, existing_cnt2 = dim_counts[config]
        if existing_cnt1 != cnt1 or existing_cnt2 != cnt2:
            print(f"Warning: Mismatch in counts for configuration {config}: "
                  f"existing ({existing_cnt1}, {existing_cnt2}) vs new ({cnt1}, {cnt2})")
    else:
        dim_counts[config] = (cnt1, cnt2)

# Build a descriptive caption string.
if len(dim_counts) == 1:
    # Only one configuration
    for conf, (cnt1, cnt2) in dim_counts.items():
        dim_caption = f"In the {conf} configuration, averages for $b_{{(i,j,i',j')}}$ are computed over {cnt1} observations and for $c$ over {cnt2} observations"
elif len(dim_counts) == 2:
    # Two configurations: write one sentence using (resp. ...) format.
    keys = sorted(dim_counts.keys())
    (cnt1_a, cnt2_a) = dim_counts[keys[0]]
    # We assume counts for both threshold metrics are the same for both approaches.
    dim_caption = f"In the {keys[0]} configuration (resp. {keys[1]} configuration), averages for $b_{{(i,j,i',j')}}$ are computed over {cnt1_a} observations and for $c$ over {cnt2_a} observations"
else:
    # Fallback: join all configurations.
    dim_caption = "; ".join([
        f"In the {conf} configuration, averages for $b_{{(i,j,i',j')}}$ are computed over {cnt1} observations and for $c$ over {cnt2} observations"
        for conf, (cnt1, cnt2) in dim_counts.items()
    ])

latex_content = ""
latex_content += "\\begin{table}[h!]\n"
latex_content += "\\centering\n"
latex_content += "\\begin{tabular}{@{}llll@{}}\n"
latex_content += "\\toprule\n"
latex_content += "Dimension & Approach & Average threshold for $b_{(i,j,i',j')}$ & Average threshold for $c$  \\\\ \\midrule\n"

for item in grouped_json:
    dimension = f"{item['Dimension']}x{item['Dimension']}"
    approach = r"$\Pi$-NeSy-1" if item["ProbPoss"] == 1 else r"$\Pi$-NeSy-2"
    
    mean_threshold_1 = f"{item['mean_threshold_1']:.3e} $\\pm$ {item['std_threshold_1']:.3e}"
    mean_threshold_2 = f"{item['mean_threshold_2']:.3e} $\\pm$ {item['std_threshold_2']:.3e}"
    
    latex_content += f"{dimension} & {approach} & {mean_threshold_1} & {mean_threshold_2} \\\\\n"
    
latex_content += "\\bottomrule\n"
latex_content += "\\end{tabular}\n"
latex_content += f"\\caption{{Average thresholds for MNIST-Sudoku problems used for selecting reliable training data during possibilistic learning. {dim_caption} for both approaches.}}\n"
latex_content += f"\\label{{tab:mnist-sudoku-thresholds}}"
latex_content += "\\end{table}\n\n"


print(latex_content)

df = pd.DataFrame(grouped_json)
csv_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_thresholds.csv")
df.to_csv(csv_out, index=False)
print(f"CSV results saved to {csv_out}")

tex_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_thresholds.tex")
with open(tex_out, "w", encoding="utf-8") as f:
    f.write(latex_content)
print(f"LaTeX table saved to {tex_out}")