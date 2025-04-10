import re
import json
import numpy as np
from collections import defaultdict
import os
import pandas as pd
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

# Regular expression to match the first line of a segment.
# Example:
# 2024-12-23 18:18:39,170 - k: 1, n_train: 50000, n_valid: 10000, n_test: 10000, , id: 1, probposs_transformation_method: 1
start_line_regex = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - k:\s*(\d+),'
    r'\s*n_train:\s*\d+,\s*n_valid:\s*\d+,\s*n_test:\s*\d+,' 
    r'\s*id:\s*(\d+),\s*probposs_transformation_method:\s*(\d+)'
)

# Regular expression for the line with thresholds:
# Example:
# 2024-12-23 18:37:54,685 - Found these thresholds for the task: [np.float64(4.1193415637860074e-08), np.float64(4.1193415637860074e-08), np.float64(4.1193415637860074e-08)]
thresholds_line_regex = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - '
    r'Found these thresholds for the task: '
    r'\[np\.float64\(([0-9.e+\-]+)\),\s*'
    r'np\.float64\(([0-9.e+\-]+)\),\s*'
    r'np\.float64\(([0-9.e+\-]+)\)\]'
)





end_regex = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - \{"
    r"'number_of_Test_samples': (\d+),\s*"
    r"'Accuracy_Test': ([0-9.]+),\s*"
    r"'ambiguous_rate_Test': ([0-9.]+),\s*"
    r"'average_inference_time_Test': ([0-9.e+\-]+),\s*"
    r"'accuracy_NN_test': ([0-9.]+),\s*"
    r"'mnistadd_train_examples': (\d+),\s*"
    r"'mnistadd_valid_examples': (\d+),\s*"
    r"'mnistadd_test_examples': (\d+),\s*"
    r"'id': (\d+),\s*"
    r"'NN_n_train': (\d+),\s*"
    r"'NN_n_valid': (\d+),\s*"
    r"'NN_n_test': (\d+)"
    r"\}"
)

def parse_log_file(filename):
    """
    Reads the log file, identifies each segment, and extracts:
      - k
      - id
      - probposs_transformation_method
      - thresholds (from the 'Found these thresholds' line)
    Returns a list of dictionaries, each containing these fields.
    """

    results = []
    current_segment_info = None
    reading_segment = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            start_match = start_line_regex.search(line)
            if start_match:
                #print("start match!")
                # If we were already reading a segment, it implies we never found 
                # an end marker for the previous segment. Usually you would 
                # close it or discard it. For simplicity, we'll just open a new one.
                reading_segment = True
                current_segment_info = {
                    "k": int(start_match.group(2)),
                    "id": int(start_match.group(3)),
                    "probposs_transformation_method": int(start_match.group(4)),
                    "thresholds_found": None  # Will fill in once found
                }
                continue

            if reading_segment:
                # Check if this line contains thresholds
                thresh_match = thresholds_line_regex.search(line)
                # Check if there's a match

                if thresh_match:
                    #print("match!")
                    #print(thresh_match.groups())
                    
                    # Extract the threshold values from groups 2, 3, and 4
                    thr_values = [
                        float(thresh_match.group(2)),
                        float(thresh_match.group(3)),
                        float(thresh_match.group(4))
                    ]
                    
                    # Store the threshold values in the current segment info
                    current_segment_info["thresholds_found"]= thr_values
                    #print("Threshold values:", thr_values)
                    
                
                
                end_match = end_regex.search(line)
                if end_match:
                    # We have finished reading the current segment
                    results.append(current_segment_info)
                    # Reset
                    current_segment_info = None
                    reading_segment = False
    
    return results



def generate_latex_table(data):
    """Generate a LaTeX table from processed data."""
    table_header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{|c|c|c|c|c|}\n"
        "\\hline\n"
        "k & Approach & Threshold \\#1 & Threshold \\#2 & Threshold \\#3 \\\\\n"
        "\\hline\n"
    )
    table_footer = (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Average thresholds for MNIST Addition-$k$ problems with  $\\Pi$-Nesy-1 and $\\Pi$-Nesy-2.}\n"
        "\\label{tab:thresholds}\n"
        "\\end{table}\n"
    )
    
    table_rows = ""
    for row in sorted(data, key=lambda row: row["k"]):
        # Determine the approach based on row['pm']
        if row.get('pm') == 1:
            trans_method = "$\\Pi$-Nesy-1"
        elif row.get('pm') == 2:
            trans_method = "$\\Pi$-Nesy-2"
        else:
            trans_method = str(row.get('pm'))
        
        # Build the row string using the approach
        table_rows += f"{row['k']} & {trans_method} & {row['avg_1']} & {row['avg_2']} & {row['avg_3']} \\\\\n"
    
    latex_table = table_header + table_rows + table_footer

    print(latex_table)
    
    tex_path = os.path.join(load_config.config["aggregated_results_directory"], "mnist_add_thresholds_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table has been saved to '{tex_path}'.")
    
    csv_path = os.path.join(load_config.config["aggregated_results_directory"], "mnist_add_thresholds_table.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"CSV file has been saved to '{csv_path}'.")




def format_scientific(number):
    return f"{number:.2e}"

def AvgThresholds(data):
    # This dictionary will map (k, probposs_transformation_method) -> list of threshold triplets
    thresholds_by_k_prob = defaultdict(list)

    latex_data = []

    for entry in data:
        k = entry["k"]
        pm = entry["probposs_transformation_method"]
        thr = entry["thresholds_found"]
        thresholds_by_k_prob[(k, pm)].append(thr)
    
    for (k, pm) in sorted(thresholds_by_k_prob.keys()):
        all_thresholds = np.array(thresholds_by_k_prob[(k, pm)])
        
        means = np.mean(all_thresholds, axis=0)
        std_devs = np.std(all_thresholds, axis=0)

        avg_1, avg_2, avg_3 = means
        std_1, std_2, std_3 = std_devs

        print(f"(k={k}, probposs_transformation_method={pm}):")
        print(f"   Average threshold #1 = {format_scientific(avg_1)}, StdDev = {format_scientific(std_1)}")
        print(f"   Average threshold #2 = {format_scientific(avg_2)}, StdDev = {format_scientific(std_2)}")
        print(f"   Average threshold #3 = {format_scientific(avg_3)}, StdDev = {format_scientific(std_3)}")
        
        latex_data.append({
            "k": k,
            "pm": pm,
            "avg_1": f"{format_scientific(avg_1)} $\\pm$ {format_scientific(std_1)}",
            "avg_2": f"{format_scientific(avg_2)} $\\pm$ {format_scientific(std_2)}",
            "avg_3": f"{format_scientific(avg_3)} $\\pm$ {format_scientific(std_3)}",
        })

    generate_latex_table(latex_data)





if __name__ == "__main__":
    filename = "error-mnist-add.err"
    segments_data = parse_log_file(filename)

    AvgThresholds(segments_data)
    