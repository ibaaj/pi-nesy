import re
import json
import ast
import numpy as np
from math import floor
import pandas as pd
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

MNIST_TRAIN_IMAGES_USED = load_config.config["mnist_train_size_for_mnistadd"]

MNIST_POSSIBLE_K = list(map(int, load_config.config["mnist_addition_k"].split()))

PROBABILITY_POSSIBILITY_TRANSFORMATION = [1, 2]

def parse_log_into_segments(filepath):
    """
    Reads the logfile and splits it into segments based on lines starting with:
        Attempt <something>
    Each new 'Attempt' line starts a new segment.
    
    Returns a list of segments, where each segment is simply a list of lines 
    (strings) belonging to that segment.
    """
    re_attempt = re.compile(r'^Attempt\s+\d+\s+of\s+50\s+for\s+k=\d+,\s+ID=\d+,\s+ProbPoss=\d+.*$')

    segments = []
    current_segment = []
    hitFirstSegment = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            # If we hit a new "Attempt" line, that marks the end of the current segment
            if re_attempt.match(line_stripped):
                hitFirstSegment = True
                if current_segment:
                    segments.append(current_segment)
                current_segment = [line_stripped]
            else:
                if current_segment is not None and hitFirstSegment is True:
                    current_segment.append(line_stripped)
            

    # End of file: if we have an unfinished segment, store it
    if current_segment:
        segments.append(current_segment)

    return segments


def analyze_segment(segment):
    """
    Given a list of lines (one segment), we want to extract:
    
      - attempt_line:         The first line in the segment (i.e., "Attempt ...").
      - last_validation_line: The LAST "Validation Evaluation: ..." line found in this segment.
      - parameters_lines:     All lines AFTER the last validation line 
                             but BEFORE the 'problem' line that contains 'learning_time': ....
      - problem_line:         The line that starts with "{'problem': 'mnist-add'..." 
                             and includes 'learning_time': ... (the last one found if multiple).
      - pi_nesy_line:         Any line starting with "Pi-Nesy-" (the last one if multiple).
    """
    re_validation_eval = re.compile(r'^Validation Evaluation: Correct Predictions:\s*\d+,\s*Incorrect Predictions:\s*\d+')
    re_problem_learning_time = re.compile(r'^\{\s*\'problem\':\s*\'mnist-add\'.*\'learning_time\':.*\}$')
    re_pi_nesy = re.compile(r'^Pi-Nesy-.*$')

    attempt_line = segment[0] if segment else None

    last_validation_line = None
    last_validation_idx = None
    problem_line = None
    problem_line_idx = None
    pi_nesy_line = None

    # Identify indices for the last validation line, last problem line with learning_time, last Pi-Nesy
    for i, line in enumerate(segment):
        if re_validation_eval.match(line):
            last_validation_line = line
            last_validation_idx = i

        if re_pi_nesy.match(line):
            pi_nesy_line = line

        if re_problem_learning_time.match(line):
            problem_line = line
            problem_line_idx = i

    # lines AFTER last_validation but BEFORE problem_line
    parameters_lines = []
    if (last_validation_idx is not None) and (problem_line_idx is not None):
        if problem_line_idx > last_validation_idx:
            parameters_lines = segment[last_validation_idx+1 : problem_line_idx]

    return {
        "attempt_line": attempt_line,
        "last_validation_line": last_validation_line,
        "parameters_lines": parameters_lines,
        "problem_line": problem_line,
        "pi_nesy_line": pi_nesy_line
    }



def parse_attempt_line(line):
    """
    Example: 'Attempt 1 of 50 for k=1, ID=3, ProbPoss=2'
    Returns (k, ID, ProbPoss) as integers or (None, None, None) if not matched.
    """
    if not line:
        return None, None, None
    match = re.match(r'^Attempt\s+\d+\s+of\s+50\s+for\s+k=(\d+),\s+ID=(\d+),\s+ProbPoss=(\d+)', line)
    if match:
        k_str, id_str, prob_str = match.groups()
        return int(k_str), int(id_str), int(prob_str)
    return None, None, None

def parse_problem_line(problem_line):
    """
    Example:
      {'problem': 'mnist-add', 'k': 1, 'learning_time': 1269.5595800876617, 'keeped_examples_count': '{"c_1": 173, "w_1": 1291, "y_0": 1291, "y_1": 173}'}
    We want to parse out keeped_examples_count, which is a JSON string inside the dict.
    Returns that dictionary or {} if something fails.
    """
    if not problem_line:
        return {}
    try:
        d = ast.literal_eval(problem_line)  # parse as Python dict
        keeped_str = d.get('keeped_examples_count', '{}')
        keeped_dict = json.loads(keeped_str)
        return {
            "keeped_examples_count": keeped_dict
        }
    except:
        return {}

def parse_pi_nesy_line(pi_line):
    """
    Example:
      Pi-Nesy-2 stats for mnist-add-1 | id:3 | Average memory usage: 577.14 MB, Maximum memory usage: 840.38 MB
    We'll parse out version=2, dataset=mnist-add, k=1, id=3, average_memory_mb=577.14, maximum_memory_mb=840.38
    """
    if not pi_line:
        return {}
    pattern = (
        r'^Pi-Nesy-(\d+)\s+stats\s+for\s+([A-Za-z0-9_-]+)-(\d+)\s*\|\s*id:(\d+)\s*\|\s*'
        r'Average memory usage:\s*([\d\.]+)\s*MB,\s*Maximum memory usage:\s*([\d\.]+)\s*MB'
    )
    match = re.match(pattern, pi_line)
    if not match:
        return {}
    version_str, dataset_str, k_str, id_str, avg_str, max_str = match.groups()
    return {
        "version": int(version_str),
        "dataset": dataset_str,
        "k": int(k_str),
        "id": int(id_str),
        "average_memory_mb": float(avg_str),
        "maximum_memory_mb": float(max_str),
    }

def parse_s_r_parameters(parameters_lines):
    """
    From the lines between last_validation_line and the problem line, 
    gather s/r parameters for each "c_j", "w_j", "y_j".
    Returns a dict like:
    {
      "c_1": { "s_parameters": [...], "r_parameters": [...] },
      "w_1": { ... },
      "y_0": { ... },
      ...
    }
    """
    rule_parameters = {}
    current_symbol = None

    re_calling = re.compile(r'^calling RuleParametersLearning for ([cwy]_\d+)$')
    re_sparams = re.compile(r'^s_parameters learned:\[(.*)\]$')
    re_rparams = re.compile(r'^r_parameters learned:\[(.*)\]$')

    count_s = 0
    count_r = 0
    

    for line in parameters_lines:
        # Check if it's a "calling" line
        m_call = re_calling.match(line)
        if m_call:
            current_symbol = m_call.group(1)  # e.g. "c_1", "w_1", "y_1"
            if current_symbol not in rule_parameters:
                rule_parameters[current_symbol] = {
                    "s_parameters": [],
                    "r_parameters": []
                }
            continue

        if current_symbol:
            m_s = re_sparams.match(line)
            if m_s:
                count_s += 1
                raw = m_s.group(1)
                arr = [float(x.strip()) for x in raw.split(',') if x.strip() != '']
                rule_parameters[current_symbol]["s_parameters"] = arr
                continue

            m_r = re_rparams.match(line)
            if m_r:
                count_r += 1
                raw = m_r.group(1)
                arr = [float(x.strip()) for x in raw.split(',') if x.strip() != '']
                rule_parameters[current_symbol]["r_parameters"] = arr
                continue

    return rule_parameters

def reArrange(data):
    for element in data:
        # Initialize "master" attributes for parameters and "master" keep_examples_count
        master_params = {"c": {"s_parameters": [], "r_parameters": []},
                        "w": {"s_parameters": [], "r_parameters": []},
                        "y": {"s_parameters": [], "r_parameters": []}}

        master_counts = {"c": [], "w": [], "y": []}

        
        for key, value in element["parameters"].items():
            if key.startswith("c_"):
                master_params["c"]["s_parameters"].extend(value["s_parameters"])
                master_params["c"]["r_parameters"].extend(value["r_parameters"])
            elif key.startswith("w_"):
                master_params["w"]["s_parameters"].extend(value["s_parameters"])
                master_params["w"]["r_parameters"].extend(value["r_parameters"])
            elif key.startswith("y_"):
                master_params["y"]["s_parameters"].extend(value["s_parameters"])
                master_params["y"]["r_parameters"].extend(value["r_parameters"])

        
        element["parameters"]["c_master"] = master_params["c"]
        element["parameters"]["w_master"] = master_params["w"]
        element["parameters"]["y_master"] = master_params["y"]

        
        for key, count in element["problem"]["keeped_examples_count"].items():
            if key.startswith("c_"):
                master_counts["c"].append(count)
            elif key.startswith("w_"):
                master_counts["w"].append(count)
            elif key.startswith("y_"):
                master_counts["y"].append(count)
        
        
       
        element["problem"]["c_master_keep_examples_count"] = master_counts["c"]
        element["problem"]["w_master_keep_examples_count"] = master_counts["w"]
        element["problem"]["y_master_keep_examples_count"] = master_counts["y"]

    

    return data

def ProperResults(data):
    
    ProbPosses = PROBABILITY_POSSIBILITY_TRANSFORMATION
    ks = MNIST_POSSIBLE_K

    new_json = []

    # Group data by (k, ProbPoss)
    for k in ks:
        for ProbPoss in ProbPosses:

            # Compute N_train
            N_train = floor(MNIST_TRAIN_IMAGES_USED / (2 * k))

            # Filter elements matching (k, ProbPoss)
            filtered = [el for el in data if el["k"] == k and el["ProbPoss"] == ProbPoss]

            if not filtered:
                continue

            arrays = {
                "s_parameters_c_master": [],
                "r_parameters_c_master": [],
                "s_parameters_w_master": [],
                "r_parameters_w_master": [],
                "s_parameters_y_master": [],
                "r_parameters_y_master": [],
                "average_memory_mb": [],
                "maximum_memory_mb": [],
                "c_master_keep_examples_count": [],
                "w_master_keep_examples_count": [],
                "y_master_keep_examples_count": []
            }

            for el in filtered:
                arrays["s_parameters_c_master"].extend(el["parameters"]["c_master"]["s_parameters"])
                arrays["r_parameters_c_master"].extend(el["parameters"]["c_master"]["r_parameters"])
                arrays["s_parameters_w_master"].extend(el["parameters"]["w_master"]["s_parameters"])
                arrays["r_parameters_w_master"].extend(el["parameters"]["w_master"]["r_parameters"])
                arrays["s_parameters_y_master"].extend(el["parameters"]["y_master"]["s_parameters"])
                arrays["r_parameters_y_master"].extend(el["parameters"]["y_master"]["r_parameters"])
                arrays["average_memory_mb"].append(el["pi_nesy"]["average_memory_mb"])
                arrays["maximum_memory_mb"].append(el["pi_nesy"]["maximum_memory_mb"])
                
                arrays["c_master_keep_examples_count"].extend(el["problem"]["c_master_keep_examples_count"])
                arrays["w_master_keep_examples_count"].extend(el["problem"]["w_master_keep_examples_count"])
                arrays["y_master_keep_examples_count"].extend(el["problem"]["y_master_keep_examples_count"])
            
            
            
            # Compute averages, standard deviations, and percentages
            avg_c_master_keep = np.mean(arrays["c_master_keep_examples_count"])
            avg_w_master_keep = np.mean(arrays["w_master_keep_examples_count"])
            avg_y_master_keep = np.mean(arrays["y_master_keep_examples_count"])

            # Compute averages and standard deviations
            results = {
                "k": k,
                "ProbPoss": ProbPoss,
                #"array_s_parameters_c_master": arrays["s_parameters_c_master"],
                "avg_s_parameters_c_master": np.mean(arrays["s_parameters_c_master"]),
                "std_s_parameters_c_master": np.std(arrays["s_parameters_c_master"]),
                #"array_r_parameters_c_master": arrays["r_parameters_c_master"],
                "avg_r_parameters_c_master": np.mean(arrays["r_parameters_c_master"]),
                "std_r_parameters_c_master": np.std(arrays["r_parameters_c_master"]),
                #"array_s_parameters_w_master": arrays["s_parameters_w_master"],
                "avg_s_parameters_w_master": np.mean(arrays["s_parameters_w_master"]),
                "std_s_parameters_w_master": np.std(arrays["s_parameters_w_master"]),
                #"array_r_parameters_w_master": arrays["r_parameters_w_master"],
                "avg_r_parameters_w_master": np.mean(arrays["r_parameters_w_master"]),
                "std_r_parameters_w_master": np.std(arrays["r_parameters_w_master"]),
                #"array_s_parameters_y_master": arrays["s_parameters_y_master"],
                "avg_s_parameters_y_master": np.mean(arrays["s_parameters_y_master"]),
                "std_s_parameters_y_master": np.std(arrays["s_parameters_y_master"]),
                #"array_r_parameters_y_master": arrays["r_parameters_y_master"],
                "avg_r_parameters_y_master": np.mean(arrays["r_parameters_y_master"]),
                "std_r_parameters_y_master": np.std(arrays["r_parameters_y_master"]),
                #"array_average_memory_mb": arrays["average_memory_mb"],
                "avg_average_memory_mb": np.mean(arrays["average_memory_mb"]),
                "std_average_memory_mb": np.std(arrays["average_memory_mb"]),
                #"array_maximum_memory_mb": arrays["maximum_memory_mb"],
                "avg_maximum_memory_mb": np.mean(arrays["maximum_memory_mb"]),
                "std_maximum_memory_mb": np.std(arrays["maximum_memory_mb"]),
                "avg_c_master_keep_examples_count": avg_c_master_keep,
                "std_c_master_keep_examples_count": np.std(arrays["c_master_keep_examples_count"]),
                "percent_c_master_keep_examples_count": (avg_c_master_keep / N_train) * 100,
                "avg_w_master_keep_examples_count": avg_w_master_keep,
                "std_w_master_keep_examples_count": np.std(arrays["w_master_keep_examples_count"]),
                "percent_w_master_keep_examples_count": (avg_w_master_keep / N_train) * 100,
                "avg_y_master_keep_examples_count": avg_y_master_keep,
                "std_y_master_keep_examples_count": np.std(arrays["y_master_keep_examples_count"]),
                "percent_y_master_keep_examples_count": (avg_y_master_keep / N_train) * 100,
                "count_s_parameters_c_master": len(arrays["s_parameters_c_master"]),
                "count_r_parameters_c_master": len(arrays["r_parameters_c_master"]),
                "count_c_master_keep_examples_count": len(arrays["c_master_keep_examples_count"]),
                "count_s_parameters_w_master": len(arrays["s_parameters_w_master"]),
                "count_r_parameters_w_master": len(arrays["r_parameters_w_master"]),
                "count_w_master_keep_examples_count": len(arrays["w_master_keep_examples_count"]),
                "count_s_parameters_y_master": len(arrays["s_parameters_y_master"]),
                "count_r_parameters_y_master": len(arrays["r_parameters_y_master"]),
                "count_y_master_keep_examples_count": len(arrays["y_master_keep_examples_count"]),
                "count_memory": len(arrays["average_memory_mb"])
            }

            new_json.append(results)
    return new_json

def generate_latex_table(data):
    latex_tables = []

    for entry in data:
        k = entry["k"]
        ProbPoss = entry["ProbPoss"]
        N_train = floor(MNIST_TRAIN_IMAGES_USED / (2 * k))

        


        caption = (
            f"Results for MNIST Addition with $k={k}$ ($N_{{\\text{{train}}}}$ = {N_train}) using $\\Pi$-Nesy-{ProbPoss} over 10 experiments. "
            f"Averages for the rule parameter values ($s_{{\\ast}}$ and $r_{{\\ast}}$) for the $c_{{\\ast}}$-, $w_{{\\ast}}$-, and $y_{{\\ast}}$-rules are computed from data comprising "
            f"$10\\cdot {int(entry['count_s_parameters_c_master'] / 10)}$, $10\\cdot {int(entry['count_s_parameters_w_master'] / 10)}$, and "
            f"$10\\cdot {int(entry['count_s_parameters_y_master'] / 10)}$ observations, respectively. "
            f"Reliable sample counts for the $c_{{\\ast}}$-, $w_{{\\ast}}$-, and $y_{{\\ast}}$-rules are averaged over "
            f"$10\\cdot {int(entry['count_c_master_keep_examples_count'] / 10)}$, $10\\cdot {int(entry['count_w_master_keep_examples_count'] / 10)}$, and "
            f"$10\\cdot {int(entry['count_y_master_keep_examples_count'] / 10)}$ observations, respectively, "
            f"and memory metrics are averaged over 10 experiments."
        )



        
        
        table = f"""
\\begin{{table}}[ht]
\\centering
\\setlength{{\\tabcolsep}}{{1.5pt}}
\\renewcommand{{\\arraystretch}}{{1.5}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\makecell{{Set of rules\\\\(output attribute)}} & \\makecell{{Average value of the\\\\rule parameters $s_{{\\ast}}$}} & \\makecell{{Average value of the\\\\rule parameters $r_{{\\ast}}$}} & \\makecell{{Average number of\\\\training data samples\\\\considered reliable}} & \\makecell{{Percent of\\\\training data samples\\\\considered reliable}} \\\\
\\midrule
$c_{{\\ast}}$ & {entry["avg_s_parameters_c_master"]:.2e} $\\pm$ {entry["std_s_parameters_c_master"]:.2e} & {entry["avg_r_parameters_c_master"]:.2e} $\\pm$ {entry["std_r_parameters_c_master"]:.2e} & {entry["avg_c_master_keep_examples_count"]:.2f} $\\pm$ {entry["std_c_master_keep_examples_count"]:.2f} & {entry["percent_c_master_keep_examples_count"]:.3f} $\\pm$ {entry["std_c_master_keep_examples_count"] / N_train * 100:.3f}\\%\\\\
$w_{{\\ast}}$ & {entry["avg_s_parameters_w_master"]:.2e} $\\pm$ {entry["std_s_parameters_w_master"]:.2e} & {entry["avg_r_parameters_w_master"]:.2e} $\\pm$ {entry["std_r_parameters_w_master"]:.2e} & {entry["avg_w_master_keep_examples_count"]:.2f} $\\pm$ {entry["std_w_master_keep_examples_count"]:.2f} & {entry["percent_w_master_keep_examples_count"]:.3f} $\\pm$ {entry["std_w_master_keep_examples_count"] / N_train * 100:.3f}\\%\\\\
$y_{{\\ast}}$ & {entry["avg_s_parameters_y_master"]:.2e} $\\pm$ {entry["std_s_parameters_y_master"]:.2e} & {entry["avg_r_parameters_y_master"]:.2e} $\\pm$ {entry["std_r_parameters_y_master"]:.2e} & {entry["avg_y_master_keep_examples_count"]:.2f} $\\pm$ {entry["std_y_master_keep_examples_count"]:.2f} & {entry["percent_y_master_keep_examples_count"]:.3f} $\\pm$ {entry["std_y_master_keep_examples_count"] / N_train * 100:.3f}\\%\\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\text{{Average memory (RAM) of an experiment:}} {entry["avg_average_memory_mb"]:.2f} $\\pm$ {entry["std_average_memory_mb"]:.2f} MB}}\\\\
\\multicolumn{{5}}{{l}}{{\\text{{Maximum memory (RAM) of an experiment:}} {entry["avg_maximum_memory_mb"]:.2f} $\\pm$ {entry["std_maximum_memory_mb"]:.2f} MB}}\\\\
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:mnist-add-{k}-pi-nesy-{ProbPoss}}}
\\end{{table}}
"""
        print(table)
        latex_tables.append(table)

    return latex_tables


def main(filepath):
    """
    1) Split the log into segments by "Attempt..." lines.
    2) For each segment, parse out the needed lines.
    3) Build a JSON structure and print it.
    """
    segments = parse_log_into_segments(filepath)

    results = []

    for idx, seg in enumerate(segments, start=1):
        info = analyze_segment(seg)

        # 1) Parse the attempt line -> k, ID, ProbPoss
        k_val, id_val, prob_val = parse_attempt_line(info["attempt_line"])

        # 2) Gather the s/r parameters
        rule_params_dict = parse_s_r_parameters(info["parameters_lines"])

        # 3) parse the problem line (with learning_time) => keeped_examples_count
        prob_data = parse_problem_line(info["problem_line"])  

        # 4) parse the pi-nesy line
        pi_data = parse_pi_nesy_line(info["pi_nesy_line"])    

        # Build a final JSON-like dict for this segment
        out_dict = {
            "k": k_val,
            "ID": id_val,
            "ProbPoss": prob_val,
            "last_validation_line": info["last_validation_line"],
            "parameters": rule_params_dict,
            "problem": prob_data,   # e.g. {"keeped_examples_count": {...}}
            "pi_nesy": pi_data
        }
        results.append(out_dict)
    
    
    results = reArrange(results)
    results = ProperResults(results)
    
    return results


if __name__ == "__main__":
    log_file = "out-mnist-add.txt"
    results = main(log_file)

    
    out_dir = load_config.config["aggregated_results_directory"]
    csv_path = os.path.join(out_dir, "mnist_add_possibilistic_learning_stats_summary.csv")
    tex_path = os.path.join(out_dir, "mnist_add_possibilistic_learning_stats__summary.tex")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    latex_tables = generate_latex_table(results)
    with open(tex_path, "w", encoding="utf-8") as f:
        for table in latex_tables:
            f.write(table)

