import re
import json
import ast
from collections import defaultdict
from statistics import mean, stdev
import os
import sys
import pandas
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

def parse_balanced(text, start_index, open_char, close_char):
    """
    Extracts a balanced substring starting from an opening character.
    """
    depth = 0
    i = start_index
    while i < len(text):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return text[start_index:i + 1], i + 1
        i += 1
    raise ValueError("No matching closing brace/bracket found.")


def parse_dict_or_list(possible_str):
    """
    Attempt to parse as dictionary or list, or return the raw string.
    """
    try:
        return ast.literal_eval(possible_str)
    except Exception:
        return possible_str.strip()


def parse_last_log(last_log):
    """
    Parse the last log to extract the three dictionaries.
    """
    result = {
        "Parameters": None,
        "Parameters2": None,
        "keep_count_examples": None,
    }

    # 1. Extract "Parameters"
    marker_params = "From Sets F:"
    idx_marker = last_log.find(marker_params)
    if idx_marker >= 0:
        idx_start = idx_marker + len(marker_params)
        idx_open_curly = last_log.find("{", idx_start)
        if idx_open_curly >= 0:
            param_str, end_pos = parse_balanced(last_log, idx_open_curly, "{", "}")
            result["Parameters"] = parse_dict_or_list(param_str)

    # 2. Extract "Parameters2"
    marker_p2 = "From the output attribute for the decision:"
    idx_marker2 = last_log.find(marker_p2)
    if idx_marker2 >= 0:
        idx_start2 = idx_marker2 + len(marker_p2)
        idx_open_brack = last_log.find("[", idx_start2)
        if idx_open_brack >= 0:
            p2_str, end_pos = parse_balanced(last_log, idx_open_brack, "[", "]")
            result["Parameters2"] = parse_dict_or_list(p2_str)

    # 3. Extract "keep_count_examples"
    marker_keep = "Keeped examples for learning | stats"
    idx_marker3 = last_log.find(marker_keep)
    if idx_marker3 >= 0:
        idx_start3 = idx_marker3 + len(marker_keep)
        idx_open_curly2 = last_log.find("{", idx_start3)
        if idx_open_curly2 >= 0:
            keep_str, end_pos = parse_balanced(last_log, idx_open_curly2, "{", "}")
            result["keep_count_examples"] = parse_dict_or_list(keep_str)

    return result


def get_experiment_segments(log_content, start_pattern, end_pattern, delimiter):
    """
    Parses the log content to extract experiment segments, ensuring each log starts and ends with the delimiter.
    """
    segments = []
    current_segment = None
    lines = log_content.splitlines()
    delimiter_pattern = re.compile(re.escape(delimiter))

    for line in lines:
        # Detect start of an experiment
        start_match = start_pattern.search(line)
        if start_match:
            if current_segment:  # Save the previous segment
                segments.append(current_segment)
            current_segment = {
                "start": line,
                "logs": [],
            }
            continue
        
        # If the line matches the delimiter and we're in a segment, close the current log
        if delimiter_pattern.fullmatch(line):
            if current_segment is not None and current_segment.get("current_log"):
                current_segment["logs"].append("\n".join(current_segment["current_log"]))
                current_segment["current_log"] = []
            continue

        # If we're in a segment, collect lines for the current log
        if current_segment is not None:
            current_segment.setdefault("current_log", []).append(line)
        
        # Detect end of an experiment
        end_match = end_pattern.search(line)
        if end_match and current_segment:
            current_segment["end"] = line
            segments.append(current_segment)
            current_segment = None

    # If there's an ongoing experiment, finalize it
    if current_segment:
        segments.append(current_segment)

    # Post-process segments: Remove empty logs
    for segment in segments:
        segment["logs"] = [log.strip() for log in segment["logs"] if log.strip()]
        segment["last_log"] = segment["logs"][-1] if segment["logs"] else None

    return segments


def extract_experiment_data(experiment_segments):
    """
    Extract all required data and create the final JSON list.
    """
    experiments = []
    for segment in experiment_segments:
        # Skip if there's no last log
        if not segment.get("last_log"):
            continue

        # Extract experiment metadata
        experiment_data = {}
        start_match = experiment_start_pattern.search(segment["start"])
        end_match = experiment_end_pattern.search(segment.get("end", ""))

        if start_match:
            experiment_data.update({
                "ProbPoss": int(start_match.group(4)),
                "Dimension": int(start_match.group(2)),
                "SplitDir": start_match.group(3),
            })

        if end_match:
            experiment_data.update({
                "AVG_MEM": float(end_match.group(2)),
                "MAX_MEM": float(end_match.group(3)),
            })

        # Parse the last log for the three dictionaries
        try:
            last_log_data = parse_last_log(segment["last_log"])
            experiment_data.update(last_log_data)
        except Exception as e:
            print(f"Error parsing last log for experiment: {e}")
            continue

        experiments.append(experiment_data)

    return experiments

def convert_keys_to_strings(obj):
    """
    Recursively convert dictionary keys to strings.
    """
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_strings(item) for item in obj]
    else:
        return obj

def process_experiments(experiments):
    # Initialize structures for aggregating data
    aggregated_data = defaultdict(lambda: {
        "AVG_MEM": [],
        "MAX_MEM": [],
        "s_parameters_all": [],  # Store all s_parameter values across tuples
        "r_parameters_all": [],  # Store all r_parameter values across tuples
        "val_1": [],
        "val_2": [],
        "keeped_examples_all": [],  # Store all keeped examples across tuples
        "keeped_examples_final_output_attr": []
    })

    # Training dataset size
    TRAINING_DATASET_SIZE = 2*load_config.config["mnist_train_size_for_mnistsudoku"]

    # Aggregate data by (ProbPoss, Dimension) pairs
    for experiment in experiments:
        key = (experiment["ProbPoss"], experiment["Dimension"])
        aggregated_data[key]["AVG_MEM"].append(experiment["AVG_MEM"])
        aggregated_data[key]["MAX_MEM"].append(experiment["MAX_MEM"])

        # Aggregate s_parameters and r_parameters
        for tuple_key, values in experiment["Parameters"].items():
            aggregated_data[key]["s_parameters_all"].extend(values[0])  # s_parameters
            aggregated_data[key]["r_parameters_all"].extend(values[1])  # r_parameters

        # Aggregate val_1 and val_2 from Parameters2
        if experiment["Parameters2"]:
            aggregated_data[key]["val_1"].extend(experiment["Parameters2"][0])
            aggregated_data[key]["val_2"].extend(experiment["Parameters2"][1])

        # Aggregate keeped examples
        for tuple_key, keep_value in experiment["keep_count_examples"].items():
            if tuple_key != "final_output_attr":
                aggregated_data[key]["keeped_examples_all"].append(keep_value)
            else:
                aggregated_data[key]["keeped_examples_final_output_attr"].append(keep_value)

    # Prepare the final output
    final_output = []

    for key, data in aggregated_data.items():
        prob_poss, dimension = key

        # Calculate statistics for memory usage
        avg_mem = mean(data["AVG_MEM"])
        max_mem = mean(data["MAX_MEM"])
        avg_mem_std = stdev(data["AVG_MEM"]) if len(data["AVG_MEM"]) > 1 else 0
        max_mem_std = stdev(data["MAX_MEM"]) if len(data["MAX_MEM"]) > 1 else 0

        # Calculate overall statistics for s_parameters and r_parameters across all tuples
        s_parameter_tuple_avg_val = {
            "mean": mean(data["s_parameters_all"]),
            "std": stdev(data["s_parameters_all"]) if len(data["s_parameters_all"]) > 1 else 0
        }
        r_parameter_tuple_avg_val = {
            "mean": mean(data["r_parameters_all"]),
            "std": stdev(data["r_parameters_all"]) if len(data["r_parameters_all"]) > 1 else 0
        }

        # Calculate statistics for val_1 and val_2
        final_output_s_parameter_avg = {
            "mean": mean(data["val_1"]),
            "std": stdev(data["val_1"]) if len(data["val_1"]) > 1 else 0
        }
        final_output_r_parameter_avg = {
            "mean": mean(data["val_2"]),
            "std": stdev(data["val_2"]) if len(data["val_2"]) > 1 else 0
        }

        # Calculate overall statistics for keeped examples across all tuples (for b_{(i,j,i',j')})
        keeped_examples_tuple_avg_val = {
            "mean": mean(data["keeped_examples_all"]),
            "std": stdev(data["keeped_examples_all"]) if len(data["keeped_examples_all"]) > 1 else 0,
            "percent": (mean(data["keeped_examples_all"]) / TRAINING_DATASET_SIZE) * 100,
            "percent_std": (stdev(data["keeped_examples_all"]) / TRAINING_DATASET_SIZE) * 100 if len(data["keeped_examples_all"]) > 1 else 0
        }

        # Calculate statistics for final output attribute (for c)
        keeped_examples_final_output_attr_stats = {
            "mean": mean(data["keeped_examples_final_output_attr"]),
            "std": stdev(data["keeped_examples_final_output_attr"]) if len(data["keeped_examples_final_output_attr"]) > 1 else 0,
            "percent": (mean(data["keeped_examples_final_output_attr"]) / TRAINING_DATASET_SIZE) * 100,
            "percent_std": (stdev(data["keeped_examples_final_output_attr"]) / TRAINING_DATASET_SIZE) * 100 if len(data["keeped_examples_final_output_attr"]) > 1 else 0
        }
        count_b_s = len(data["s_parameters_all"])
        count_b_keep = len(data["keeped_examples_all"])
        count_c_s = len(data["val_1"])  # from Parameters2
        count_c_keep = len(data["keeped_examples_final_output_attr"])

        # Compile results
        final_output.append({
            "ProbPoss": prob_poss,
            "Dimension": dimension,
            "avg_mem": {"mean": avg_mem, "std": avg_mem_std},
            "max_mem": {"mean": max_mem, "std": max_mem_std},
            "s_parameter_tuple_avg_val": s_parameter_tuple_avg_val,
            "r_parameter_tuple_avg_val": r_parameter_tuple_avg_val,
            "final_output_s_parameter_avg": final_output_s_parameter_avg,
            "final_output_r_parameter_avg": final_output_r_parameter_avg,
            "keeped_examples_tuple_avg_val": keeped_examples_tuple_avg_val,
            "keeped_examples_final_output_attr_stats": keeped_examples_final_output_attr_stats,
            "count_b_s": count_b_s,
            "count_b_keep": count_b_keep,
            "count_c_s": count_c_s,
            "count_c_keep": count_c_keep
        })

    return final_output

def generate_latex_table(experiment):
    prob_poss = experiment["ProbPoss"]
    dimension = experiment["Dimension"]
    avg_mem = experiment["avg_mem"]
    max_mem = experiment["max_mem"]
    s_param = experiment["s_parameter_tuple_avg_val"]
    r_param = experiment["r_parameter_tuple_avg_val"]
    keeped_examples = experiment["keeped_examples_tuple_avg_val"]
    final_attr = experiment["keeped_examples_final_output_attr_stats"]
    final_attr_s_param = experiment["final_output_s_parameter_avg"]
    final_attr_r_param = experiment["final_output_r_parameter_avg"]
    
    # Retrieve the new count fields
    count_b_s = experiment["count_b_s"]
    count_b_keep = experiment["count_b_keep"]
    count_c_s = experiment["count_c_s"]
    count_c_keep = experiment["count_c_keep"]

    # Build the table (header and rows remain unchanged)
    table = []
    table.append(r"\begin{table}[ht]")
    table.append(r"\centering")
    table.append(r"\setlength{\tabcolsep}{1.5pt}")
    table.append(r"\renewcommand{\arraystretch}{1.5}")
    table.append(r"\begin{tabular}{lcccc}")
    table.append(r"\toprule")
    table.append(r"\makecell{Set of rules\\(output attribute)} & \makecell{Average value of the\\rule parameters $s$} & \makecell{Average value of the\\rule parameters $r$} & \makecell{Average number of\\training data samples\\considered reliable} & \makecell{Percent of\\training data samples\\considered reliable} \\")
    table.append(r"\midrule")

    # Add the row for tuple stats (for b_{(i,j,i',j')})
    tuple_name = "$b_{(i,j,i',j')}$"
    table.append(
        f"{tuple_name} & {s_param['mean']:.2e} $\\pm$ {s_param['std']:.2e} & {r_param['mean']:.2e} $\\pm$ {r_param['std']:.2e} & "
        f"{keeped_examples['mean']:.1f} $\\pm$ {keeped_examples['std']:.1f} & {keeped_examples['percent']:.2f} $\\pm$ {keeped_examples['percent_std']:.2f}\\% \\\\"
    )

    # Add the row for final output attribute stats (for c)
    table.append(
        f"c & {final_attr_s_param['mean']:.2e} $\\pm$ {final_attr_s_param['std']:.2e} & {final_attr_r_param['mean']:.2e} $\\pm$ {final_attr_r_param['std']:.2e} & "
        f"{final_attr['mean']:.1f} $\\pm$ {final_attr['std']:.1f} & {final_attr['percent']:.2f} $\\pm$ {final_attr['percent_std']:.2f}\\% \\\\"
    )

    table.append(r"\midrule")

    # Add footer for memory stats
    table.append(
        f"\\multicolumn{{5}}{{l}}{{\\text{{Average memory (RAM) of an experiment:}} {avg_mem['mean']:.1f} $\\pm$ {avg_mem['std']:.1f} MB}}\\\\"
    )
    table.append(
        f"\\multicolumn{{5}}{{l}}{{\\text{{Maximum memory (RAM) of an experiment:}} {max_mem['mean']:.1f} $\\pm$ {max_mem['std']:.1f} MB}}\\\\"
    )
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(
        f"\\caption{{Possibilistic learning results on the MNIST Sudoku {dimension}x{dimension} problem using $\\Pi$-NeSy-{prob_poss}. Results are averaged over ten experiments. The respective averages for the rule parameters corresponding to $b_{{(i,j,i',j')}}$ are computed from $10\\cdot {count_b_s // 10}$ observations, while those for the attribute $c$ are computed from $10\\cdot {count_c_s // 10}$ observations. Reliable training sample counts for $b_{{(i,j,i',j')}}$ and $c$ are computed from $10\\cdot {count_b_keep // 10}$ and $10\\cdot {count_c_keep // 10}$ observations, respectively.}}"
    )
    table.append(f"\\label{{tab:mnist-sudoku-{dimension}-pi-nesy-{prob_poss}}}")
    table.append(r"\end{table}")

    return "\n".join(table)




# Compile patterns for identifying experiment start and end
experiment_start_pattern = re.compile(
    r"Attempt (\d+) of 50 for Dimension=(\d+), SplitDir=(.+?), ProbPoss=(\d+)"
)
experiment_end_pattern = re.compile(
    r"Pi-Nesy-\d+ stats for mnist-sudoku-\d+ \| id:(.+?) \| Average memory usage: ([\d\.]+) MB, Maximum memory usage: ([\d\.]+) MB"
)
delimiter = "+++++++++++++++++++++++++++++++++++++++"

log_file_path = 'out-mnist-sudoku.txt'
with open(log_file_path, 'r') as file:
    log_content = file.read()

# Extract experiment segments
experiment_segments = get_experiment_segments(
    log_content, experiment_start_pattern, experiment_end_pattern, delimiter
)

# Generate the final experiment data
experiment_data = extract_experiment_data(experiment_segments)

# Convert all dictionary keys to strings
experiment_data = [convert_keys_to_strings(experiment) for experiment in experiment_data]





final_output = process_experiments(experiment_data)


# Generate LaTeX tables for all experiments
latex_tables = []
for experiment in final_output:
    table = generate_latex_table(experiment)
    latex_tables.append(table)
    print(table)

csv_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_possibilistic_learning_stats_summary.csv")
pandas.DataFrame(final_output).to_csv(csv_out, index=False)

tex_out = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_possibilistic_learning_stats_summary.tex")
with open(tex_out, 'w', encoding='utf-8') as out_file:
    out_file.write("\n\n".join(latex_tables))

print(f"CSV results saved to {csv_out}")
print(f"LaTeX tables saved to {tex_out}")