import os
import sys
import csv
import numpy as np
import glob
import re
import pandas as pd

from pathlib import Path

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config


def aggregate_results(base_dir, output_dir):
    
    results_path = Path(base_dir)
    aggregated_results_path = Path(output_dir) / "vspc"
    aggregated_results_path.mkdir(parents=True, exist_ok=True)

    for method_dir in results_path.iterdir():
        if method_dir.is_dir():  
            method_id = method_dir.name.split('::')[-1]
            for experiment_dir in method_dir.iterdir():
                if experiment_dir.is_dir():  
                    exp_id = experiment_dir.name.split('::')[-1]
                    accuracies_test = []
                    accuracies_NN_test = []
                    file_count = 0

                    
                    for csv_file in experiment_dir.glob('*.csv'):
                        if csv_file.is_file(): 
                            with open(csv_file, mode='r') as file:
                                reader = csv.DictReader(file)
                                for row in reader:
                                    accuracies_test.append(float(row['accuracy_test']))
                                    accuracies_NN_test.append(float(row['accuracy_NN_test']))
                            file_count += 1

                    if accuracies_test and accuracies_NN_test:
                        avg_accuracy_test = np.mean(accuracies_test)
                        std_accuracy_test = np.std(accuracies_test)
                        avg_accuracy_NN_test = np.mean(accuracies_NN_test)
                        std_accuracy_NN_test = np.std(accuracies_NN_test)

                        results = [
                            method_id, exp_id, file_count,
                            round(avg_accuracy_test, 1), round(std_accuracy_test, 2),
                            round(avg_accuracy_NN_test, 2), round(std_accuracy_NN_test, 2)
                        ]

                        
                        results_filename = f"aggregated_results_{method_id}_{exp_id}.csv"
                        results_filepath = aggregated_results_path / results_filename
                        with open(results_filepath, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            headers = ['Method ID', 'Experiment ID', 'Number of Experiments', 'Avg Accuracy Test', 'Std Dev Accuracy Test', 'Avg Accuracy NN Test', 'Std Dev Accuracy NN Test']
                            writer.writerow(headers)
                            writer.writerow(results)

                        print(f"Results written to {results_filepath}")



def parse_mnist_sudoku_agg_results(folder_path):
    res = {}
    for csv_file in glob.glob(os.path.join(folder_path, "*.csv")):
        with open(csv_file, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                m = row.get("Method ID","").strip()
                e = row.get("Experiment ID","").strip()
                if m and e:
                    a = float(row.get("Avg Accuracy Test","0"))
                    s = float(row.get("Std Dev Accuracy Test","0"))
                    if m not in res:
                        res[m] = {}
                    res[m][e] = (a, s)
    return res

def generate_latex_table_mnist_sudoku(res):
    all_exps = set()
    for m in res:
        all_exps.update(res[m].keys())
    def sudoku_key(e):
        found = re.findall(r'(\d+)x(\d+)', e)
        return int(found[0][0]) if found else 999
    exps_sorted = sorted(all_exps, key=sudoku_key)

    t = []
    t.append(r"\begin{table}[ht]")
    t.append(r"\centering")
    t.append(r"\renewcommand{\arraystretch}{1.5}")
    t.append(r"\begin{tabular}{l" + "c"*len(exps_sorted) + "}")
    t.append(r"\hline")
    header = ["{Approach}"] + [f"{{{exp.split('-')[-1]} Sudoku}}" for exp in exps_sorted]
    t.append(" & ".join(header) + r"\\")
    t.append(r"\hline")
    for approach in ["pi-nesy-1", "pi-nesy-2"]:
        if approach in res:
            label = r"$\Pi$-NeSy-1" if approach == "pi-nesy-1" else r"$\Pi$-NeSy-2"
            row = [label]
            for e in exps_sorted:
                avg,std = res[approach].get(e,(0,0))
                row.append(f"{avg:.1f} $\\pm$ {std:.2f}")
            t.append(" & ".join(row) + r"\\")
    t.append(r"\hline")
    t.append(r"\end{tabular}")
    t.append(r"\caption{Accuracy and standard deviation on the test dataset for the MNIST-Sudoku problem over 10 runs.}")
    t.append(r"\label{tab:visual_Sudoku_classification}")
    t.append(r"\end{table}")
    return "\n".join(t)


def generate_agg_result_mnist_sudoku_latex_tables():
    sudoku_path = load_config.config["aggregated_results_directory"] + "/vspc"
    sudoku_res = parse_mnist_sudoku_agg_results(sudoku_path)
    latex_sudoku = generate_latex_table_mnist_sudoku(sudoku_res)
    print("\nMNIST-Sudoku table:\n")
    print(latex_sudoku)

    out_file = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_agg_results_in_latex_table.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_sudoku)
    
    rows = []
    for method_id in sudoku_res:
        for exp_id, (avg, std) in sudoku_res[method_id].items():
            rows.append({
                "Method ID": method_id,
                "Experiment ID": exp_id,
                "Average Accuracy Test": avg,
                "Std Dev Accuracy Test": std
            })
            
    csv_file = os.path.join(load_config.config["aggregated_results_directory"], "mnist_sudoku_agg_results_summary.csv")
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    df = pd.DataFrame(rows).sort_values(by=["Method ID", "Experiment ID"], ascending=[True, True])
    df.to_csv(csv_file, index=False)
    print(f"CSV summary saved to {csv_file}")

base_directory = load_config.config["results_dir"] + "/vspc"
output_directory = load_config.config["aggregated_results_directory"]
aggregate_results(base_directory, output_directory)
generate_agg_result_mnist_sudoku_latex_tables()