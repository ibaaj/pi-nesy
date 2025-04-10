import os
import sys
import csv
import numpy as np
import glob
import re
import pandas as pd

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config


def aggregate_results(experiment, method_number, k, number_train_mnist_images, base_dir='results', output_dir='aggregated_results'):
    
    directory = f"{base_dir}/{experiment}/method::pi-nesy-{method_number}/{k}/{number_train_mnist_images}"
    results_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
    accuracy_test_values = []
    ambiguous_rate_test_values = []
    accuracy_nn_test_values = []

    for filename in results_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Skip headers
            values = next(reader)

            accuracy_test_values.append(float(values[1]))
            ambiguous_rate_test_values.append(float(values[2]))
            accuracy_nn_test_values.append(float(values[4]))

   
    accuracy_test_avg = round(np.mean(accuracy_test_values), 1)
    accuracy_test_std = round(np.std(accuracy_test_values), 2)
    ambiguous_rate_test_avg = round(np.mean(ambiguous_rate_test_values), 2)
    ambiguous_rate_test_std = round(np.std(ambiguous_rate_test_values), 2)
    accuracy_nn_test_avg = round(np.mean(accuracy_nn_test_values), 2)
    accuracy_nn_test_std = round(np.std(accuracy_nn_test_values), 2)

   
    aggregated_dir = f"./{output_dir}/{experiment}"
    os.makedirs(aggregated_dir, exist_ok=True)

    
    output_file = f"{aggregated_dir}/aggregated_result_mnistadd_{k}_with_pi_nesy_{method_number}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Configuration', 'Metric', 'Average', 'Standard Deviation'])
        config = f"pi-nesy-{method_number}, k={k}, train={number_train_mnist_images}"
        writer.writerow([config, 'Accuracy_Test', accuracy_test_avg, accuracy_test_std])
        writer.writerow([config, 'Ambiguous_Rate_Test', ambiguous_rate_test_avg, ambiguous_rate_test_std])
        writer.writerow([config, 'Accuracy_NN_Test', accuracy_nn_test_avg, accuracy_nn_test_std])

    print(f"Aggregated results saved to {output_file}")

def run_aggregation_for_all_configs(experiment, base_dir='results', output_dir='aggregated_results'):
    experiment_dir = f"{base_dir}/{experiment}"
    for method in os.listdir(experiment_dir):
        method_path = os.path.join(experiment_dir, method)
        if os.path.isdir(method_path) and method.startswith("method::pi-nesy-"):
            method_number = method.split('-')[-1]
            for k in os.listdir(method_path):
                k_path = os.path.join(method_path, k)
                if os.path.isdir(k_path):
                    for number_train_mnist_images in os.listdir(k_path):
                        train_images_path = os.path.join(k_path, number_train_mnist_images)
                        if os.path.isdir(train_images_path):
                            aggregate_results(experiment, method_number, k, number_train_mnist_images, base_dir, output_dir)





def parse_mnist_addition_agg_results(folder_path):
    res = {'pi-nesy-1': {}, 'pi-nesy-2': {}}
    for csv_file in glob.glob(os.path.join(folder_path, "*.csv")):
        with open(csv_file, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("Metric","") == "Accuracy_Test":
                    cfg = row["Configuration"].replace('"','')
                    parts = [p.strip() for p in cfg.split(',')]
                    approach = None
                    kval = None
                    for p in parts:
                        if p.startswith("pi-nesy-"):
                            approach = p
                        if p.startswith("k="):
                            kval = int(p.split('=')[1])
                    if approach and kval:
                        a = float(row.get("Average","0"))
                        s = float(row.get("Standard Deviation","0"))
                        res[approach][kval] = (a, s)
    return res

def generate_latex_table_mnist_add(res):
    all_ks = set(res['pi-nesy-1'].keys()).union(res['pi-nesy-2'].keys())
    ks_sorted = sorted(all_ks)
    t = []
    t.append(r"\begin{table}[ht]")
    t.append(r"\centering")
    t.append(r"\setlength{\tabcolsep}{1.5pt}")
    t.append(r"\renewcommand{\arraystretch}{1.5}")
    t.append(r"\begin{tabular}{l" + "c"*len(ks_sorted) + "}")
    t.append(r"\hline")
    t.append("Digits per number $k$ & " + " & ".join(str(k) for k in ks_sorted) + r"\\")
    t.append(r"\hline")
    for approach in ["pi-nesy-1", "pi-nesy-2"]:
        label = r"$\Pi$-NeSy-1" if approach == "pi-nesy-1" else r"$\Pi$-NeSy-2"
        row = [label]
        for k in ks_sorted:
            avg,std = res[approach].get(k,(0,0))
            row.append(f"{avg:.1f} $\\pm$ {std:.2f}")
        t.append(" & ".join(row) + r"\\[2mm]")
    t.append(r"\hline")
    t.append(r"\end{tabular}")
    t.append(r"\caption{Average accuracy and standard deviation on the test dataset for the MNIST-Addition-$k$ problem over 10 runs.}")
    t.append(r"\label{tab:mnist_add_results}")
    t.append(r"\end{table}")
    return "\n".join(t)



def generate_agg_result_mnist_add_latex_tables():
    agg_dir = load_config.config["aggregated_results_directory"]
    add_path = os.path.join(agg_dir, "mnist-addition")
    add_res = parse_mnist_addition_agg_results(add_path)
    latex_add = generate_latex_table_mnist_add(add_res)
    print("MNIST-Addition table:\n")
    print(latex_add)
    output_file = os.path.join(agg_dir, "mnist_addition_agg_results_in_latex_table.tex")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_add)
    
    rows = []
    for approach in add_res:
        for k, (avg, std) in add_res[approach].items():
            rows.append({"Approach": approach, "k": k, "Average": avg, "StdDev": std})
            
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Approach", "k"], ascending=[True, True])
    
    csv_file = os.path.join(agg_dir, "mnist_addition_agg_results_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to {csv_file}")


run_aggregation_for_all_configs('mnist-addition', load_config.config["results_dir"], load_config.config["aggregated_results_directory"])
generate_agg_result_mnist_add_latex_tables()