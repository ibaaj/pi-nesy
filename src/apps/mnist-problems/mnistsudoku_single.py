from setup_experiment import setup_experiment

import argparse
import sys
import time
import threading
import resource
import gc
import os
import re

from mnistsudoku import experiment

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config


"""
This script is a helper for the bash script that runs experiments on mnist sudoku datasets.
"""
# Shared list to store memory readings
memory_readings = []

def monitor_memory(interval=0.5):
    """
    Monitor memory usage every `interval` seconds.
    """
    global stop_monitoring
    while not stop_monitoring:
        memory_in_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_readings.append(memory_in_kb / (1024 * 1024) )  # Convert to MB for OSX
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mnist sudoku experiments with given parameters.')
    parser.add_argument('dimension_number', type=int, help='The dimension of the puzzle (e.g., 4 for 4x4)')
    parser.add_argument('split_dir', type=str, help='Directory of the split')
    parser.add_argument('probposs_transformation_method', type=int, choices=[1, 2], help='ProbPoss transformation method (1 or 2)')
    parser.add_argument('status_file', type=str, help='Path to the status file to update upon successful execution')
    
    args = parser.parse_args()

    number_split = int(re.search(r'split[^\d]*(\d+)', args.split_dir).group(1)) if re.search(r'split[^\d]*(\d+)', args.split_dir) else None

    load_config.update_config("problem_studied", "sudoku_" + str(args.dimension_number) + "x" + str(args.dimension_number) + "_" + str(number_split) + "_" + str(args.probposs_transformation_method))

    setup_experiment()
    

    # Start monitoring in a separate thread
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    try:
        experiment(args.dimension_number, args.split_dir, args.probposs_transformation_method)

        # Update the status file upon successful execution
        with open(args.status_file, 'w') as file:
            file.write("experiment was ok")
    finally:
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
    
    # Analyze memory readings
    avg_mem = sum(memory_readings) / len(memory_readings) if memory_readings else 0
    max_mem = max(memory_readings) if memory_readings else 0
    print(f"Pi-Nesy-{args.probposs_transformation_method} stats for mnist-sudoku-{args.dimension_number} | id:{args.split_dir} | Average memory usage: {avg_mem:.2f} MB, Maximum memory usage: {max_mem:.2f} MB")
    print("Closing. A segfault may occur afterwards, due to a dealloc problem with pybind11, but everything worked fine (as long as no error occurred earlier).")

    del experiment
    gc.collect()

    # A segfault may happen because of dealloc of pybind11
    sys.exit(0)