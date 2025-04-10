from setup_experiment import setup_experiment

import argparse
import sys
import time
import threading
import resource
import gc
import os 

from mnistadd import experiment


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(os.path.abspath(parent_dir))

import load_config

"""
This script is a helper for the bash script mnistadd.sh
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
    parser = argparse.ArgumentParser(description='Run mnist-addition experiments with given parameters.')
    parser.add_argument('k', type=int, help='Digits per number')
    parser.add_argument('id', type=int, help='id/number of cross-validation')
    parser.add_argument('probposs_transformation_method', type=int, choices=[1, 2], help='ProbPoss transformation method (1 or 2)')
    parser.add_argument('status_file', type=str, help='Path to the status file to update upon successful execution')
    
    args = parser.parse_args()

    load_config.update_config("problem_studied", "addition_" + str(args.k)  + "_" + str(args.probposs_transformation_method) + "_"  + str(args.id))

    setup_experiment()

    # Start monitoring in a separate thread
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    try:
        experiment(args.k, args.id, args.probposs_transformation_method)

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
    print(f"Pi-Nesy-{args.probposs_transformation_method} stats for mnist-add-{args.k} | id:{args.id} | Average memory usage: {avg_mem:.2f} MB, Maximum memory usage: {max_mem:.2f} MB")
    print("Closing. A segfault may occur afterwards, due to a dealloc problem with pybind11, but everything worked fine (as long as no error occurred earlier).")

    del experiment
    gc.collect()
    # A segfault may happen cause of dealloc of pybind11
    sys.exit(0)