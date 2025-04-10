import random
import sys
import numpy as np
import csv
import datetime
import logging
import time
import os
import string
import argparse
import pickle 
import gc
import json
import re
import time
import torch

current_dir = os.path.dirname(__file__)

parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config

from nn import run_mnist_experiment

num_cpu_cores = os.cpu_count()  
torch.set_num_threads(num_cpu_cores)

seed = 42

# Set seeds 
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior in CuDNN
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


load_config.update_config("same_NN_as_DeepSoftLog", False)
load_config.update_config("problem_studied", "addition")

acc_test = []
for i in range(10):
    logging.info(f"Iteration {i+1}/10: Training a new NN model...")
    file_name_train_c, file_name_valid_c, file_name_test_c, accuracy_NN_test_c, accuracy_NN_val_c, file_checkpoint_model = run_mnist_experiment(60000, 10000, 10000)
    logging.info(f"Training completed. Model achieved validation accuracy: {accuracy_NN_val_c:.4f}, test accuracy: {accuracy_NN_test_c:.4f}")
    acc_test.append(accuracy_NN_test_c)


m = round(np.mean(acc_test),2)
s = round(np.std(acc_test),2)

logging.info(f"Average test accuracy: {m} pm  {s}")
sys.exit(0)