import os
import sys
import re
import torch
import random
import numpy as np
import logging

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config


def setup_experiment():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[ logging.StreamHandler()  ])


    num_cpu_cores = os.cpu_count()  
    torch.set_num_threads(num_cpu_cores)

    match = re.findall(r'\d+', load_config.config['problem_studied'])
    nb_problem_studied = int("".join(match)) if match else 0
    
    
    seed = 1992 + nb_problem_studied
    

    # Set seeds 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    logging.info(f"Experiment setup complete. Seed: {seed}, CPU cores: {num_cpu_cores}")

    

