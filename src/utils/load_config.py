import json
import os

current_dir = os.path.dirname(__file__)
config_file_path = os.path.join(current_dir, '../../config.json')

def load_config(file_path):
    """Load the configuration from the JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(file_path, config):
    """Save the configuration back to the JSON file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def update_config(key, value):
    """Update a specific key in the config file and reload it."""
    global config
    config[key] = value
    save_config(config_file_path, config)
    config = load_config(config_file_path)

config = load_config(config_file_path)
