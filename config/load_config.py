import yaml
import os


def load_config(path):
    """
    Load YAML configuration file.
    Args:
        path (str): Path to YAML config file
    Returns:
        dict: Loaded configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config