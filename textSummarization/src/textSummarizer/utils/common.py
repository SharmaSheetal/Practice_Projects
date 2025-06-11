import os
from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any
from pathlib import Path
from textSummarizer.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
        
    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
        
    Raises:
        ValueError: If the file is empty.
        Exception: If any other error occurs while reading the file.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
# @ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, prints the status of directory creation.
        
    Returns:
        None
    """
    for path in path_to_directories:
        os.makedirs(Path(path), exist_ok=True)
        if verbose:
            print(f"Created directory: {path}") if not os.path.exists(path) else print(f"Directory already exists: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    returns the size of a file in kilobytes (KB).
    Args:
        path (Path): Path to the file.
    Returns:
        str: Size of the file in KB.
    """
    if not path.exists():
        raise Exception(f"File {path} does not exist")
    
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
