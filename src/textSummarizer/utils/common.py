import os
import yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from textSummarizer.logging.logger import logger


def read_yaml(path: Path) -> ConfigBox:
    """Read YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        ConfigBox with configuration
        
    Raises:
        ValueError: If YAML is empty
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(path, encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"Loaded config from {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Invalid YAML structure in {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")


def create_directories(paths: list[Path], verbose: bool = True) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
        verbose: Log creation messages
    """
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    """Get file size in human-readable format.
    
    Args:
        path: File path
        
    Returns:
        Size string (e.g., "1.5 MB")
    """
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

# ============================================================================
# src/textSummarizer/utils/common.py
# ============================================================================
# import os
# import json
# import joblib
# from box.exceptions import BoxValueError
# import yaml
# from textSummarizer.logging.logger import logger
# from ensure import ensure_annotations
# from box import ConfigBox
# from pathlib import Path
# from typing import Any, List, Union


# @ensure_annotations
# def read_yaml(path_to_yaml: Path) -> ConfigBox:
#     """Reads yaml file and returns ConfigBox
    
#     Args:
#         path_to_yaml (Path): path to yaml file
        
#     Raises:
#         ValueError: if yaml file is empty
#         FileNotFoundError: if yaml file doesn't exist
        
#     Returns:
#         ConfigBox: ConfigBox type
#     """
#     try:
#         with open(path_to_yaml) as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             logger.info(f"yaml file: {path_to_yaml} loaded successfully")
#             return ConfigBox(content)
#     except BoxValueError:
#         raise ValueError("yaml file is empty")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"yaml file not found at: {path_to_yaml}")
#     except Exception as e:
#         raise e


# @ensure_annotations
# def create_directories(path_to_directories: list, verbose: bool = True):
#     """Create list of directories
    
#     Args:
#         path_to_directories (list): list of paths of directories
#         verbose (bool, optional): log directory creation. Defaults to True.
#     """
#     for path in path_to_directories:
#         os.makedirs(path, exist_ok=True)
#         if verbose:
#             logger.info(f"created directory at: {path}")


# @ensure_annotations
# def get_size(path: Path) -> str:
#     """Get size in KB
    
#     Args:
#         path (Path): path of the file
        
#     Returns:
#         str: size in KB
#     """
#     size_in_kb = round(os.path.getsize(path) / 1024)
#     return f"~ {size_in_kb} KB"


# @ensure_annotations
# def save_json(path: Path, data: dict):
#     """Save json data
    
#     Args:
#         path (Path): path to json file
#         data (dict): data to be saved in json file
#     """
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)
    
#     logger.info(f"json file saved at: {path}")


# @ensure_annotations
# def load_json(path: Path) -> ConfigBox:
#     """Load json files data
    
#     Args:
#         path (Path): path to json file
        
#     Returns:
#         ConfigBox: data as class attributes instead of dict
#     """
#     with open(path) as f:
#         content = json.load(f)
    
#     logger.info(f"json file loaded successfully from: {path}")
#     return ConfigBox(content)


# @ensure_annotations
# def save_bin(data: Any, path: Path):
#     """Save binary file
    
#     Args:
#         data (Any): data to be saved as binary
#         path (Path): path to binary file
#     """
#     joblib.dump(value=data, filename=path)
#     logger.info(f"binary file saved at: {path}")


# @ensure_annotations
# def load_bin(path: Path) -> Any:
#     """Load binary data
    
#     Args:
#         path (Path): path to binary file
        
#     Returns:
#         Any: object stored in the file
#     """
#     data = joblib.load(path)
#     logger.info(f"binary file loaded from: {path}")
#     return data