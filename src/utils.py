"""
Project-wide utility functions
"""

from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    """Return absolute path to project root"""
    return Path(__file__).parent.parent

def update_progress(tqdm_instance, mode, section, content):  
    """Update parameters of a tqdm progress bar"""          
    if section == 'desc':
        if mode == 'train':
            content_str = f"Batch [{content[0]+1}/{len(content[1])}]\t"
        if mode == 'val':
            pass
        if mode == 'predict':
            pass
        tqdm_instance.set_description_str(desc=content_str)
    if section == 'postfix':
        if mode == 'train':
            content_str = (
                    f"Batch Loss: {content[0].item():.4g} ({np.mean(content[1]):.4g})\t"
                    f"Loss: {content[2]:.4g} ({content[3]:.4g})\t"
                    f"Val Loss: {content[4]:.4g} ({content[5]:.4g})"
                    )
        if mode == 'val':
            content_str = f"Loss: {content[0].item():.4g} ({content[1]})"
        if mode == 'predict':
            pass
            
        tqdm_instance.set_postfix_str(content_str)
        
def print_to_console(verbose, mode, model_name):
    if verbose:
        if mode == 'train':
            print(f'Training --------- Model: {model_name}')
        if mode == 'val':
            print(f'Validating --------- Model: {model_name}')
        if mode == 'predict':
            print(f'Predicting --------- Model: {model_name}')