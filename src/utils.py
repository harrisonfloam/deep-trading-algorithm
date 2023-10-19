"""
Project-wide utility functions
"""

from pathlib import Path
import numpy as np
import torch

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
                    f"Loss: {content[2].mean_loss_epoch:.4g} ({content[2].mean_loss_training:.4g})\t"
                    f"Val Loss: {content[2].val_loss:.4g} ({content[2].mean_loss_val:.4g})"
                    )
        if mode == 'val':
            content_str = f"Loss: {content[0].item():.4g} ({content[1]})"
        if mode == 'predict':
            pass
            
        tqdm_instance.set_postfix_str(content_str)
        
def print_to_console(mode, model_name, verbose, show_progress=True):
    if verbose:
        if mode == 'train':
            print(f'Training --------- Model: {model_name}')
        if mode == 'val':
            if show_progress:
                print(f'Validating --------- Model: {model_name}')
        if mode == 'predict':
            print(f'Predicting --------- Model: {model_name}')
            
def google_colab_handler(use_TPU, verbose=False):
    try:
        from google.colab import drive
        running_in_colab = True
    except:
        running_in_colab = False

    # runningInColab = 'google.colab' in str(get_ipython())

    if running_in_colab:
        if verbose:
            print('Running in Google Colab')
        path = '/content/drive/MyDrive/Google Colab/time-series-project/'
        if use_TPU:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as met
            import torch_xla.distributed.parallel_loader as pl

            device = xm.xla_device()
            if xm.xla_device().type == 'xla':
                if verbose:
                    print('TPU is available')
            # Initialize the TPU cores
            xm.rendezvous('init')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    else:
        path = 'working/data/'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
        if device == 'cuda':
            torch.cuda.empty_cache()
    if verbose:
        print(f"Using Device: {device}")

    return running_in_colab, use_TPU, path, device