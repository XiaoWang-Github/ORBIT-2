"""Small shared utilities for the forecasting scripts."""

import os
import random
import torch
import numpy as np


def seed_everything(seed):
    """Set random seeds for reproducibility across all libraries.

    This function sets random seeds for Python's random module, NumPy,
    and PyTorch (both CPU and CUDA) to ensure reproducible results.
    It also sets PyTorch's cuDNN to deterministic mode.

    Args:
        seed (int): Random seed value to use

    Note:
        Setting cuDNN to deterministic mode may impact performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
