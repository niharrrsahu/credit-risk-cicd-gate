"""Utility functions for seed control, hashing, and reporting."""

import random
import os
import numpy as np

def set_global_seed(seed: int = 42):
    """Sets deterministic seeds across python, numpy, etc."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
