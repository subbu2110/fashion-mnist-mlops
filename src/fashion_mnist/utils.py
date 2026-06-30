import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs reproducible across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return "mps"
    return "cpu"
