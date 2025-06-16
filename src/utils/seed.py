import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa courantes."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        import torch

        torch.manual_seed(seed)
        tf.random.set_seed(seed)
    except ModuleNotFoundError:
        pass
