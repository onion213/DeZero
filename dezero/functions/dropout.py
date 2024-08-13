import numpy as np

from dezero.core import Config, as_variable


def dropout(x, dropout_ratio: float = 0.5):
    x = as_variable(x)
    if Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
