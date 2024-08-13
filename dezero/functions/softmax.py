import numpy as np

from dezero.core import Variable, as_variable
from dezero.functions.exp import exp


def softmax1d(x: Variable | np.ndarray) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = y.sum()
    return y / sum_y
