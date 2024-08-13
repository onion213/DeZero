import numpy as np

from dezero.core import Variable, as_variable
from dezero.functions.clip import clip
from dezero.functions.exp import exp
from dezero.functions.log import log

# TODO: Implement SoftMax class


def softmax1d(x: Variable | np.ndarray) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = y.sum()
    return y / sum_y


def softmax_simple(x: Variable | np.ndarray, axis: int = 1) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = y.sum(axis=axis, keepdims=True)
    return y / sum_y


def softmax_cross_entropy_simple(x: Variable | np.ndarray, t: Variable | np.ndarray) -> Variable:
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * tlog_p.sum() / N
    return y
