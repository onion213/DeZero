import numpy as np

from dezero.core import Function, Variable
from dezero.functions import broadcast_to
from dezero.utils import reshape_sum_backward


class Sum(Function):
    def __init__(self, axis=None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: Variable) -> Variable:
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x: Variable, axis=None, keepdims: bool = False) -> Variable:
    return Sum(axis, keepdims)(x)
