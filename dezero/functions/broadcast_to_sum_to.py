import numpy as np

from dezero.core import Function, Variable, as_variable


class BroadcastTo(Function):
    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: Variable, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx


def sum_to(x: Variable, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
