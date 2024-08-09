import numpy as np

from dezero.core import Function, Variable, as_variable


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


def reshape(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
