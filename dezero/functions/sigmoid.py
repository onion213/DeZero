import numpy as np

from dezero.core import Function, Variable


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, gy: Variable) -> Variable:
        y: np.ndarray = self.outputs[0].data
        gx: np.ndarray = gy * y * (1 - y)
        return gx


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)
