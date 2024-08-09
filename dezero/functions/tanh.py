import numpy as np

from dezero.core import Function, Variable


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy.data * (1 - y**2)
        return gx


def tanh(x: Variable) -> Variable:
    return Tanh()(x)
