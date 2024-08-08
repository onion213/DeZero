import numpy as np

from dezero.core import Function, Variable


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data

        gx: np.ndarray = np.cos(x) * gy
        return gx


def sin(x: Variable) -> Variable:
    f = Sin()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Sin` is 1-value function, but not returns Variable. returned value: {y}")
    return y
