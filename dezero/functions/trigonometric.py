import numpy as np

from dezero.core import Function, Variable


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: Variable) -> Variable:
        x: Variable = self.inputs[0]

        gx: Variable = cos(x) * gy
        return gx


def sin(x: Variable) -> Variable:
    f = Sin()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Sin` is 1-value function, but not returns Variable. returned value: {y}")
    return y


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: Variable) -> Variable:
        x: Variable = self.inputs[0]

        gx: Variable = -sin(x) * gy
        return gx


def cos(x: Variable) -> Variable:
    f = Cos()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Cos` is 1-value function, but not returns Variable. returned value: {y}")
    return y
