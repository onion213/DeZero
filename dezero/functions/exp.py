import numpy as np

from dezero.core import Function, Variable


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data

        gx: np.ndarray = np.exp(x) * gy
        return gx


def exp(x: Variable) -> Variable:
    f = Exp()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Exp` is 1-value function, but not returns Variable. returned value: {y}")
    return y
