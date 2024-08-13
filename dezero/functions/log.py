import numpy as np

from dezero.core import Function, Variable


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data

        gx: np.ndarray = gy / x
        return gx


def log(x: Variable) -> Variable:
    f = Log()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Log` is 1-value function, but not returns Variable. returned value: {y}")
    return y
