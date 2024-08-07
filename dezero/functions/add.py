import numpy as np

from dezero.core import Function, Variable


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        if self.inputs is None:
            raise AttributeError
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    f = Add()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Add` is 1-value function, but not returns Variable. returned value: {y}")
    return y
