import numpy as np

from dezero.core import Function, Variable


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        if self.inputs is None:
            raise AttributeError
        return gy * self.inputs[1].data, gy * self.inputs[0].data


def mul(x0: Variable, x1: Variable) -> Variable:
    f = Mul()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Mul` is 1-value function, but not returns Variable. returned value: {y}")
    return y
