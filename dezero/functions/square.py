import numpy as np

from dezero.core import Function, Variable


class Square(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return np.array(xs[0] ** 2)

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data
        gx: np.ndarray = 2 * x * gys[0]
        return gx


def square(x: Variable) -> Variable:
    f = Square()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Square` is 1-value function, but not returns Variable. returned value: {y}")
    return y
