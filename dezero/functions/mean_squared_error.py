import numpy as np

from dezero.core import Function, Variable


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        return (diff**2).sum() / len(diff)

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    return MeanSquaredError()(x0, x1)
