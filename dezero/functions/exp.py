import numpy as np

from dezero.core import Function


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data

        gx: np.ndarray = np.exp(x) * gy
        return gx
