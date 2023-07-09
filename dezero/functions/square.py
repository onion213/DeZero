import numpy as np

from dezero.core import Function


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.input.data
        gx: np.float64 = 2 * x * gy
        return gx
