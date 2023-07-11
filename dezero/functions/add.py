import numpy as np

from dezero.core import Function


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        if self.inputs is None:
            raise AttributeError
        return gy, gy
