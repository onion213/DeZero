import numpy as np

from dezero.core import Function
from dezero.functions.matmul import matmul


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        y = np.dot(x, W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        gb = gy.sum(0)
        return gx, gW, gb


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    return Linear()(x, W, b)
