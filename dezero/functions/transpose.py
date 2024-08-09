import numpy as np

from dezero.core import Function, Variable


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...]) -> None:
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.transpose(x, axes=self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, axes=inv_axes)


def transpose(x: Variable, axes: tuple[int, ...] = None) -> Variable:
    return Transpose(axes)(x)
