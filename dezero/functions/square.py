import numpy as np

from dezero.core import Function


class Square(Function):
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        return (np.array(xs[0] ** 2),)

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray]:
        x: np.ndarray = self.inputs[0].data
        gx: np.ndarray = 2 * x * gys[0]
        return (gx,)
