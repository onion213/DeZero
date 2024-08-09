import numpy as np

from dezero.core import Function


class MatMul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)
