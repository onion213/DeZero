import numpy as np

from dezero.functions.function import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
