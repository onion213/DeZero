from typing import Optional

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: Optional[np.float64] = None
        self.creater: Optional["Function"] = None

    def set_creater(self, func: "Function") -> None:
        self.creater = func

    def backward(self):
        f = self.creater
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creater(self)
        self.output = output
        self.input = input
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.float64) -> np.float64:
        raise NotImplementedError()
