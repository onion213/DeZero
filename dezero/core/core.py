from typing import Optional, Union

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"data must be np.ndarray. given: {type(data)}")

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creater: Optional["Function"] = None

    def set_creater(self, func: "Function") -> None:
        self.creater = func

    def backward(self):
        if self.creater is None:
            raise AttributeError("`creater` is not set for this variable.")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creater]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creater is not None:
                funcs.append(x.creater)


def as_array(x: Union[np.ndarray, np.number]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creater(self)
        self.output = output
        self.input = input
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
