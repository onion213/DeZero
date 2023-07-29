from typing import Optional, Union

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"data must be np.ndarray. given: {type(data)}")

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional["Function"] = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func

    def backward(self):
        if self.creator is None:
            raise AttributeError("`creator` is not set for this variable.")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys: tuple[np.ndarray] = tuple(output.grad for output in f.outputs)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self) -> None:
        self.grad = None


def as_array(x: Union[np.ndarray, np.number]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable) -> Union[Variable, tuple[Variable, ...]]:
        xs = (input.data for input in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple(Variable(as_array(y)) for y in ys)

        for output in outputs:
            output.set_creator(self)
        self.inputs = tuple(inputs)
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError()

    def backward(self, *gys: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError()
