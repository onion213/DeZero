import weakref
from typing import Optional, Union

import numpy as np

from dezero.core.config import Config


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"data must be np.ndarray. given: {type(data)}")

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional["Function"] = None
        self.generation: int = 0

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = True) -> None:
        if self.creator is None:
            raise AttributeError("`creator` is not set for this variable.")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        def add_func(
            f: "Function", funcs: list["Function"], seen_funcs: set[int]
        ) -> tuple[list["Function"], set[int]]:
            if id(f) in seen_funcs:
                return funcs, seen_funcs
            if len(funcs) == 0:
                funcs = [f]
                seen_funcs.add(id(f))
                return funcs, seen_funcs
            for i, func in enumerate(funcs):
                if f.generation <= func.generation:
                    funcs.insert(i, f)
                    seen_funcs.add(id(f))
                    break
            return funcs, seen_funcs

        funcs = [self.creator]
        seen_funcs = set((id(self.creator),))
        while funcs:
            f = funcs.pop()
            gys: tuple[np.ndarray] = tuple(output().grad for output in f.outputs)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs, seen_funcs = add_func(x.creator, funcs, seen_funcs)
            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

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

        if Config.enable_backprop:
            self.generation = max(input.generation for input in inputs)

            for output in outputs:
                output.set_creator(self)
            self.inputs = tuple(inputs)
            self.outputs = tuple(weakref.ref(output) for output in outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError()

    def backward(self, *gys: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError()
