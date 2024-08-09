import weakref
from typing import Optional, Union

import numpy as np

from dezero.core import config


def as_array(x: Union[np.ndarray, np.number]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"data must be np.ndarray. given: {type(data)}")

        self.data: np.ndarray = data
        self.name: Optional[str] = name
        self.grad: Optional[Variable] = None
        self.creator: Optional["Function"] = None
        self.generation: int = 0

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = True) -> None:
        if self.creator is None:
            raise AttributeError("`creator` is not set for this variable.")

        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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
                    return funcs, seen_funcs
            funcs.append(f)
            seen_funcs.add(id(f))
            return funcs, seen_funcs

        funcs, seen_funcs = add_func(self.creator, [], set())
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

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


def as_variable(obj: Union[Variable, np.ndarray]) -> Variable:
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, np.ndarray):
        return Variable(obj)
    raise TypeError(f"Invalid type: {type(obj)}")


class Function:
    def __call__(self, *inputs: Union[Variable, np.ndarray]) -> Union[Variable, tuple[Variable, ...]]:
        inputs = [as_variable(x) for x in inputs]
        xs = tuple(x.data for x in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if config.Config.enable_backprop:
            self.generation = max(x.generation for x in inputs)

            for output in outputs:
                output.set_creator(self)
            self.inputs = tuple(inputs)
            self.outputs = tuple(weakref.ref(output) for output in outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError()

    def backward(self, *gys: Variable) -> Union[Variable, tuple[Variable, ...]]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        if self.inputs is None:
            raise AttributeError
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    f = Add()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Add` is 1-value function, but not returns Variable. returned value: {y}")
    return y


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        if self.inputs is None:
            raise AttributeError
        return gy * self.inputs[1], gy * self.inputs[0]


def mul(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    f = Mul()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Mul` is 1-value function, but not returns Variable. returned value: {y}")
    return y


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


def neg(x: Variable) -> Variable:
    f = Neg()
    return f(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        if self.inputs is None:
            raise AttributeError
        return gy, -gy


def sub(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    f = Sub()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Sub` is 1-value function, but not returns Variable. returned value: {y}")
    return y


def rsub(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    f = Sub()
    y = f(x1, x0)
    if not isinstance(y, Variable):
        raise TypeError(f"`Sub` is 1-value function, but not returns Variable. returned value: {y}")
    return y


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        if self.inputs is None:
            raise AttributeError
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    f = Div()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Div` is 1-value function, but not returns Variable. returned value: {y}")
    return y


def rdiv(x0: Variable, x1: Variable) -> Variable:
    if not isinstance(x0, Variable):
        x0 = as_array(x0)
    f = Div()
    y = f(x1, x0)
    if not isinstance(y, Variable):
        raise TypeError(f"`Div` is 1-value function, but not returns Variable. returned value: {y}")
    return y


class Pow(Function):
    def __init__(self, c: float) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**self.c

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: Variable, c: float) -> Variable:
    f = Pow(c)
    return f(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
