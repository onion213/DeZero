from dezero.core import Variable
from dezero.functions.add import Add
from dezero.functions.exp import Exp
from dezero.functions.square import Square


def exp(x: Variable) -> Variable:
    f = Exp()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Exp` is 1-value function, but not returns Variable. returned value: {y}")
    return y


def square(x: Variable) -> Variable:
    f = Square()
    y = f(x)
    if not isinstance(y, Variable):
        raise TypeError(f"`Square` is 1-value function, but not returns Variable. returned value: {y}")
    return y


def add(x0: Variable, x1: Variable) -> Variable:
    f = Add()
    y = f(x0, x1)
    if not isinstance(y, Variable):
        raise TypeError(f"`Add` is 1-value function, but not returns Variable. returned value: {y}")
    return y
