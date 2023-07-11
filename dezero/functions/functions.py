from dezero.core import Variable
from dezero.functions.add import Add
from dezero.functions.exp import Exp
from dezero.functions.square import Square


def exp(x: Variable) -> tuple[Variable]:
    f = Exp()
    return f(x)


def square(x: Variable) -> tuple[Variable]:
    f = Square()
    return f(x)


def add(x0: Variable, x1: Variable) -> tuple[Variable]:
    f = Add()
    return f(x0, x1)
