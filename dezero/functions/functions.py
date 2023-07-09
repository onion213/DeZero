from dezero.core import Variable
from dezero.functions.exp import Exp
from dezero.functions.square import Square


def exp(x: Variable) -> Variable:
    f = Exp()
    return f(x)


def square(x: Variable) -> Variable:
    f = Square()
    return f(x)
