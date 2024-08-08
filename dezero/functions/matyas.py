from dezero.core import Variable


def matyas(x: Variable, y: Variable) -> Variable:
    return 0.26 * (x**2 + y**2) - 0.48 * x * y
