from dezero.core import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    return 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
