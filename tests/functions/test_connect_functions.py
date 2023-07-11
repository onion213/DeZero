import numpy as np
import pytest

from dezero.core import Variable
from dezero.functions import Exp, Square


class TestConnectFunctions:
    params = {f"(e^({i}^2))^2": (i, np.exp(i**2) ** 2) for i in (0, 1, 2, -1, 0.1)}

    @pytest.mark.parametrize("input, expected_output", list(params.values()), ids=params.keys())
    def test_関数の連結が正しく行われる(self, input, expected_output):
        # Arrange
        sq = Square()
        exp = Exp()
        v = Variable(np.array(input))

        def f(v: Variable) -> Variable:
            return sq(exp(sq(v)[0])[0])[0]

        # Act
        o = f(v)

        # Assert
        assert isinstance(o, Variable)
        assert o.data == expected_output

    def test_連結した関数で逆電波が正しく行われる(self):
        # Arrange
        input = np.array(0.5)
        sq1 = Square()
        exp = Exp()
        sq2 = Square()

        def f(v: Variable) -> Variable:
            return sq2(exp(sq1(v)[0])[0])[0]

        x = Variable(np.array(input))
        y = f(x)
        y.grad = np.array(1.0)

        expected_x_grad = 2 * 0.5 * np.exp(0.5**2) * 2 * np.exp(0.5**2)

        # Act
        y.backward()

        # Assert
        assert x.grad is not None
        assert x.grad == expected_x_grad
