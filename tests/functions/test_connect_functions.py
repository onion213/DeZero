import numpy as np
import pytest

from dezero.functions import Exp, Square
from dezero.variable import Variable


class TestConnectFunctions:
    params = {f"(e^({i}^2))^2": (i, np.exp(i**2) ** 2) for i in (0, 1, 2, -1, 0.1)}

    @pytest.mark.parametrize("input, expected_output", list(params.values()), ids=params.keys())
    def test_関数の連結が正しく行われる(self, input, expected_output):
        # Arrange
        sq = Square()
        exp = Exp()
        v = Variable(np.array(input))

        # Act
        o = sq(exp(sq(v)))

        # Assert
        assert isinstance(o, Variable)
        assert o.data == expected_output
