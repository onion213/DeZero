import numpy as np
import pytest

from dezero.functions import Square
from dezero.variable import Variable


class TestSquare:
    params = {"0^2=0": (0, 0), "1^2=1": (1, 1), "2^2=4": (2, 4), "-5^2=25": (-5, 25)}

    @pytest.mark.parametrize("input, expected_output", list(params.values()), ids=params.keys())
    def test_2上の計算が正常に行われること(self, input, expected_output):
        # Arrange
        f = Square()
        i = Variable(np.array(input))

        # Act
        o = f(i)

        # Assert
        assert isinstance(o, Variable)
        assert o.data == np.array(expected_output)
