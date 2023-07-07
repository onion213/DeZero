import math

import numpy as np
import pytest

from dezero.functions import Exp
from dezero.variable import Variable


class TestExp:
    params = {f"e^{i}": (i, np.exp(i)) for i in (0, 1, 2, -1, 0.1)}

    @pytest.mark.parametrize("input, expected_output", list(params.values()), ids=params.keys())
    def test_指数関数の計算が正しく行われること(self, input, expected_output):
        # Arrange
        f = Exp()
        v = Variable(np.array(input))

        # Act
        o = f(v)

        # Assert
        assert isinstance(o, Variable)
        assert o.data == expected_output
