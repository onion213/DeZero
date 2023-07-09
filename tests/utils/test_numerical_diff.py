import numpy as np

from dezero.core import Variable
from dezero.functions import Square
from dezero.utils import numerical_diff


class TestNumericalDiff:
    def test_数値微分ができる(self):
        # Arrange
        f = Square()
        v = Variable(np.array((1,)))
        eps = 10e-4

        # Act
        d = numerical_diff(f, v, eps)

        # Assert
        assert np.allclose(2, d)  # allowed error is just fake implmentation for now
