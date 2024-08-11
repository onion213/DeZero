import numpy as np

import dezero.layers as L
from dezero import Variable


class TestLinearLayer:
    def test_forward(self):
        # Arrange
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        linear = L.Linear(3)

        # Act
        y = linear(x)

        # Assert
        assert y.shape == (2, 3)
        assert (y.data == x.data @ linear.W.data + linear.b.data).all()

    def test_cleargrads(self):
        # Arrange
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        linear = L.Linear(3)
        linear(x)

        # Act
        linear.cleargrads()

        # Assert
        for param in linear.params():
            assert param.grad is None
