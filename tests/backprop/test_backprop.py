import numpy as np

import dezero.core
import dezero.functions


class TestBackprop:
    def test_順伝播(self):
        # Arrange
        x = dezero.core.Variable(np.array(2))
        a = dezero.functions.square(x)
        b = dezero.functions.square(a)
        c = dezero.functions.square(a)

        # Act
        y = dezero.core.add(b, c)

        # Assert
        assert y.data == 32

    def test_逆伝播(self):
        # Arrange
        x = dezero.core.Variable(np.array(2))
        a = dezero.functions.square(x)
        b = dezero.functions.square(a)
        c = dezero.functions.square(a)
        y = dezero.core.add(b, c)

        # Act
        y.backward()

        # Assert
        assert x.grad.data == 64
