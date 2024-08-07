import numpy as np

import dezero.core
import dezero.functions


class TestBackprop:
    def test_順伝播(self):
        # Arrange
        x = dezero.core.Variable(np.array(2))
        a = dezero.functions.functions.square(x)
        b = dezero.functions.functions.square(a)
        c = dezero.functions.functions.square(a)

        # Act
        y = dezero.functions.functions.add(b, c)

        # Assert
        assert y.data == 32

    def test_逆伝播(self):
        # Arrange
        x = dezero.core.Variable(np.array(2))
        a = dezero.functions.functions.square(x)
        b = dezero.functions.functions.square(a)
        c = dezero.functions.functions.square(a)
        y = dezero.functions.functions.add(b, c)

        # Act
        y.backward()

        # Assert
        assert x.grad == 64
