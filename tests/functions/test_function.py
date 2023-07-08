import numpy as np
import pytest

from dezero.functions.function import Function
from dezero.variable import Variable


class TestFunction:
    def test___call__メソッドを呼ぶと例外を返す(self):
        # Arrange
        f = Function()
        v = Variable(np.array(1.0))

        # Act
        with pytest.raises(NotImplementedError) as e:
            _ = f(v)

        # assert
        assert e.type == NotImplementedError

    def test_forwardメソッドを呼ぶと例外を返す(self):
        # Arrange
        f = Function()
        x = np.array(1.0)

        # Act
        with pytest.raises(NotImplementedError) as e:
            _ = f.forward(x)

        # assert
        assert e.type == NotImplementedError

    def test_backwardメソッドを呼ぶと例外を返す(self):
        # Arrange
        f = Function()
        gy = np.array(1.0)

        # Act
        with pytest.raises(NotImplementedError) as e:
            _ = f.backward(gy)

        # assert
        assert e.type == NotImplementedError
