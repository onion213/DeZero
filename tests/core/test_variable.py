import numpy as np
import pytest

from dezero.core import Function, Variable
from dezero.functions.functions import add, square


class TestVariable:
    def test_データを与えて初期化できること(self):
        # Arrange
        var = Variable(np.array(1.0))

        # Act
        # Nothing for this case

        # Assert
        assert isinstance(var, Variable)
        assert var.data == np.array(1.0)

    def test_データを与えない場合初期化できないこと(self):
        # Arrange
        with pytest.raises(TypeError) as e:
            _ = Variable()

        # Act
        # Nothing for this case

        # Assert
        assert e.type == TypeError

    def test_データにndarray以外を渡した場合初期化できないこと(self):
        # Arrange
        with pytest.raises(TypeError) as e:
            _ = Variable(1.0)

        # Act
        # Nothing for this case

        # Assert
        assert e.type == TypeError

    def test_初期化後にデータを代入できること(self):
        # Arrange
        var = Variable(np.array(1.0))

        # Act
        var.data = np.array(2.0)

        # Assert
        assert isinstance(var, Variable)
        assert var.data == np.array(2.0)

    def test_set_creatorメソッドでcreatorを保存できること(self):
        # Arrange
        var = Variable(np.array(1.0))
        func = Function()

        # Act
        var.set_creator(func)

        # Assert
        assert var.creator == func

    def test_逆伝播を正しく行える(self):
        # Arrange
        x = Variable(np.array(1.0))
        y = square(x)

        # Act
        y.backward()

        # Assert
        assert x.grad == np.array(2.0)

    def test_変数を繰り返し使用できる(self):
        # Arrange
        x = Variable(np.array(3))
        y = add(x, x)

        # Act
        y.backward()

        # Assert
        assert y.grad == np.array(1.0)
        assert x.grad == np.array(2.0)

    def test_変数のgradを初期化できる(self):
        # Arrange
        x = Variable(np.array(3))

        y1 = add(x, x)
        y1.backward()
        y2 = add(add(x, x), x)

        # Act
        x.cleargrad()
        y2.backward()

        # Assert
        assert x.grad == np.array(3.0)
