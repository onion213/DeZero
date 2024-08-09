import numpy as np
import pytest

from dezero.core import Variable, add
from dezero.functions import square


class TestVariable:
    def test_データを与えて初期化できる(self):
        # Arrange
        var = Variable(np.array(1.0))

        # Act
        # Nothing for this case

        # Assert
        assert isinstance(var, Variable)
        assert var.data == np.array(1.0)
        assert var.name is None
        assert var.grad is None
        assert var.creator is None
        assert var.generation == 0

    def test_データを与えない場合初期化できない(self):
        # Arrange
        with pytest.raises(TypeError) as e:
            _ = Variable()

        # Act
        # Nothing for this case

        # Assert
        assert e.type is TypeError

    def test_データにndarray以外を渡した場合初期化できない(self):
        # Arrange
        with pytest.raises(TypeError) as e:
            _ = Variable(1.0)

        # Act
        # Nothing for this case

        # Assert
        assert e.type is TypeError

    def test_初期化後にデータを代入できる(self):
        # Arrange
        var = Variable(np.array(1.0))

        # Act
        var.data = np.array(2.0)

        # Assert
        assert isinstance(var, Variable)
        assert var.data == np.array(2.0)
        assert var.grad is None
        assert var.creator is None
        assert var.generation == 0

    def test_順伝播でcreatorが保存される(self):
        # Arrange
        x = Variable(np.array(1.0))
        func = square
        y = func(x)

        # Act
        # Nothing for this case

        # Assert
        assert y.creator is not None
        assert y.generation == 1

    def test_逆伝播を正しく行える(self):
        # Arrange
        x = Variable(np.array(1.0))
        y = square(x)

        # Act
        y.backward()

        # Assert
        assert x.grad.data == np.array(2.0)

    def test_変数を繰り返し使用できる(self):
        # Arrange
        x = Variable(np.array(3))
        y = add(x, x)

        # Act
        y.backward(retain_grad=True)

        # Assert
        assert y.grad.data == np.array(1.0)
        assert x.grad.data == np.array(2.0)

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
        assert x.grad.data == np.array(3.0)

    def test_変数にnameを設定できる(self):
        # Arrange
        x = Variable(np.array(3), name="x")

        # Act
        # Nothing for this case

        # Assert
        assert x.name == "x"

    def test_変数から直接dataのpropertyを取得できる(self):
        # Arrange
        x = Variable(np.array([3]))

        # Act
        # Nothing for this case

        # Assert
        assert x.shape == (1,)
        assert x.ndim == 1
        assert x.size == 1
        assert x.dtype == np.dtype(np.int64)
        assert len(x) == 1
