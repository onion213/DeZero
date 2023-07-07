import numpy as np
import pytest

from dezero.variable import Variable


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

    def test_初期化後にデータを代入できること(self):
        # Arrange
        var = Variable(np.array(1.0))

        # Act
        var.data = np.array(2.0)

        # assert
        assert isinstance(var, Variable)
        assert var.data == np.array(2.0)