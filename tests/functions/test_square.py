from typing import Union

import numpy as np
import pytest

from dezero.core import Variable
from dezero.functions import Square


class TestExp:
    params = {
        "__call__": {f"e^{i}": (i,) for i in (0, 1, 2, -1, 0.1)},
        "backward": {f"x={x},gy={gy}": (x, gy) for x in (0, 1, 2, -1, 0.1) for gy in (0, 1, 2, -1, 0.1)},
    }

    @pytest.mark.parametrize("input", list(params["__call__"].values()), ids=params["__call__"].keys())
    def test_指数関数の計算が正しく行われること(self, input: Union[int, float]):
        # Arrange
        f = Square()
        v = Variable(np.array(input))
        expected_output = np.power(input, 2)

        # Act
        o = f(v)

        # Assert
        assert isinstance(o, Variable)
        np.testing.assert_array_equal(o.data, expected_output)

    @pytest.mark.parametrize("input, gy", list(params["backward"].values()), ids=params["backward"].keys())
    def test_backwardの計算が正しく行われること(self, input, gy):
        # Arrange
        f = Square()
        v = Variable(np.array(input))
        _ = f(v)
        gy = np.float64(gy)

        # Act
        gx = f.backward(gy)

        # Assert
        assert isinstance(gx, np.float64)
        assert gx == 2 * input * gy

    def test_forwardされていない関数でbackwardをすると例外が発生する(self):
        # Arrange
        f = Square()

        # Act
        with pytest.raises(AttributeError) as e:
            _ = f.backward(np.array(1.0))

        # Assert
        assert e.type == AttributeError
