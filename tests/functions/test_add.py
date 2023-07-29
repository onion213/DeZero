from typing import Union

import numpy as np
import pytest

from dezero.core import Variable
from dezero.functions import Add


class TestAdd:
    params = {
        "__call__": {f"{x0}+{x1}": (x0, x1) for x0 in (0, 1, 2, -1, 0.1) for x1 in (0, 1, 2, -1, 0.1)},
        "backward": {
            f"x0={x0},x1={x1},gy={gy}": (x0, x1, gy)
            for x0 in (0, 1, 2, -1, 0.1)
            for x1 in (0, 1, 2, -1, 0.1)
            for gy in (0, 1, 2, -1, 0.1)
        },
    }

    @pytest.mark.parametrize("x0, x1", list(params["__call__"].values()), ids=params["__call__"].keys())
    def test_和の計算が正しく行われること(self, x0: Union[int, float], x1: Union[int, float]):
        # Arrange
        f = Add()
        v0 = Variable(np.array(x0))
        v1 = Variable(np.array(x1))
        expected_output = np.array(v0.data + v1.data)

        # Act
        o = f(v0, v1)

        # Assert
        assert isinstance(o, Variable)
        assert o.data == expected_output

    @pytest.mark.parametrize("x0, x1, gy", list(params["backward"].values()), ids=params["backward"].keys())
    def test_backwardの計算が正しく行われること(self, x0, x1, gy):
        # Arrange
        f = Add()
        v0 = Variable(np.array(x0))
        v1 = Variable(np.array(x1))
        _ = f(v0, v1)
        gy = np.array(gy)
        expected_output = (gy, gy)

        # Act
        gxs = f.backward(gy)

        # Assert
        assert gxs == expected_output

    def test_forwardされていない関数でbackwardをすると例外が発生する(self):
        # Arrange
        f = Add()

        # Act
        with pytest.raises(AttributeError) as e:
            _ = f.backward(np.array(1.0))

        # Assert
        assert e.type == AttributeError
