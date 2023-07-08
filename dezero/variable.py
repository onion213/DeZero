from typing import Optional

import numpy as np


class Variable:
    def __init__(self, data: np.array) -> None:
        self.data: np.array = data
        self.grad: Optional[np.float64] = None
