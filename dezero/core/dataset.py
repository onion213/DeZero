from typing import Callable

import numpy as np


class Dataset:
    def __init__(
        self, train: bool = True, transform: Callable | None = None, target_transform: Callable | None = None
    ):
        self.train: bool = train
        if transform is None:
            self.transform: callable = lambda x: x
        else:
            self.transform = transform
        if target_transform is None:
            self.target_transform: callable = lambda x: x
        else:
            self.target_transform = target_transform

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index: int):
        assert np.isscalar(index)
        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
