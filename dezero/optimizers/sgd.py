from dezero.core import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
