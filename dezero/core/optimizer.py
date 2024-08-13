from dezero.core.model import Model
from dezero.core.parameter import Parameter


class Optimizer:
    def __init__(self):
        self.target: Model | None = None
        self.hooks: list[callable] = []

    def setup(self, target: Model):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter):
        raise NotImplementedError()

    def add_hook(self, f: callable):
        self.hooks.append(f)
