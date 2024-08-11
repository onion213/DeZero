import weakref

from dezero.core.core import Variable
from dezero.core.parameter import Parameter


class Layer:
    def __init__(self) -> None:
        self._params: set[Parameter] = set()

    def __setattr__(self, name: str, value: Variable) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def params(self) -> set[Parameter]:
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def __call__(self, *inputs) -> Variable:
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs) -> Variable:
        raise NotImplementedError()

    def cleargrads(self) -> None:
        for param in self.params():
            param.cleargrad()
