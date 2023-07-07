from dezero import variable


class Function:
    def __call__(self, input: variable.Variable):
        x = input.data
        y = self.forward(x)
        output = variable.Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()
