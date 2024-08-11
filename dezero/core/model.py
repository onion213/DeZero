from dezero.core.layer import Layer
from dezero.utils import plot_dot_graph


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)
