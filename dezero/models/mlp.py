import dezero.functions as F
import dezero.layers as L
from dezero.core import Model, Variable


class MLP(Model):
    def __init__(self, fc_output_sizes: tuple[int, ...], activation: callable = F.sigmoid) -> None:
        super().__init__()
        self.fc_output_sizes = fc_output_sizes
        self.activation = activation

        self.layers = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
