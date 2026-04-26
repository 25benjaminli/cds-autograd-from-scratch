import random

import numpy as np

from vectorized_engine import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, in_channels, out_channels, nonlin=True):
        self.in_channels = in_channels
        # some initialization scheme to keep variance of activations stable across layers
        self.mat = Tensor(np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels), _name="weight")
        self.bias = Tensor(np.zeros((1, out_channels)), _name="bias")
        self.nonlin = nonlin

    def __call__(self, x):
        x = x @ self.mat + self.bias
        return x.relu() if self.nonlin else x

    def parameters(self):
        return [self.mat, self.bias]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.in_channels)}]"

class MLP(Module):
    def __init__(self, in_channels: int, hidden_neurons: list[int]):
        sizes = [in_channels] + hidden_neurons
        # apply the nonlinear layer to all but the final layer
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i!=len(hidden_neurons)-1) for i in range(len(hidden_neurons))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == "__main__":
    # test the mlp init
    mlp = MLP(4, [8, 16, 8, 4])
    x = Tensor(np.random.randn(2, 4), _name="input")
    out = mlp(x)
    print(out)