from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], lr=0.1, beta1=0.9):
        super().__init__(layers)
        self.lr = lr
        self.beta1 = beta1
    # def __init__(self):
    #     pass

    def step(self):
        # pass
        for layer in self.layers:
            if not hasattr(layer, 'cache'):
                print('initialize cache')
                layer.cache = {'v': {'weight': np.zeros_like(layer.weight.data)}}
                if layer.need_bias:
                    layer.cache['v']['bias'] = np.zeros_like(layer.bias.data)

            weight_grad = layer.weight.grad.data
            layer.cache['v']['weight'] = self.beta1 * layer.cache['v']['weight'] + (1 - self.beta1) * weight_grad
            layer.weight.data -= self.lr * layer.cache['v']['weight']

            if layer.need_bias:
                bias_grad = layer.bias.grad.data
                layer.cache['v']['bias'] = self.beta1 * layer.cache['v']['bias'] + (1 - self.beta1) * bias_grad
                layer.bias.data -= self.lr * layer.cache['v']['bias']
            layer.zero_grad()
