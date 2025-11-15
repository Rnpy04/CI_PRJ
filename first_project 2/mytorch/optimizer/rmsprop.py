from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement RMSprop optimizer"
class RMSprop(Optimizer):
    # def __init__(self):
    #     pass
    def __init__(self, layers: List[Layer], lr=0.1, beta2=0.99, eps=1e-8):
        super().__init__(layers)
        self.lr = lr
        self.beta2 = beta2
        self.eps = eps

    def step(self):
        for layer in self.layers:
            if not hasattr(layer, 'cache'):
                print('initialize cache')
                layer.cache = {'s': {'weight': np.zeros_like(layer.weight.data)}}
                if layer.need_bias:
                    layer.cache['s']['bias'] = np.zeros_like(layer.bias.data)

            weight_grad = layer.weight.grad.data
            layer.cache['s']['weight'] = self.beta2 * layer.cache['s']['weight'] + (1 - self.beta2) * weight_grad**2
            layer.weight.data -= self.lr * weight_grad / (np.sqrt(layer.cache['s']['weight']) + self.eps)

            if layer.need_bias:
                bias_grad = layer.bias.grad.data
                layer.cache['s']['bias'] = self.beta2 * layer.cache['s']['bias'] + (1 - self.beta2) * bias_grad**2
                layer.bias.data -= self.lr * bias_grad / (np.sqrt(layer.cache['s']['bias']) + self.eps)
        # pass
