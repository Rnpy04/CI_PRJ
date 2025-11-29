from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
from mytorch.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        # pass
        for layer in self.layers:
            if getattr(layer.weight, 'grad', None) is not None:
                layer.weight.data = layer.weight.data - self.learning_rate * layer.weight.grad.data

            if layer.need_bias and getattr(layer.bias, 'grad', None) is not None:
                layer.bias.data = layer.bias.data - self.learning_rate * layer.bias.grad.data
                
                
