from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

"TODO: (optional) implement Adam optimizer"
class Adam(Optimizer):
    def __init__(self, layers: List[Layer], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(layers)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
    # def __init__(self):
    #     pass
    
    # def step(self):
    #     # pass
    #     self.t += 1
    #     for layer in self.layers:
    #         if not hasattr(layer, 'cache'):
    #             print('initialize cache')
    #             layer.cache = {
    #                 's': {'weight': np.zeros_like(layer.weight.data)},
    #                 'v': {'weight': np.zeros_like(layer.weight.data)}
    #             }
    #             if layer.need_bias:
    #                 layer.cache['s']['bias'] = np.zeros_like(layer.bias.data)
    #                 layer.cache['v']['bias'] = np.zeros_like(layer.bias.data)

    #         weight_grad = layer.weight.grad.data
    #         layer.cache['s']['weight'] = self.beta1 * layer.cache['s']['weight'] + (1 - self.beta1) * weight_grad
    #         layer.cache['v']['weight'] = self.beta2 * layer.cache['v']['weight'] + (1 - self.beta2) * (weight_grad ** 2)

    #         s_hat_weight = layer.cache['s']['weight'] / (1 - self.beta1 ** self.t)
    #         v_hat_weight = layer.cache['v']['weight'] / (1 - self.beta2 ** self.t)

    #         layer.weight.data -= self.lr * s_hat_weight / (np.sqrt(v_hat_weight) + self.eps)

    #         if layer.need_bias:
    #             bias_grad = layer.bias.grad.data
    #             layer.cache['s']['bias'] = self.beta1 * layer.cache['s']['bias'] + (1 - self.beta1) * bias_grad
    #             layer.cache['v']['bias'] = self.beta2 * layer.cache['v']['bias'] + (1 - self.beta2) * (bias_grad ** 2)

    #             m_hat_bias = layer.cache['s']['bias'] / (1 - self.beta1 ** self.t)
    #             v_hat_bias = layer.cache['v']['bias'] / (1 - self.beta2 ** self.t)

    #             layer.bias.data -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.eps)


    def step(self):
        self.t += 1
        for layer in self.layers:
            if not hasattr(layer, 'weight'):
                continue

            # initialize cache silently
            if not hasattr(layer, 'cache'):
                layer.cache = {
                    'm': {'weight': np.zeros_like(layer.weight.data, dtype=layer.weight.data.dtype)},
                    'v': {'weight': np.zeros_like(layer.weight.data, dtype=layer.weight.data.dtype)}
                }
                if layer.need_bias:
                    layer.cache['m']['bias'] = np.zeros_like(layer.bias.data, dtype=layer.bias.data.dtype)
                    layer.cache['v']['bias'] = np.zeros_like(layer.bias.data, dtype=layer.bias.data.dtype)

            # weight update
            if getattr(layer.weight, 'grad', None) is not None:
                weight_grad = layer.weight.grad.data
                # optional: clip to avoid explosion
                # weight_grad = np.clip(weight_grad, -1e3, 1e3)

                layer.cache['m']['weight'] = self.beta1 * layer.cache['m']['weight'] + (1 - self.beta1) * weight_grad
                layer.cache['v']['weight'] = self.beta2 * layer.cache['v']['weight'] + (1 - self.beta2) * (weight_grad ** 2)

                m_hat_w = layer.cache['m']['weight'] / (1 - self.beta1 ** self.t)
                v_hat_w = layer.cache['v']['weight'] / (1 - self.beta2 ** self.t)

                layer.weight.data -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

            # bias update
            if layer.need_bias and getattr(layer.bias, 'grad', None) is not None:
                bias_grad = layer.bias.grad.data

                layer.cache['m']['bias'] = self.beta1 * layer.cache['m']['bias'] + (1 - self.beta1) * bias_grad
                layer.cache['v']['bias'] = self.beta2 * layer.cache['v']['bias'] + (1 - self.beta2) * (bias_grad ** 2)

                m_hat_b = layer.cache['m']['bias'] / (1 - self.beta1 ** self.t)
                v_hat_b = layer.cache['v']['bias'] / (1 - self.beta2 ** self.t)

                layer.bias.data -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
