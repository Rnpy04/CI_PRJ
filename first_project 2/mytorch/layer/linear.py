from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        if self.need_bias:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight
        # return ...

    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        self.weight = Tensor(
            # data=...,
            # requires_grad=...
            data= initializer((self.inputs, self.outputs), mode=self.initialize_mode),
            requires_grad= True
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                # data=...,
                # requires_grad=...
                data= initializer((1, self.outputs), mode="zero"),
                requires_grad=...
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()
        # pass

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return {"weight": self.weight, "bias": self.bias}
        else:
            return {"weight": self.weight}
        # return ...

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
