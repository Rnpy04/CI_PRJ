import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    e_x = x.exp()
    e__x = (-x).exp()
    
    up = e_x - e__x
    down = (e_x + e__x) ** -1

    return up * down
