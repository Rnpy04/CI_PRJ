import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    
    ones = np.ones((x.shape[-1], 1))
    e_x = x.exp()
    sum = (e_x @ ones)

    return e_x * (sum**-1)

def softmax_no_overflow(x: Tensor) -> Tensor:
    """
    Numerically stable softmax.
    Assumes x.data shape = (batch, classes)
    Gradient is computed automatically via chain rule.
    """
    # subtract max for numerical stability
    x_max = np.max(x.data, axis=1, keepdims=True)        # (batch, 1)
    e_x = np.exp(x.data - x_max)                         # stable exp
    sum_e = np.sum(e_x, axis=1, keepdims=True)          # (batch, 1)
    data = e_x / sum_e                                  # softmax output

    return Tensor(data.astype(np.float32), requires_grad=x.requires_grad)
