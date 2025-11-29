import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    z = -x
    z = z.exp()

    ones = Tensor(data=np.ones_like(z.data), requires_grad=False)
    z = (z + ones)** -1
    
    return z

def sigmoid_no_overflow(x: Tensor) -> Tensor:
    """
    Numerically stable sigmoid using a vectorized approach.
    Gradient is computed automatically via chain rule.
    """
    # Convert to numpy array for masks
    data = x.data
    # vectorized computation
    out_data = np.where(
        data >= 0,
        1 / (1 + np.exp(-data)),      # x >= 0
        np.exp(data) / (1 + np.exp(data))  # x < 0
    )
    return Tensor(out_data, requires_grad=True)