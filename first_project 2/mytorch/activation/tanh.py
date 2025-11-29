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

def tanh_no_overflow(x: Tensor) -> Tensor:
    """
    Numerically stable tanh using a vectorized approach.
    Gradient is computed automatically via chain rule.
    """
    # Convert to numpy array for masks
    data = x.data
    # vectorized computation
    out_data = np.where(
        data >= 0,
        (1 - np.exp(-2 * data)) / (1 + np.exp(-2 * data)),  # x >= 0
        (np.exp(2 * data) - 1) / (np.exp(2 * data) + 1)     # x < 0
    )
    return Tensor(out_data, requires_grad=True)