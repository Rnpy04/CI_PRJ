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
