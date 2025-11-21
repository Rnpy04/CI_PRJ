from mytorch import Tensor
import numpy as np

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    # loss = -np.sum(label * np.log(preds)) / preds.shape[0]
    log = preds.log(base=None) # natural log
    mul = label * log
    loss = -mul.sum()

    size = Tensor(np.array([loss.data.size], dtype=np.float64))
    size = size ** -1
    return loss * size
    # return ...

def BinaryCrossEntropy(preds: Tensor, label: Tensor):
    log1 = preds.log(base=None)
    mul1 = label * log1

    log2 = (1 - preds).log(base=None)
    mul2 = (1 - label) * log2

    loss = -(mul1 + mul2).sum()

    size = Tensor(np.array([loss.data.size], dtype=np.float64))
    size = size ** -1
    return loss * size

