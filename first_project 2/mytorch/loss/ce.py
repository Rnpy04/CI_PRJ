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

def CategoricalCrossEntropy2(preds: Tensor, label: Tensor):
    """
    Categorical Cross Entropy Loss (multi-class)
    preds: (batch, classes), label: one-hot (batch, classes)
    """
    eps = 1e-12  # prevent log(0)
    p = np.clip(preds.data, eps, 1.0)
    log = np.log(p)
    loss = -np.sum(label.data * log) / preds.data.shape[0]  # mean over batch
    return Tensor(data=np.array(loss, dtype=np.float32), requires_grad=True) 


def BinaryCrossEntropy2(preds: Tensor, label: Tensor):
    log1 = preds.log(base=None)
    mul1 = label * log1

    log2 = (1 - preds).log(base=None)
    mul2 = (1 - label) * log2

    loss = -(mul1 + mul2).sum()

    size = Tensor(np.array([loss.data.size], dtype=np.float64))
    size = size ** -1
    return loss * size


def BinaryCrossEntropy(preds: Tensor, label: Tensor):
    """
    Binary Cross Entropy Loss
    preds: probabilities (0-1), label: 0 or 1
    """
    eps = 1e-12
    p = np.clip(preds.data, eps, 1.0 - eps)
    loss_per_elem = -(label.data * np.log(p) + (1 - label.data) * np.log(1 - p))
    loss = np.mean(loss_per_elem)
    return Tensor(data=np.array(loss, dtype=np.float32), requires_grad=True)
