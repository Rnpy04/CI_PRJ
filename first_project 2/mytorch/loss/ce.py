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

# def BinaryCrossEntropy(preds: Tensor, label: Tensor):
#     log = preds.log(base=None)
#     mul = label * log

#     log2 = (1 - preds).log(base=None)
#     mul2 = (1 - label) * log2

#     loss = -(mul + mul2).sum()

#     size = Tensor(np.array([loss.data.size], dtype=np.float64))
#     size = size ** -1
#     return loss * size
    
def BinaryCrossEntropy(preds: Tensor, label: Tensor, eps=1e-12):
    # فرض: preds و label اشکال مشابه دارند و label شامل 0/1 است
    # جلوگیری از log(0) با clamp
    p = preds.data.copy()
    p = np.clip(p, eps, 1.0 - eps)   # numpy-like, یا mytorch.clip اگر موجوده
    # حالا محاسبه‌ی BCE عنصر به عنصر
    log_p = np.log(p)
    log_1_p = np.log(1.0 - p)
    y = label.data.astype(p.dtype)

    loss_elem = -(y * log_p + (1.0 - y) * log_1_p)   # برداری
    # میانگین روی همه‌ی عناصر
    mean_loss = loss_elem.mean()

    return Tensor(np.array(mean_loss)) 