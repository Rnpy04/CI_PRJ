from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    mse = (preds - actual)**2
    # #یک اسکالر میانگین کلی
    # loss = mse.sum()   # جمع همه المان‌ها
    # n = np.prod(preds.shape)   # یا preds.numel() اگر فراهمه؛ بهتر اینکه از Tensor shape استفاده کنی
    # return loss * (1.0 / n)
    
    # #علی
    size = Tensor(np.array([mse.data.size],dtype=np.float64))
    size = size**-1
    return mse * size
    
    #اگه بخوایم برای هر نمونه بدست بیاریم batch-wise
    # per_sample = mse.sum(axis=1) / preds.shape[1]   # sum روی outputs، تقسیم بر تعداد خروجی‌ها
    # return per_sample.mean()
    
    # return ...
