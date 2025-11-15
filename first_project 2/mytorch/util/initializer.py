import numpy as np

def xavier_initializer(shape):
    "TODO: implement xavier_initializer" 
    # Uniform xavier
    input, output = shape[0], shape[1]
    limit = np.sqrt(6 / (input + output))
    return np.random.uniform(-limit, limit, size=shape)

    # Normal xavier
    # std = np.sqrt(2 / (shape[0] + shape[1]))
    # return np.random.normal(0, std, shape)
    # return ...

def he_initializer(shape):
    "TODO: implement he_initializer" 
    std = np.sqrt(2 / shape[0])
    return np.random.randn(*shape) * std
    # return ...

def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    "TODO: implement random_normal_initializer" 
    return np.random.randn(*shape) * stddev + mean
    # return ...

def zero_initializer(shape):
    "TODO: implement zero_initializer" 
    return np.zeros(shape, dtype=np.float64)
    # return ...

def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)

def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
