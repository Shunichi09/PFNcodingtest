import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ activation function of NN, sigmoid
    Parameters
    ------------
    x : array-like or float

    Returns
    ---------
    output : numpy.ndarray, shape is the same as x
    """
    # check condition
    if not (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, float)):
        raise TypeError("x should be array-like or float")

    x = np.array(x)

    output = 1. / (1. + np.exp(-x))

    return output

def relu(x):
    """ activation function of NN, relu
    Parameters
    ------------
    x : array-like or float

    Returns
    ---------
    output : numpy.ndarray, shape is the same as x
    """
    # check condition
    if not (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, float)):
        raise TypeError("x should be array-like or float")

    x = np.array(x)

    output = x.copy()
    mask = (x <= 0)
    output[mask] = 0

    return output

if __name__ == "__main__":
    x = np.arange(-6, 6, 0.1)
    y1 = sigmoid(x)
    y2 = relu(x)

    plt.plot(x, y1)
    plt.plot(x, y2)

    plt.show()