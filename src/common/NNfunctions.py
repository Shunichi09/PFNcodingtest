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
    x = np.array(x)

    output = x.copy()
    mask = (x <= 0)
    output[mask] = 0

    return output

if __name__ == "__main__":
    x = np.arange(-6, 6, 0.1)
    y = sigmoid(x)
    y = relu(x)

    plt.plot(x, y)
    plt.show()