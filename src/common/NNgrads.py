import numpy as np
from collections import OrderedDict

# original
from .NNbases import Parameter

def calc_numerical_gradient(f, x):
    """
    Parameters
    ------------
    f : function
        forward function of NN
    x : numpy.ndarray
        input
    
    Returns
    ---------
    grad : numpy.ndarray, shape is the same as x
        results of numercial gradient of the input

    References
    -----------
    - oreilly japan 0 から作るdeeplearning
    https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/gradient.py
    """
    # check condition
    if not callable(f):
        raise TypeError("f should be callable")

    if not (isinstance(x, list) or isinstance(x, np.ndarray)):
        raise TypeError("x should be array-like")

    h = 1e-3 # 0.001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return np.array(grad)

def numerical_gradient(parameters, forward_fn):
    """ calculated the gradients of parameters, the gradients are placed in each Parameter class
    Parameters
    -------------
    parameters : OrderedDict
        Ordered dictionary of Parameter class
    forward_fn : function
        function of NN's forward
    """
    if not isinstance(parameters, OrderedDict):
        raise TypeError("parameters should be OrderedDict of Parameter class")

    if not callable(forward_fn):
        raise TypeError("foward_fn should be callable")

    for _, param in parameters.items():
        grad = calc_numerical_gradient(forward_fn, param.val)
        param.grad = grad.copy()
