import numpy as np

# original
from .NNbase import Parameter

def calc_numerical_gradient(f, x):
    """
    Parameters
    ------------

    References
    -----------
    - oreilly japan 0 から作るdeeplearning
    https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/gradient.py
    """
    h = 1e-4 # 0.0001
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
        
    return grad

def numercial_grad(parameters, forward_fn):
    """ calculated the gradients of parameters, the gradients are placed in each Parameter class
    Parameters
    -------------
    parameters : Orderedict
        dictionary of Parameter class
    forward_fn : function
    loss_fn : function
    """
    for param in parameters:
        print("name = {}".format(param.key()))



if __name__ == "__main__":
    

