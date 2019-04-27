import numpy as np

class Parameter():
    """
    parameters class
    this module is for variables such as A in NN linear module
    Attributes
    -------------
    val : array-like or float
        value of varaiables
    grad : array-like or float
        gradient of varaiables 
    """
    def __init__(self, val):
        """
        Parameters
        -----------
        val : array-like or float
            value of varaiables
        """
        if not (isinstance(val, list) or isinstance(val, np.ndarray)):
            raise TypeError("the value of parameter should be array-like")

        self.val = val
        self.grad = None

class Module():
    """
    Base class for nn module
    """
    def __init__(self):
        """
        """
        pass
    
    def forward(self, *inputs):
        """
        forward methods for nn module
        you should implement this method in your module
        """
        raise NotImplementedError

    def __call__(self, *inputs):
        """
        """
        output = self.forward(*inputs)
        return output
