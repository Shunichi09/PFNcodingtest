import numpy as np
from collections import OrderedDict

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
        self.parameters = OrderedDict()
    
    def forward(self, *inputs):
        """
        forward methods for nn module
        you should implement this method in your module
        """
        raise NotImplementedError

    def __call__(self, *inputs):
        """call forward method
        """
        output = self.forward(*inputs)
        return output
    
    def get_parameters(self):
        """return the parameters
        Returns
        ----------
        parameters : OrderedDict
        """
        return self.parameters

    def register_parameters(self, parameters):
        """register the parameters
        Parameters
        ------------
        parameters : Moudle or Parameter class or list of them
        """
        # check conditions
        if isinstance(parameters, list):
            if not (isinstance(parameters[0], Module) or isinstance(parameters[0], Parameter)):
                raise TypeError("the parameters should be Module or Parameter class, or list of them")
        else:
            if not (isinstance(parameters, Module) or isinstance(parameters, Parameter)):
                raise TypeError("the parameters should be Module or Parameter class, or list of them")            

        # to list
        if isinstance(parameters, Module) or isinstance(parameters, Parameter): # if the parameters is not list
            parameters = [parameters] # to list

        # register the parameters
        if isinstance(parameters[0], Module): # if the parameters is modules
            for num_module, module in enumerate(parameters):
                for key, param in module.get_parameters().items():
                    self.parameters['layer_{}-'.format(num_module) + key] = param
                
        elif isinstance(parameters[0], Parameter): # if the parameters is Parameter
            for num_param, param in enumerate(parameters):
                self.parameters['param_{}'.format(num_param)] = param