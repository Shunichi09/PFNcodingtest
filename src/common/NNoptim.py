import numpy as np
from collections import OrderedDict
from .NNbase import Parameter

class Optimizer():
    """
    Attributes
    -----------
    parameters : OrderedDict
        OrderedDict of Parameter class
    """
    def __init__(self, param):
        """
        Parameters
        ----------
        param : OrderedDict
            OrderedDict of network's parameters     
        """
        # check condition
        if not isinstance(param, OrderedDict):
            raise TypeError("param should be OrderedDict of Parameter class")

        self.parameters = param

    def step(self):
        """
        you should implement in your optimizer
        """
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, param, alpha=0.0001):
        """
        Paramters
        -----------
        param : OrderedDict
            OrderedDict of network's parameters
        alpha : float, optional
            learning rate, default is 0.0001
        """
        super(SGD, self).__init__(param)

        # check condition
        if not isinstance(alpha, float):
            raise ValueError("alpha should be positive float")

        if alpha < 0.:
            raise ValueError("alpha should be positive float")

        self.alpha = alpha

    def step(self):
        """update the parameters
        """
        for _, param in self.parameters.items():
            param.val += -self.alpha * param.grad

class MomentumSGD(Optimizer):
    def __init__(self, param, alpha=0.0001, beta=0.9):
        """
        Paramters
        -----------
        param : OrderedDict
            OrderedDict of network's parameters
        alpha : float, optional
            learning rate, default is 0.0001
        beta : float, optional
            momentum rate, default is 0.9
        """
        super(MomentumSGD, self).__init__(param)
        self.alpha = alpha
        self.beta = beta
        self.pre_param = None

    def step(self):
        """update the parameters
        """
        # make pre param
        if self.pre_param is None: 
            self.pre_param = OrderedDict()
            for key, param in self.parameters.items():
                self.pre_param[key] = Parameter(np.zeros_like(param.val))
                self.pre_param[key].grad = param.grad.copy()

        # update
        for key, param in self.parameters.items():
            param.val += -self.alpha * param.grad + self.beta * self.pre_param[key].grad
            self.pre_param[key].grad = -self.alpha * param.grad + self.beta * self.pre_param[key].grad

class Adam(Optimizer):
    def __init__(self, param):
        """
        """
        pass

