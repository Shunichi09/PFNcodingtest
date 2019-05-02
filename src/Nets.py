import numpy as np

# original modules
from .common.NNbases import Module
from .common.NNfunctions import relu, sigmoid
from .common.NNmodules import GNN, Linear, BinaryCrossEntropyLossWithSigmoid, GIN

class VanillaGNN(Module):
    """ vanilla gnn
    Attributes
    ------------
    layers : list
        list of layers
    fc1 : GNN class
        1st layer
    fc2 : Linear class
        2nd layer

    See also
    -----------
    GNN, Linear in NNmodules.py
    """
    def __init__(self, seed=None):
        """
        Parameters
        -----------
        seed : int, optional
            seed of random state, default is None
        """
        super(VanillaGNN, self).__init__()
        self.layers = []

        # make network
        D = 8 # dimension of GNN
        self.fc1 = GNN(D, seed=seed)
        self.layers.append(self.fc1)
        self.fc2 = Linear(D, 1, seed=seed)
        self.layers.append(self.fc2)

        self.register_parameters(self.layers)
        
    def forward(self, x, T=2):
        """
        Parameters
        -------------
        x : array-like, shape(N, in_features) or (in_features)
            input of NN (Graph)
        T : int, optional
            times of aggregate, default is 2

        Returns
        ----------
        p : numpy.ndarray, shape(N, 1)
            predicted value
        predict : numpy.ndarray, shape(N, 1)
            predicted label,（0 or 1）
        s : numpy.ndarray, shape(N, 1)
            state before the activation layer
        """
        # to numpy
        x = np.array(x)

        # first(GNN)
        hg = self.fc1(x, T)

        # second(Linear)
        s = self.fc2(hg)

        # activation
        p = sigmoid(s)
        predict = p > 0.5
    
        return p, predict, s

class VanillaGIN(Module):
    """ vanilla gin
    Attributes
    ------------
    layers : list
        list of layers
    fc1 : GIN class
        1st layer
    fc2 : Linear class
        2nd layer

    See also
    -----------
    GIN, Linear in NNmodules.py
    """
    def __init__(self, seed=None):
        """
        Parameters
        -----------
        seed : int, optional
            seed of random state, default is None
        """
        super(VanillaGIN, self).__init__()
        self.layers = []

        # make network
        D = 8 # dimension of GNN
        self.fc1 = GIN(D, seed=seed)
        self.layers.append(self.fc1)
        self.fc2 = Linear(D, 1, seed=seed)
        self.layers.append(self.fc2)

        self.register_parameters(self.layers)
        
    def forward(self, x, T=2):
        """
        Parameters
        -------------
        x : array-like, shape(N, in_features) or (in_features)
            input of NN (Graph)
        T : int, optional
            times of aggregate, default is 1

        Returns
        ----------
        p : numpy.ndarray, shape(N, 1)
            predicted value
        predict : numpy.ndarray, shape(N, 1)
            predicted label,（0 or 1）
        s : numpy.ndarray, shape(N, 1)
            state before the activation layer
        """
        # to numpy
        x = np.array(x)

        # first(GNN)
        hg = self.fc1(x, T)

        # second(Linear)
        s = self.fc2(hg)

        # activation
        p = sigmoid(s)
        predict = p > 0.5
    
        return p, predict, s
