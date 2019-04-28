import numpy as np

# original modules
from .NNfunctions import relu, sigmoid
from .NNmodules import GNN, Linear

class VanillaGNN():
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
    def __init__(self):
        """
        """
        self.layers = []

        # make network
        D = 7 # dimension of GNN
        self.fc1 = GNN(D, seed=100)
        self.layers.append(self.fc1)
        self.fc2 = Linear(D, 1)
        self.layers.append(self.fc2)

    def forward(self, x, T=2):
        """
        Parameters
        -------------
        x : array-like, shape(N, in_features) or (in_features)
            ネットワークへの入力、グラフの構造
        T : int, optional
            aggregateする回数, default is 2

        Returns
        ----------
        p : numpy.ndarray, shape(N, 1)
            predicted value, 予測確率
        predict : numpy.ndarray, shape(N, 1)
            predicted label, 予測ラベル（0 or 1）
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
    
        return p, predict