import numpy as np

# original modules
from NNfuntions import softmax, relu

class VanillaGNN():
    """
    Attributes
    ------------


    See also
    -----------
    relu, in layers.py

    """
    def __init__(self):
        """
        """
        self.layers = []

        # make network
        D = 8
        self.fc1 = GNN(D)
        self.layers.append(self.fc1)
        self.fc2 = Linear(D, 1)
        self.layers.append(self.fc2)

        # weight initialize
    
    def forward(self, x, T=2):
        """
        Parameters
        -------------
        x : array-like
            ネットワークへの入力、グラフの構造
        T : int, optional
            aggregateする回数, default is 2

        Returns
        ----------
        p : numpy.ndarray, shape(N, 1)
            予測確率
        predict : numpy.ndarray, shape(N, 1)
            予測ラベル（0 or 1）
        """
        # to numpy
        x = np.array(x)

        # first(GNN)
        hg = self.fc1(x, T)

        # second(Linear)
        s = self.fc2(hg)

        # activation
        p = softmax(s)
        predict = p > 0.5
    
        return p, predict

    def backward(self, output):
        """

        """
        pass 

        return None