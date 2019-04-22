import numpy as np

# original modules
from .NNfunctions import sigmoid, relu

class Module():
    """
    Base class for nn module
    NN の moduleのベースクラス
    """
    def __init__(self):
        """
        """
        pass
    
    def forward(self, *inputs):
        """
        継承した先でforward methodを実装してください
        """
        raise NotImplementedError

    def __call__(self, *inputs):
        """
        """
        output = self.forward(*inputs)
        return output

class GNN(Module):
    """graph neural network

    Attributes
    -------------
    W : numpy.ndarray, shape(D, D)
        weights of graph, GNNの重み
    
    Example
    ---------
    >>> gnn = GNN(3)
    >>> input = [[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]] # non batch
    >>> output = gnn(input, 1)
    >>> output.shape # 1 * D
    (1, 3)

    >>> gnn = GNN(3)
    >>> input = [[[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]], 
                    [[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]]] # batch

    >>> output = gnn(input, 2)
    >>> output.shape
    (2, 3) # batch_size * D
    """
    def __init__(self, D, W=None):
        """
        Parameters
        -----------
        D : int
            size of dimention of the weights, 重みWの次元
        """
        super().__init__()
        self.W = W
        if W is None:
            self.W = np.random.normal(loc=0, scale=0.4, size=(D, D))

        if self.W.shape[1] != D:
            raise ValueError("dimention of weight and dimesion are not equal")
    
    def forward(self, x, T):
        """ Forward propagation
        Parameters
        --------------
        x : array-like, shape(num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
            グラフの隣接行列、グラフのノード数×グラフのノード数の対称行列
        T : int
            aggregateする回数

        Returns
        ------------
        output : numpy.ndarray, shape(1, D) or shape(batch_size, D)
            xが2次元の場合、batch_sizeを1としてoutputを返します
        """
        # save initial conditions
        D = self.W.shape[0]

        x = np.array(x)
        x_shape = x.shape

        if len(x_shape) <= 2: # 2d to 3d
            x = x[np.newaxis, :, :]

        # initialize states
        states = np.array([np.zeros((D, single_x.shape[1])) for single_x in x]) # make each state of Graph, states has (batch_size, D, nodes_num)
        states[:, 0, :] = np.ones_like(states[:, 0, :])
        # print("init state = \n{}".format(states))

        for _ in range(T):
            a = np.matmul(states, x)
            # print("middle = \n{}".format(a))
            states = relu(np.matmul(self.W, a))
        
        # print("state = \n{}".format(states))
        output = np.sum(states, axis=-1)
        # print("output = \n{}".format(output))

        return output

class Linear(Module):
    """ Linear layer
    Attributes
    -----------
    A : numpy.ndarray, shape(D,)
        weights of the linear layer
    b : numpy.ndarray, shape(1,)
        bias of the linear layer
    """
    def __init__(self, in_features, out_features):
        """
        Parameters
        -----------
        in_features : int
            size of input
        out_features: int
            size of output
        """
        super().__init__()
        self.A = np.random.normal(loc=0, scale=0.4, size=(in_features, out_features))
        self.b = np.zeros(1)

    def forward(self, x):
        """ Forward propagation
        Parameters
        -----------
        x : array-like, shape(D, )
        Returns
        -------
        output : numpy.ndarray, shape(1)
        """
        x = np.array(x)
        output = np.dot(self.A, x) + self.b

        return output

class CrossEntropyLoss(Module):
    """ Cross entropy loss function
    """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x, target):
        """
        Parameters
        -------------
        x : array-like, shape()
            output of NN, NNの出力
        target : array-like, shape()
            target of data, 教師信号
        
        Returns
        ------------
        output : numpy.ndarray()
        """
        x = np.array(x)
        target = np.array(target)

        if not x.shape[0] == target.shape[0]:
            raise ValueError("y and target should have same row size")

        try:
            loss = target * np.log(1. + np.exp(-x)) + (1 - target) * np.log(1 + np.exp(x))
        except OverflowError:
            loss = target * np.log(1. + np.exp(-x)) + (1 - target) * x

        output = loss.copy()

        return output

if __name__ == "__main__":

    D = 4
    gnn = GNN(D)

    x = [[0., 1., 0., 0.],
         [1., 0., 1., 1.],
         [0., 1., 0., 1.],
         [0., 1., 1., 0.]]

    T = 2

    output = gnn(x, T)

    print(output)