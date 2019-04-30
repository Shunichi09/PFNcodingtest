import warnings
import numpy as np

# original modules
from .NNfunctions import sigmoid, relu
from .NNbase import Parameter, Module

class GNN(Module):
    """graph neural network

    Attributes
    -------------
    W : Parameter class, shape(D, D)
        weights of graph, GNNの重み
    
    See also
    ------------
    Parameter in NNbase.py
    Module in NNbase.py
    
    Example
    ---------
    >>> gnn = GNN(3)
    >>> input = [[0., 1., 0., 0.],
                 [1., 0., 1., 1.],
                 [0., 1., 0., 1.],
                 [0., 1., 1., 0.]] # non batch
    >>> output = gnn(input, 1)
    >>> gnn.W
    [[0.135401]]
    >>> output
    [[1.08320799]]
    >>> output.shape # 1 * D
    (1, 3)

    >>> gnn = GNN(3)
    >>> input_2_1 = [[[0., 1., 0., 0.],
                      [1., 0., 1., 1.],
                      [0., 1., 0., 1.],
                      [0., 1., 1., 0.]], 
                     [[0., 1., 0.],
                      [1., 0., 1.],
                      [0., 1., 0.]]] # batch
    >>> output = gnn(input, 2)
    >>> output.shape
    (2, 3) # N(batch_size) * D
    """
    def __init__(self, D, W=None, seed=None):
        """
        Parameters
        -----------
        D : int
            size of dimention of the weights, 重みWの次元
        W :  array-like, shape(D, D), optional
            weights of graph, GNNの重み, default is None
        seed : int, optional
            seed of random state, default is None 
        """
        super(GNN, self).__init__()

        self.W = Parameter(np.array(W))

        if W is None:
            np.random.seed(seed)
            self.W = Parameter(np.random.normal(loc=0, scale=0.4, size=(D, D)))
        
        if self.W.val.shape != (D, D):
            raise ValueError("dimention of weight and dimesion are not equal")

        self.register_parameters(self.W)
    
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
        
        Notes
        --------
        - xがbatchを持つ、3dの場合、0埋めでpaddingを行い、計算
        """
        self._check_condition(x, T)

        # save initial conditions
        D = self.W.val.shape[0] # dimension        
        x = np.array(x) 
        if x.ndim == 2: # 2d to 3d
            x = x[np.newaxis, :, :]
        
        N = len(x) # batch_size
        nums_node = [np.array(single_x).shape[1] for single_x in x] # get each number of nodes
        
        # initialize states and padding the input
        states = np.zeros((N, D, max(nums_node)))
        pad_x = np.zeros((N, max(nums_node), max(nums_node)))

        for i, num_node in enumerate(nums_node): # padding
            # x padding
            pad_x[i, :num_node, :num_node] = np.array(x[i], dtype=float)
            # initialize states
            states[i, 0, :num_node] = np.ones(num_node)

        for _ in range(T): # aggregate
            a = np.matmul(states, pad_x)
            states = relu(np.matmul(self.W.val, a))
        
        output = np.sum(states, axis=-1)

        return output

    def _check_condition(self, x, T):
        """check the parameters
        """
        # check condition
        # type
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            raise TypeError("x should be array-like")
        
        # shape
        if not (isinstance(x[0], list) or isinstance(x[0], np.ndarray)):
            raise ValueError("x shape should be 2dim or 3dim")
        
        if isinstance(x[0][0], list) or isinstance(x[0][0], np.ndarray):
            if isinstance(x[0][0][0], list) or isinstance(x[0][0][0], np.ndarray):
                raise ValueError("x shape should be 2dim or 3dim")

class Linear(Module):
    """ Linear layer, y = xA + b
    Attributes
    -----------
    A : Parameter class, shape(in_features, out_features)
        weights of the linear layer
    b : Parameter class, shape(out_features, )
        bias of the linear layer
    
    See also
    ------------
    Parameter in NNbase.py
    Module in NNbase.py
    
    Example
    ----------
    >>> linear = Linear(3, 1)
    >>> inputs = [2., 1., 0.5]  # non batch
    >>> output = linear(inputs)
    >>> linear.A, linear.b
    A
    [[-0.27587793  0.07364357 -0.44339573]]
    b
    [0.]
    >>> output
    [[-0.69981015]]
    >>> output.shape
    (1, 1) # 1 * out_features

    >>> linear = Linear(3, 1)
    >>> inputs = [[2., 1., 0.5],
                  [1., 0., -1.]] # batch
    >>> output = linear(inputs)
    >>> output.shape
    (2, 1) # N(batch_size) * out_features
    """
    def __init__(self, in_features, out_features, A=None, b=None, seed=None):
        """
        Parameters
        -----------
        in_features : int
            size of input
        out_features: int
            size of output
        A : array-like, shape(in_features, out_features)
            weights of the linear layer, default is None
        b : array-like, shape(out_features, )
            bias of the linear layer, default is None
        seed : int, optional
            seed of random state, default is None 
        """
        super(Linear, self).__init__()
        self.A = Parameter(np.array(A))
        self.b = Parameter(np.array(b))
        
        if A is None:
            np.random.seed(seed)
            self.A = Parameter(np.random.normal(loc=0, scale=0.4, size=(in_features, out_features)))
        if b is None:
            self.b = Parameter(np.zeros(out_features))
        
        # check shape         
        if not self.A.val.shape[0] == in_features:
            raise ValueError("row size of A should have same size as in_features")
        
        if not self.b.val.shape[0] == out_features:
            raise ValueError("b should have same size as out_features")
        
        if not self.A.val.shape[1] == self.b.val.shape[0]:
            raise ValueError("row size of A and b should have same size")

        self.register_parameters([self.A, self.b])        

    def forward(self, x):
        """ Forward propagation
        Parameters
        -----------
        x : array-like, shape(in_features) or (N, in_features)
        Returns
        -------
        output : numpy.ndarray, shape(1, out_features) or (N, out_features)
        """
        self._check_condition(x)

        x = np.array(x)

        if x.ndim < 2:
            x = x[np.newaxis, :]

        output = np.matmul(x, self.A.val) + self.b.val

        return output
    
    def _check_condition(self, x):
        """check the parameters
        """
        if np.array(x).ndim == 0 or np.array(x).ndim > 2:
            raise ValueError("x shape should be (in_features,) or (N, in_features)")

        if not np.array(x).shape[-1] == self.A.val.shape[0]:
            raise ValueError("x shape should be (in_features,) or (N, in_features)")

class BinaryCrossEntropyLossWithSigmoid(Module):
    """ Cross entropy loss function
    """
    def __init__(self):
        """
        """
        super(BinaryCrossEntropyLossWithSigmoid, self).__init__()

    def forward(self, x, target):
        """
        Parameters
        -------------
        x : array-like, shape(in_feartures) or (N, in_features)
            output of NN, NNの出力
        target : array-like, shape(N)
            target of data, 教師信号
        
        Returns
        ------------
        output : float
            loss value, 評価関数の値(バッチサイズで平均をとったもの)     
        """
        self._check_condition(x, target)

        x = np.array(x)
        target = np.array(target)
        N = target.shape[0]

        if x.ndim < 2:
            x = x[np.newaxis, :].astype(float)
        
        target = target[:, np.newaxis].astype(float)

        try:
            with np.errstate(all='raise'): # raise FloatingPointError
                loss = target * np.log(1. + np.exp(-x)) + (1 - target) * np.log(1 + np.exp(x))
        except FloatingPointError:
                loss = target * np.log(1. + np.exp(-x)) + (1 - target) * x

        output = np.sum(loss) / N

        return output

    def _check_condition(self, x, target):
        """
        """
        x = np.array(x)
        target = np.array(target)

        if x.ndim > 2:
            raise ValueError("x should have 1d or 2d")
        
        if target.ndim > 1:
            raise ValueError("target should have 1d")

        if x.ndim < 2:
            x = x[np.newaxis, :]

        if not x.shape[0] == target.shape[0]:
            raise ValueError("x and target should have same row size")

        if (target > 1).any():
            raise ValueError("target should only have 1 or 0")