import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np

# original modules
from src.common.NNfunctions import sigmoid, relu
from src.common.NNmodules import Module, GNN, Linear, BinaryCrossEntropyLossWithSigmoid

# test Module
class TestModule(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_forward(self):
        # forwardのエラーがでるか
        test_input = [None]
        with self.assertRaises(NotImplementedError):
            self.module.forward(test_input)

    def test_call(self): 
        # forwardが呼ばれているか
        test_input = [None]

        self.module.forward = MagicMock(return_value="forward_propagation")
        self.assertEqual(self.module(test_input), "forward_propagation")
        self.module.forward.assert_called_once() # 一回かどうか

# test GNN
class TestGNN(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_initialize(self):
        # weightのエラーがでるか
        with self.assertRaises(ValueError):
            D = 4
            W = np.ones((3, 3))
            GNN(D, W)

    def test_forward(self):
        self.check_condition_forward()
        self.check_forward()
    
    def check_forward(self):
        """値の計算があっているか
        """
        # case input_1 with non batch
        D = 1
        W = np.ones((D, D)) * 1.5
        gnn = GNN(D, W)

        input_1_1 = [[0., 1., 0., 0.],
                     [1., 0., 1., 1.],
                     [0., 1., 0., 1.],
                     [0., 1., 1., 0.]] # non batch in numpy ndarray
        
        T = 1
        output = gnn(input_1_1, T)
        self.assertTrue(output.shape == (1, D)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[12.]])).all()) # value
        T = 2
        output = gnn(input_1_1, T)
        self.assertTrue((np.round(output, 5) ==  np.array([[40.5]])).all()) # value
        
        input_1_2 = np.array([[0., 1., 0., 0.],
                              [1., 0., 1., 1.],
                              [0., 1., 0., 1.],
                              [0., 1., 1., 0.]]) # non batch in numpy.ndarray

        T = 1
        output = gnn(input_1_2, T)
        self.assertTrue(output.shape == (1, D)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[12.]])).all()) # value
        T = 2
        output = gnn(input_1_2, T)
        self.assertTrue((np.round(output, 5) ==  np.array([[40.5]])).all()) # value

        # case input_2 with batch
        D = 2
        batch_size = 2
        W = np.array([[0.5, -1.],
                      [-0.5, 0.1]])
        gnn = GNN(D, W)

        input_2_1 = [[[0., 1., 0., 0.],
                      [1., 0., 1., 1.],
                      [0., 1., 0., 1.],
                      [0., 1., 1., 0.]], 
                     [[0., 1., 0.],
                      [1., 0., 1.],
                      [0., 1., 0.]]] # batch

        T = 1
        output = gnn(input_2_1, T)
        self.assertTrue(output.shape == (batch_size, D)) # shape
        self.assertTrue((np.round(output, 5) == np.array([[4., 0.], [2., 0.]])).all()) # value

        T = 2
        output = gnn(input_2_1, T)
        self.assertTrue((np.round(output, 5) == np.array([[4.5, 0.], [ 1.5, 0.]])).all()) # value

        input_2_2 = np.array([[[0., 1., 0., 0.],
                               [1., 0., 1., 1.],
                               [0., 1., 0., 1.],
                               [0., 1., 1., 0.]], 
                              [[0., 1., 0.],
                               [1., 0., 1.],
                               [0., 1., 0.]]]) # batch in numpy.ndarray

        T = 1
        output = gnn(input_2_2, T)
        self.assertTrue(output.shape == (batch_size, D)) # shape
        self.assertTrue((np.round(output, 5) == np.array([[4., 0.], [2., 0.]])).all()) # value

        T = 2
        output = gnn(input_2_2, T)
        self.assertTrue((np.round(output, 5) == np.array([[4.5, 0.], [ 1.5, 0.]])).all()) # value

    def check_condition_forward(self):
        """入力の型があっているか
        """
        D = 1
        T = 1
        gnn = GNN(D)

        # case input_3
        input_3 = 0. # float

        with self.assertRaises(TypeError):
            gnn(input_3, T)

        # case input_4
        input_4_1 = [[[[0.]]]] # too many dim    
        input_4_2 = np.zeros((4, 3, 2, 1)) # too many dim in numpy.ndarray
        input_4_3 = [0.] # less dim 
        input_4_4 = np.array([0.]) # less dim in numpy.ndarray

        with self.assertRaises(ValueError):
            gnn(input_4_1, T)

        with self.assertRaises(ValueError):
            gnn(input_4_2, T)

        with self.assertRaises(ValueError):
            gnn(input_4_3, T)

        with self.assertRaises(ValueError):    
            gnn(input_4_4, T)

# test Linear layer
class TestLinear(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        """
        """
        # case 1
        in_features = 2
        out_features = 5
        A = np.zeros((2, 3))
        b = np.zeros(3)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)
        
        # case 2
        in_features = 1
        out_features = 2
        A = np.zeros((2, 1))
        b = np.zeros(3)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)

        # case 3
        in_features = 3
        out_features = 2
        A = np.zeros((1, 3))
        b = np.zeros(2)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)
    
    def test_forward(self):
        """
        """
        self.check_condition_forward()
        self.check_forward()

    def check_forward(self):
        """
        """
        # case 1 with non batch
        A = [[2., 1., 0.5]]
        b = [1.]
        
        in_features = 3
        out_features = 1

        linear = Linear(in_features, out_features, A, b)

        input_1_1 = [2., 1., 0.5]  # non batch

        output = linear(input_1_1)
        self.assertTrue(output.shape == (1, out_features)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[6.25]])).all()) # value
        

        # case 2 with batch
        input_2 = [[2., 1., 0.5],
                   [1., 0., -1.]] # batch
        
        output = linear(input_2)
        batch_size = 2
        self.assertTrue(output.shape == (batch_size, out_features)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[6.25], [2.5]])).all()) # value

    def check_condition_forward(self):
        """
        """
        # case 3
        in_features = 3
        out_features = 2
        
        input_3_1 = 0.
        input_3_2 = np.zeros((3, 3, 4))

        linear = Linear(in_features, out_features)
        
        with self.assertRaises(ValueError):
            linear(input_3_1)

        with self.assertRaises(ValueError):
            linear(input_3_2)

        # case 4
        in_features = 4
        out_features = 2
        
        input_4_1 = np.zeros((5, 3))

        linear = Linear(in_features, out_features)
        
        with self.assertRaises(ValueError):
            linear(input_4_1)

# test CrossEntropyLoss
class TestBinaryCrossEntropyLossWithSigmoid(unittest.TestCase):
    def setUp(self):
        """
        """
        self.loss_fn = BinaryCrossEntropyLossWithSigmoid()

    def test_forward(self):
        """
        """
        self.check_condition_forward()
        self.check_forward() 

    def check_forward(self):
        """
        """
        # case 1 with no batch
        input_1_1 = [3.]
        target_1_1 = [1]
        loss = self.loss_fn(input_1_1, target_1_1)
        loss_test = self._other_loss_forward(input_1_1, target_1_1)

        self.assertTrue(np.round(loss, 5) == np.round(loss_test, 5))
        self.assertTrue(np.round(loss, 5) == np.round(0.0485873, 5))
        
        # case 2 with batch
        input_2_1 = [[3.], [7.], [1.]]
        target_2_1 = [1, 0, 0]
        loss = self.loss_fn(input_2_1, target_2_1)
        loss_test = self._other_loss_forward(input_2_1, target_2_1)

        self.assertTrue(np.round(loss, 5) == np.round(loss_test, 5))
        self.assertTrue(np.round(loss, 5) == np.round(2.787586, 5))
        
        # case 3 over flow
        input_3_1 = [1.e5]
        target_3_1 = [0]
        loss = self.loss_fn(input_3_1, target_3_1)

        with self.assertRaises(FloatingPointError): # over flow check
            loss_test = self._other_loss_forward(input_3_1, target_3_1)

        self.assertTrue(np.round(loss, 5) == np.round(1.e5, 5))

    def _other_loss_forward(self, x, target):
        """別の方法でのbinary cross entropy lossを算出
        sigmoid --> binary cross entropy loss
        """
        x = np.array(x)
        target = np.array(target)
        N = target.shape[0]

        x = sigmoid(x)

        # print("x = \n {}".format(x))

        if x.ndim < 2:
            x = x[np.newaxis, :].astype(float)
        
        target = target[:, np.newaxis].astype(float)

        with np.errstate(all='raise'):
            loss = -target * np.log(x) - (1. - target) * np.log(1. - x)

        output = np.sum(loss) / N

        # print("output = {}".format(output))
        
        return output

    def check_condition_forward(self):
        """
        """
        # case 4
        input_4_1 = np.zeros((1, 4, 3))
        target_4_1 = [1, 0, 1, 0]
        with self.assertRaises(ValueError):
            self.loss_fn(input_4_1, target_4_1)
        
        # case 5
        input_5_1 = np.zeros((4, 3))
        target_5_1 = np.array([[1, 0, 1, 0]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_5_1, target_5_1)

        # case 6
        input_6_1 = np.zeros((4, 3))
        target_6_1 = np.array([[1, 0, 1]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_6_1, target_6_1)

        # case 7
        input_7_1 = np.zeros((4, 3))
        target_7_1 = np.array([[4, 0, 1]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_7_1, target_7_1)