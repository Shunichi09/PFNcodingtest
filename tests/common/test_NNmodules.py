import unittest
import numpy as np

# original modules
from src.common.NNfunctions import sigmoid, relu
from src.common.NNmodules import GNN, Linear, BinaryCrossEntropyLossWithSigmoid, GIN

# test GNN
class TestGNN(unittest.TestCase):
    """
    Notes
    --------
    this test class is for 課題1
    """
    def setUp(self):
        pass
        
    def test_initialize(self):
        # check weight initialize
        # test for mismatch D and W dimention
        with self.assertRaises(ValueError): 
            D = 4
            W = np.ones((3, 3))
            GNN(D, W)

    def test_forward(self):
        self.check_condition_forward()
        self.check_forward()
    
    def check_forward(self):
        # check value
        # test for non batch
        D = 1
        W = np.ones((D, D)) * 1.5
        gnn = GNN(D, W)

        input_1_1 = [[0., 1., 0., 0.],
                     [1., 0., 1., 1.],
                     [0., 1., 0., 1.],
                     [0., 1., 1., 0.]] # non batch
        
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

        # test for with batch
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
        # check initialize
        D = 1
        T = 1
        gnn = GNN(D)

        # test for input of type
        input_3 = 0. # float

        with self.assertRaises(TypeError):
            gnn(input_3, T)

        # test for input of shape
        input_4_1 = [[[[0.]]]] # test for too many dim    
        input_4_2 = np.zeros((4, 3, 2, 1)) # test for too many dim in numpy.ndarray
        input_4_3 = [0.] # test for less dim 
        input_4_4 = np.array([0.]) # test for less dim in numpy.ndarray

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
        # test for dim mismatch between A and in features
        in_features = 1
        out_features = 2
        A = np.zeros((2, 1))
        b = np.zeros(3)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)

        # test for dim mismatch between b and out features
        in_features = 2
        out_features = 5
        A = np.zeros((2, 3))
        b = np.zeros(3)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)
                
        # test for dim mismatch between A and b
        in_features = 3
        out_features = 2
        A = np.zeros((3, 3))
        b = np.zeros(2)
        
        with self.assertRaises(ValueError):
            Linear(in_features, out_features, A, b)
    
    def test_forward(self):
        self.check_condition_forward()
        self.check_forward()

    def check_condition_forward(self):
        # test for wrong dim of input
        in_features = 3  
        out_features = 2
        
        input_1_1 = 0.
        input_1_2 = np.zeros((3, 3, 4))

        linear = Linear(in_features, out_features)
        
        with self.assertRaises(ValueError):
            linear(input_1_1)

        with self.assertRaises(ValueError):
            linear(input_1_2)

        # test for dim mismatch between in_features and input
        in_features = 4 
        out_features = 2
        
        input_2_1 = np.zeros((5, 3))

        linear = Linear(in_features, out_features)
        
        with self.assertRaises(ValueError):
            linear(input_2_1)

    def check_forward(self):
        # test for non batch
        A = [[2.], [1.], [0.5]]
        b = [1.]
        
        in_features = 3
        out_features = 1

        linear = Linear(in_features, out_features, A, b)

        input_1_1 = [2., 1., 0.5]  # non batch

        output = linear(input_1_1)
        self.assertTrue(output.shape == (1, out_features)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[6.25]])).all()) # value
        
        # test for batch
        input_2 = [[2., 1., 0.5],
                   [1., 0., -1.]] # batch
        
        output = linear(input_2)
        batch_size = 2
        self.assertTrue(output.shape == (batch_size, out_features)) # shape
        self.assertTrue((np.round(output, 5) ==  np.array([[6.25], [2.5]])).all()) # value


# test CrossEntropyLoss
class TestBinaryCrossEntropyLossWithSigmoid(unittest.TestCase):
    def setUp(self):
        self.loss_fn = BinaryCrossEntropyLossWithSigmoid()

    def test_forward(self):
        self.check_condition_forward()
        self.check_forward() 

    def check_condition_forward(self):
        # test for 3 dim of input
        input_1_1 = np.zeros((1, 4, 3)) 
        target_1_1 = [1, 0, 1, 0]
        with self.assertRaises(ValueError):
            self.loss_fn(input_1_1, target_1_1)

        # test for 2 dim of target
        input_2_1 = np.zeros((4, 3)) 
        target_2_1 = np.array([[1, 0, 1, 0]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_2_1, target_2_1)

        # test for dim mismatch between input and target
        input_3_1 = np.zeros((4, 3)) 
        target_3_1 = np.array([[1, 0, 1]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_3_1, target_3_1)
        
        # test for wrong value in target(binary)
        input_4_1 = np.zeros((4, 3)) 
        target_4_1 = np.array([[4, 0, 1]])
        with self.assertRaises(ValueError):
            self.loss_fn(input_4_1, target_4_1)

    def check_forward(self):
        # test for non batch
        input_1_1 = [3.] 
        target_1_1 = [1]
        loss = self.loss_fn(input_1_1, target_1_1)
        loss_test = self._other_loss_forward(input_1_1, target_1_1)

        self.assertTrue(np.round(loss, 5) == np.round(loss_test, 5)) # checking with other forward
        self.assertTrue(np.round(loss, 5) == np.round(0.0485873, 5)) # checking with hand calculation
        
        # test for batch
        input_2_1 = [[3.], [7.], [1.]] 
        target_2_1 = [1, 0, 0]
        loss = self.loss_fn(input_2_1, target_2_1)
        loss_test = self._other_loss_forward(input_2_1, target_2_1)

        self.assertTrue(np.round(loss, 5) == np.round(loss_test, 5)) # checking with other forward
        self.assertTrue(np.round(loss, 5) == np.round(2.787586, 5)) # checking with hand calculation
        
        # test for over flow
        input_3_1 = [1.e5]
        target_3_1 = [0]
        loss = self.loss_fn(input_3_1, target_3_1)

        with self.assertRaises(FloatingPointError): # test for over flow check of other method
            loss_test = self._other_loss_forward(input_3_1, target_3_1)

        self.assertTrue(np.round(loss, 5) == np.round(1.e5, 5)) # test for being able to calculate the approximate loss

    def _other_loss_forward(self, x, target):
        """sigmoid --> binary cross entropy loss
        Notes
        ------
        - sigmoidを通して通常のbinary cross entropy lossを算出しています
        """
        x = np.array(x)
        target = np.array(target)
        N = target.shape[0]

        x = sigmoid(x)

        if x.ndim < 2:
            x = x[np.newaxis, :].astype(float)
        
        target = target[:, np.newaxis].astype(float)

        with np.errstate(all='raise'):
            loss = -target * np.log(x) - (1. - target) * np.log(1. - x)

        output = np.sum(loss) / N

        return output

class TestGIN(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_initialize(self):
        # check weight initialize
        # test for mismatch D and W dimention
        with self.assertRaises(ValueError): 
            D = 4
            W1 = np.ones((3, 3))
            GIN(D, W1=W1)
        
        with self.assertRaises(ValueError): 
            D = 4
            W2 = np.ones((3, 3))
            GIN(D, W2=W2)
        
        with self.assertRaises(ValueError): 
            D = 4
            b1 = np.ones((3, 1))
            GIN(D, b1=b1)

        with self.assertRaises(ValueError): 
            D = 4
            b2 = np.ones((3, 1))
            GIN(D, b2=b2)

    def test_forward(self):
        self.check_condition_forward()
        self.check_forward()
    
    def check_forward(self):
        # check value
        # test for non batch
        D = 2
        W1 = np.ones((D, D)) * 1.5
        W2 = np.eye((D)) # eye
        b1 = np.arange(D).reshape((D, 1))
        b2 = np.zeros(D).reshape((D, 1)) # zero

        gin = GIN(D, W1=W1, W2=W2, b1=b1, b2=b2)

        input_1 = [[0., 1., 0.],
                     [1., 0., 1.],
                     [0., 1., 0.]] # non batch
        
        T = 1
        output = gin(input_1, T)
        self.assertTrue(output.shape == (1, D)) # shape

        test_output = relu(relu(np.array([[[1.5, 3., 1.5], [2.5, 4., 2.5]]]))) # hand calculation
        test_output = np.sum(test_output, axis=-1)

        self.assertTrue((np.round(output, 5) == np.round(test_output, 5)).all()) # value
        
        # test for with batch
        D = 2
        batch_size = 2
        W1 = np.ones((D, D)) * 1.5
        W2 = np.eye((D)) # eye
        b1 = np.arange(D).reshape((D, 1))
        b2 = np.zeros(D).reshape((D, 1)) # zero

        gin = GIN(D, W1=W1, W2=W2, b1=b1, b2=b2)
        
        input_2 = [[[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]], 
                   [[0., 1., 0.],
                    [1., 0., 1.],
                    [0., 1., 0.]]] # batch

        T = 1
        output = gin(input_2, T)
        self.assertTrue(output.shape == (batch_size, D)) # shape

        array_1 = np.array([[1.5, 4.5, 3., 3.],
                            [2.5, 5.5, 4., 4.]])
        
        array_2 = np.array([[1.5, 3., 1.5, 0.],
                            [2.5, 4., 2.5, 0.]])

        test_output = np.array([array_1, array_2])

        test_output = relu(relu(test_output)) # hand calculation
        test_output = np.sum(test_output, axis=-1)

        self.assertTrue((np.round(output, 5) == np.round(test_output, 5)).all()) # value
        

    def check_condition_forward(self):
        # check initialize
        D = 1
        T = 1
        gnn = GIN(D)

        # test for input of type
        input_3 = 0. # float

        with self.assertRaises(TypeError):
            gnn(input_3, T)

        # test for input of shape
        input_4_1 = [[[[0.]]]] # test for too many dim    
        input_4_2 = np.zeros((4, 3, 2, 1)) # test for too many dim in numpy.ndarray
        input_4_3 = [0.] # test for less dim 
        input_4_4 = np.array([0.]) # test for less dim in numpy.ndarray

        with self.assertRaises(ValueError):
            gnn(input_4_1, T)

        with self.assertRaises(ValueError):
            gnn(input_4_2, T)

        with self.assertRaises(ValueError):
            gnn(input_4_3, T)

        with self.assertRaises(ValueError):    
            gnn(input_4_4, T)