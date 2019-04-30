import numpy as np
from collections import OrderedDict
import unittest
from unittest.mock import patch

# original
from src.common.NNgrads import calc_numerical_gradient, numerical_gradient
from src.common.NNbases import Parameter
from src.common.NNmodules import Linear

class TestCalcNumercialGradient(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_calc_numerical_gradient(self):
        # test for type(x)
        input_1_weight = 3.
        input_1_fn = lambda weight: None

        with self.assertRaises(TypeError): 
            calc_numerical_gradient(input_1_fn, input_1_weight)

        # test for type(f)
        input_2_weight = [3.]
        input_2_fn = 4.

        with self.assertRaises(TypeError): 
            calc_numerical_gradient(input_2_fn, input_2_weight)

    def test_calc_numerical_gradient(self):
        # test for linear value
        input_1_weight = np.array([[1., 1.5], [2., 3.]])
        
        def forward_fn_1(a, b):
            """
            Parameters
            -----------
            a : float
            b : float
            """
            return np.sum(a * input_1_weight + b)

        input_1_fn = lambda weight: forward_fn_1(3., 5.)

        grad = calc_numerical_gradient(input_1_fn, input_1_weight)

        self.assertTrue(isinstance(grad, np.ndarray)) # type
        self.assertTrue(grad.shape == input_1_weight.shape) # shape
        self.assertTrue((np.round(grad, 5) == np.array([[3., 3.], [3., 3.]])).all()) # checking with hand calculation

        # test for nonlinear value
        input_2_weight = np.array([[1., 1.5], [2., 3.]])
        
        def forward_fn_2(b):
            """
            Parameters
            -----------
            b : float
            """
            return np.sum(np.sin(input_2_weight) + b)

        input_2_fn = lambda weight: forward_fn_2(5.)

        grad = calc_numerical_gradient(input_2_fn, input_2_weight)

        self.assertTrue((np.round(grad, 5) == np.round(np.cos(input_2_weight), 5)).all()) # checking with hand calculation

        # test with backprop
        layer_3 = Linear(2, 1) # layer

        def forward_fn_3(x):
            """
            Parameters
            ----------
            x : numpy.ndarray
            t : numpy.ndarray
            """
            return np.sum(layer_3.forward(x)) / 2. # sum / batch size
        
        input_3 = np.array([[1., 1.5], [2., 3.]])
        input_3_fn = lambda weight : forward_fn_3(input_3)
        grad_A = calc_numerical_gradient(input_3_fn, layer_3.A.val)
        grad_b = calc_numerical_gradient(input_3_fn, layer_3.b.val)

        # backprop
        backprop_grad_A = np.dot(input_3.T, np.ones((2, 1))) / 2. # grad / batch_size
        backprop_grad_b = 1.

        self.assertTrue((np.round(backprop_grad_A, 5) == np.round(grad_A, 5)).all()) # checking with hand calculation
        self.assertTrue((np.round(backprop_grad_b, 5) == np.round(grad_b, 5)).all()) # checking with hand calculation

class TestNumercialGradient(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_numerical_gradient(self):
        # test for type(x)
        input_1_param = 3.
        input_1_fn = lambda weight: None

        with self.assertRaises(TypeError): 
            numerical_gradient(input_1_param, input_1_fn)

        # test for type(f)
        input_2_param = OrderedDict()
        input_2_param['test_1'] = Parameter(np.ones(4))
        input_2_fn = 4.

        with self.assertRaises(TypeError): 
            numerical_gradient(input_2_param, input_2_fn)
    
    def test_numerical_gradient(self):
        # test with backprop
        layer_1 = Linear(2, 1) # layer

        def forward_fn_1(x):
            """
            Parameters
            ----------
            x : numpy.ndarray
            t : numpy.ndarray
            """
            return np.sum(layer_1.forward(x)) / 2. # sum / batch size
        
        input_1 = np.array([[1., 1.5], [2., 3.]])
        input_1_fn = lambda weight : forward_fn_1(input_1)

        # test for times of call
        with patch('src.common.NNgrads.calc_numerical_gradient', return_value=np.zeros(2)) as mock_calc_numerical_gradient:
            numerical_gradient(layer_1.parameters, input_1_fn)
            self.assertTrue(mock_calc_numerical_gradient.call_count == 2)

        # test for value 
        numerical_gradient(layer_1.parameters, input_1_fn)

        # backprop
        backprop_grad_A = np.dot(input_1.T, np.ones((2, 1))) / 2. # grad / batch_size
        backprop_grad_b = 1.

        self.assertTrue((np.round(backprop_grad_A, 5) == np.round(layer_1.A.grad, 5)).all()) # checking with backprop
        self.assertTrue((np.round(backprop_grad_b, 5) == np.round(layer_1.b.grad, 5)).all()) # checking with backprop
