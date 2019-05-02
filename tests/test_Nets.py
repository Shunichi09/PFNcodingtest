import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.common.NNmodules import GNN, Linear, GIN
from src.common.NNfunctions import sigmoid
from src.Nets import VanillaGNN, VanillaGIN

class TestVanillaGNN(unittest.TestCase):
    def setUp(self):
        self.net = VanillaGNN()

    def test_initialize(self):
        with patch.object(VanillaGNN, 'register_parameters', return_value=None):
            with patch.object(GNN, '__init__', return_value=None) as mock_gnn:
                VanillaGNN()
                mock_gnn.assert_called_once()

            with patch.object(Linear, '__init__', return_value=None) as mock_linear:
                VanillaGNN()
                mock_linear.assert_called_once()

    def test_forward(self):
        # compare with GNN --> Linear --> sigmoid and VanillaGNN.forward
        # non batch
        
        input_1 =  [[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]] # non batch
            
        hg = self.net.fc1(input_1, 2)
        s_test = self.net.fc2(hg)
        p_test = sigmoid(s_test)
        predict_test = p_test > 0.5

        p, predict, s = self.net.forward(input_1)

        self.assertTrue((np.round(s, 5) == np.round(s_test, 5)).all())
        self.assertTrue((np.round(p, 5) == np.round(p_test, 5)).all())
        self.assertTrue(predict == predict_test)

        input_2 = [[[0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 1., 0.]], 
                   [[0., 1., 0.],
                    [1., 0., 1.],
                    [0., 1., 0.]]] # batch

        hg = self.net.fc1(input_2, 2)
        s_test = self.net.fc2(hg)
        p_test = sigmoid(s_test)
        predict_test = p_test > 0.5

        p, predict, s = self.net.forward(input_2)

        self.assertTrue((np.round(s, 5) == np.round(s_test, 5)).all())
        self.assertTrue((np.round(p, 5) == np.round(p_test, 5)).all())
        self.assertTrue((predict == predict_test).all())

class TestVanillaGIN(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        with patch.object(VanillaGIN, 'register_parameters', return_value=None):
            with patch.object(GIN, '__init__', return_value=None) as mock_gin:
                VanillaGIN()
                mock_gin.assert_called_once()

            with patch.object(Linear, '__init__', return_value=None) as mock_linear:
                VanillaGIN()
                mock_linear.assert_called_once()