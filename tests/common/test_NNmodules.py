import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np

# original modules
from src.common.NNfunctions import sigmoid, relu
from src.common.NNmodules import Module, GNN, Linear, CrossEntropyLoss

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
        D = 3
        W = np.ones((D, D))
        self.gnn = GNN(D, W)

        input_1 = [[0., 1., 0., 0.],
                         [1., 0., 1., 1.],
                         [0., 1., 0., 1.],
                         [0., 1., 1., 0.]] # non batch
        input_2 = [[[0., 1., 0., 0.],
                          [1., 0., 1., 1.],
                          [0., 1., 0., 1.],
                          [0., 1., 1., 0.]]] # batch

        input_3 = np.array([[0., 1., 0., 0.],
                                 [1., 0., 1., 1.],
                                 [0., 1., 0., 1.],
                                 [0., 1., 1., 0.]]) # non batch
        input_4 = np.array([[[0., 1., 0., 0.],
                                  [1., 0., 1., 1.],
                                  [0., 1., 0., 1.],
                                  [0., 1., 1., 0.]]]) # batch
        
        self.inputs = [input_1, input_2, input_3, input_4]
    
    def test_initialize(self):
        # weightのエラーがでるか
        with self.assertRaises(ValueError):
            D = 4
            W = np.ones((3, 3))
            GNN(D, W)

    def test_forward(self):
        self.check_numetric_forward(self.inputs)
        self.check_forward(self.inputs)
    
    def check_numetric_forward(self, inputs):
        # 値の計算があっているか
        pass
    
    def check_forward(self, inputs):
        # 行列演算にしたことがあっているか
        pass