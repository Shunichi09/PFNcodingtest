import unittest
from unittest.mock import patch
import numpy as np

# original modules
from src.common.NNbase import Module, Parameter
from src.common.NNmodules import GNN

# test Parameter
class TestParameter(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        input_1 = 3
        with self.assertRaises(TypeError):
            Parameter(input_1)

        input_2 = None
        with self.assertRaises(TypeError):
            Parameter(input_2)

# test Module
class TestModule(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_forward(self):
        test_input = [None]
        with self.assertRaises(NotImplementedError):
            self.module.forward(test_input)

    def test_call(self): 
        test_input = [None]

        with patch.object(Module, "forward", return_value=None) as test_forward:
            self.module.forward(test_input)
            test_forward.assert_called_once() # check once
    
    def test_register_parameters(self):

        input_1 = 0.5 # float

        with self.assertRaises(TypeError):
            self.module.register_parameters(input_1)

        input_2 = np.array([0.1]) # numpy.ndarray

        with self.assertRaises(TypeError):
            self.module.register_parameters(input_2)

        input_3 = Parameter(np.zeros(5))
        self.module.register_parameters(input_3)
        self.assertTrue(id(input_3)==id(self.module.parameters['param_0']))        