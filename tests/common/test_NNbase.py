import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np

# original modules
from src.common.NNbase import Module, Parameter

# test Parameter
class TestParameter(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        """
        """
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