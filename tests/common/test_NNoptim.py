import numpy as np
from collections import OrderedDict
import unittest

# original module
from src.common.NNbase import Parameter
from src.common.NNoptim import Optimizer, SGD, MomentumSGD

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_initialize(self):
        # test for type of param
        param = [np.zeros(4), np.zeros(5)]

        with self.assertRaises(TypeError):
            Optimizer(param)

    def test_step(self):
        # test for raising error
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))
        optimizer = Optimizer(param)

        with self.assertRaises(NotImplementedError):
            optimizer.step()

class TestSGD(unittest.TestCase):
    def setUp(self):
        pass
    
    

