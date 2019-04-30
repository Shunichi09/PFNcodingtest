import numpy as np
from collections import OrderedDict
import unittest

# original module
from src.common.NNbases import Parameter
from src.common.NNoptims import Optimizer, SGD, MomentumSGD

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_initialize(self):
        # test for type of param
        param = [np.zeros(4), np.zeros(5)] # wrong type of param

        with self.assertRaises(TypeError):
            Optimizer(param)

    def test_step(self):
        # test for raising error
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))
        optimizer = Optimizer(param)

        with self.assertRaises(NotImplementedError): # not imeplement the step method of Optimizer
            optimizer.step()

class TestSGD(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        # test for raising error
        # invalid alpha
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))

        with self.assertRaises(TypeError):
            SGD(param, alpha=[0.1]) # wrong value of alpha
    
        with self.assertRaises(ValueError):
            SGD(param, alpha=-0.1) # wrong type of alpha

    def test_step(self):
        # test for value
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))
        param['test_1'].grad = np.ones(4)

        optimizer = SGD(param)
        optimizer.step()

        self.assertTrue((param['test_1'].val == np.ones(4) * (1.-0.0001)).all()) # checking with hand calculation

class TestMomentumSGD(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        # test for raising error
        # invalid alpha
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))

        with self.assertRaises(TypeError):
            MomentumSGD(param, alpha=[0.1]) # wrong value of alpha
    
        with self.assertRaises(ValueError):
            MomentumSGD(param, alpha=-0.1) # wrong type of alpha
        
        with self.assertRaises(TypeError):
            MomentumSGD(param, beta=[0.1]) # wrong value of beta
    
        with self.assertRaises(ValueError):
            MomentumSGD(param, beta=-0.1) # wrong type of beta

    def test_step(self):
        # test for value
        param = OrderedDict()
        param['test_1'] = Parameter(np.ones(4))
        param['test_1'].grad = np.ones(4)

        optimizer = MomentumSGD(param)
        optimizer.step() # first time

        self.assertTrue((param['test_1'].val == np.ones(4) * (1.-0.0001)).all()) # checking with hand calculation
      
        pre_grad = -np.ones(4) * 0.0001 # omega
        grad = -np.ones(4) * 0.0001 + pre_grad * 0.9 # omega

        val = param['test_1'].val + grad

        optimizer.step() # second time

        self.assertTrue((param['test_1'].val == val).all()) # checking with hand calculation