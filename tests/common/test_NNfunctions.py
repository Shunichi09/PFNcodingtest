import unittest
import numpy as np

# original modules
from src.common.NNfunctions import sigmoid, relu

class TestSigmoid(unittest.TestCase):

    def test_initialize_sigmoid(self):
        input_1 = "a"
        with self.assertRaises(TypeError):
            sigmoid(input_1)

    def test_sigmoid(self):
        input_1 = [1., -0.5, -0.25, -3.] # test for list
        self.assertTrue((np.round(sigmoid(input_1), 5) == np.round(np.array([0.7310585, 0.3775406, 0.4378234, 0.0474258]), 5)).all())
        self.assertTrue(type(sigmoid(input_1)) == np.ndarray)

        input_2 = np.array([[-1., 0.5], [-0.3, 2.]]) # test for numpy.ndarray
        self.assertTrue((np.round(sigmoid(input_2), 5) == np.round(np.array([[0.2689414, 0.6224593], [0.42555748, 0.8807970]]), 5)).all())

        input_3 = -1. # test for float
        self.assertTrue((np.round(sigmoid(input_3), 5) == np.round(np.array(0.2689414), 5)).all())

class TestRelu(unittest.TestCase):
    def test_initialize_relu(self):
        input_1 = "a"
        with self.assertRaises(TypeError):
            relu(input_1)

    def test_relu(self):
        input_1 = [-1., 0.5, 0.25, 3.] # test for list
        self.assertTrue((np.round(relu(input_1), 5) == np.round(np.array([0., 0.5, 0.25, 3.]), 5)).all())
        self.assertTrue(type(relu(input_1)) == np.ndarray)
        
        input_2 = np.array([[1., -0.5], [0.3, -2.]]) # test for numpy.ndarray
        self.assertTrue((np.round(relu(input_2), 5) == np.round(np.array([[1.0, 0.], [0.3, 0.]]), 5)).all())

        input_3 = 1. # test for float
        self.assertTrue((np.round(relu(input_3), 5) == np.round(np.array(1.), 5)).all())

if __name__ == "__main__":
    unittest.main()