import unittest
import numpy as np

# original modules
from src.common.NNfunctions import sigmoid, relu

class TestNNfunctions(unittest.TestCase):
    def test_sigmoid(self):
        # case 1-1
        x = [1., -0.5, -0.25, -3.]
        self.assertTrue((np.round(sigmoid(x), 5) == np.round(np.array([0.7310585, 0.3775406, 0.4378234, 0.0474258]), 5)).all())
        self.assertTrue(type(sigmoid(x)) == np.ndarray)
        # case 1-2
        x = np.array([[-1., 0.5], [-0.3, 2.]])
        self.assertTrue((np.round(sigmoid(x), 5) == np.round(np.array([[0.2689414, 0.6224593], [0.42555748, 0.8807970]]), 5)).all())

        # case 2
        x = -1.
        self.assertTrue((np.round(sigmoid(x), 5) == np.round(np.array(0.2689414), 5)).all())

    def test_relu(self):
        # case 1-1
        x = [-1., 0.5, 0.25, 3.]
        self.assertTrue((np.round(relu(x), 5) == np.round(np.array([0., 0.5, 0.25, 3.]), 5)).all())
        self.assertTrue(type(relu(x)) == np.ndarray)
        # case 1-2
        x = np.array([[1., -0.5], [0.3, -2.]])
        self.assertTrue((np.round(relu(x), 5) == np.round(np.array([[1.0, 0.], [0.3, 0.]]), 5)).all())

        # case 2
        x = 1.
        self.assertTrue((np.round(relu(x), 5) == np.round(np.array(1.), 5)).all())

if __name__ == "__main__":
    unittest.main()