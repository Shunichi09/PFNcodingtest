import numpy as np
import unittest
from unittest.mock import patch, mock_open

# original module
from src.common.Datafunctions import load_graph_data, load_label_data, write_prediction_data

class TestLoadGraphData(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_initialize_load_graph_data(self):
        # test for type(path)
        path = 2 # wrong type of path
        with self.assertRaises(TypeError):
            load_graph_data(path)

    def test_load_graph_data(self):
        path = "testpath"
        test_data = "2\n0 1\n1 0\n"

        with patch("builtins.open", mock_open(read_data=test_data)) as mock_file:                
            graph_data = load_graph_data(path)
            
            # test for called once
            mock_file.assert_called_once()

            # test for value
            self.assertTrue(type(graph_data) == np.ndarray) # type
            # self.assertTrue(type(graph_data[0, 0]) == np.float64) # type
            self.assertTrue((graph_data == np.array([[0., 1], [1., 0.]])).all()) # value

class TestLoadLableData(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_initialize_load_graph_data(self):
        # test for type(path)
        path = 2 # wrong type of path
        with self.assertRaises(TypeError):
            load_label_data(path)

    def test_load_graph_data(self):
        path = "testpath"
        test_data = "1\n"

        with patch("builtins.open", mock_open(read_data=test_data)) as mock_file:                
            label_data = load_label_data(path)
            
            # test for called once
            mock_file.assert_called_once()

            # test for value
            self.assertTrue(type(label_data) == int) # type
            self.assertTrue(label_data == 1) # value

class TestWriteData(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_write_prediction_data(self):
        # test for type of parameters
        path = 2
        prediction_data = np.zeros(3)
        with self.assertRaises(TypeError): # wrong type of path
            write_prediction_data(path, prediction_data)
        
        path = "testpath"
        prediction_data = 3.
        with self.assertRaises(TypeError): # wrong type of prediction_data
            write_prediction_data(path, prediction_data)

    def test_write_prediction_data(self):
        # test for called the write
        path = "testpath"
        prediction_data = np.zeros(3)
        mock = mock_open() # mock the open

        with patch("builtins.open", mock):
            mock_write = mock() # create mock(open)
            write_prediction_data(path, prediction_data)
            mock_write.writelines.assert_called_once_with(["0\n","0\n","0\n"]) # value