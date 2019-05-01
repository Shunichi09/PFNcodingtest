import numpy as np
import unittest
from unittest.mock import patch

from src.common.Datamodules import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_test(self):
        # test for type of train_path
        train_path = 4
        test_path = "path/"
        with self.assertRaises(TypeError):
            data_loader = DataLoader()
            data_loader.load(train_path, test_path)
        
        # test for type of train_path
        train_path = "path/"
        test_path = 4
        with self.assertRaises(TypeError):
            data_loader = DataLoader()
            data_loader.load(train_path, test_path)

    @patch('src.common.Datamodules.load_label_data', return_value=None)
    @patch('src.common.Datamodules.load_graph_data', return_value=None)
    @patch('src.common.Datamodules.natural_number_sort', return_value=["path/0_graph.txt", "path/1_graph.txt"])
    @patch('glob.glob', return_value= ["path/0_graph.txt", "path/1_graph.txt"])
    def test_load(self, mock_glob, mock_natural_number_sort, mock_load_graph_data, mock_load_label_data):
        print(mock_natural_number_sort)
        
        # test for times of function call
        data_loader = DataLoader()
        data_loader.load("test", "test")

        self.assertTrue(mock_natural_number_sort.call_count == 3) 
        self.assertTrue(mock_load_label_data.call_count == 2)
        self.assertTrue(mock_load_graph_data.call_count == 4)

    def test_initialize_hold_out(self):
        # test for value of ratio
        with self.assertRaises(ValueError):
            data_loader = DataLoader()
            data_loader.hold_out(ratio=-0.5)
        
        # test for value of ratio
        with self.assertRaises(ValueError):
            data_loader = DataLoader()
            data_loader.hold_out(ratio=1.5)
    
    def test_hold_out(self):
        # hold outのテストはhold outの関数内にかかれています
        pass