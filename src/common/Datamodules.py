import numpy as np

from .Datafunctions import load_graph_data, load_label_data

class DataLoader():
    def __init__(self):
        self.X_train = None 
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None

        self.X_test = None
        self.Y_test = None
    
    def load(self, train_path, test_path):
        """
        folder名で取得
        """
        pass

    def fold(self, percent=0.3, shuffle=False, seed=None):
        """
        適切に区切ったデータを返す

        Returns
        --------
        self.X_train = None 
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        """

        # 割合計算


        # 個数計算-->そのまま、index指定

        pass

