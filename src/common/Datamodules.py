import glob
import random
import numpy as np

from .Datafunctions import load_graph_data, load_label_data, natural_number_sort

class DataLoader():
    """
    Attributes
    -------------
    X_train : numpy.ndarray, shape(number_train, D, D)
            input data of train  
    Y_valid : numpy.ndarray, shape(number_train, 1)
        output data of train
    X_valid : numpy.ndarray, shape(number_valid, D, D)
        input data of validation
    Y_valid : numpy.ndarray, shape(number_valid, 1)
        output data of validation 
    X_test : numpy.ndarray, shape(number_test, 1)
        input data of test 
    """
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
    
    def load(self, train_path, test_path):
        """
        Parameters
        -------------
        train_path : str
            path of train data folder 
        test_path : str
            path of test data folder
        """
        # check condition
        if not isinstance(train_path, str):
            raise TypeError("train path should str")

        if not isinstance(test_path, str):
            raise TypeError("test path should str")

        # get all file names
        # graph txt
        names_graph_train_data = natural_number_sort(glob.glob(train_path + "*_graph.txt")) # train
        names_graph_test_data = natural_number_sort(glob.glob(test_path + "*_graph.txt")) # test

        # label txt
        names_label_train_data = natural_number_sort(glob.glob(train_path + "*_label.txt")) # train

        # check
        print("data info : ")
        print("- number of train data : {} \n- number of test data : {}".format(len(names_label_train_data), len(names_graph_test_data)))

        # get train data
        self.X_train = []
        self.Y_train = []
        for i, file_name in enumerate(names_graph_train_data):
            graph_train_data = load_graph_data(file_name)
            label_train_data = load_label_data(names_label_train_data[i])

            self.X_train.append(graph_train_data)
            self.Y_train.append(label_train_data)

        # get test data
        self.X_test = []
        self.Y_test = []
        for i, file_name in enumerate(names_graph_test_data):
            graph_test_data = load_graph_data(file_name)

            self.X_test.append(graph_test_data)
        
        # to numpy
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        self.X_test = np.array(self.X_test)

    def hold_out(self, ratio=0.4, shuffle=True, seed=None):
        """ hold out the train data with keeping the ratio of label data
        Parameters
        -------------
        ratio : float > 0.0 and < 1.0
            ratio of valid data
        shuffle : bool
            if this parameter is true, we shuffle the data and divide
        seed : int
            seed of random state

        Returns
        --------
        X_train : numpy.ndarray, shape(number_train, D, D)
            input data of train  
        Y_valid : numpy.ndarray, shape(number_train, 1)
            output data of train
        X_valid : numpy.ndarray, shape(number_valid, D, D)
            input data of validation
        Y_valid : numpy.ndarray, shape(number_valid, 1)
            output data of validation 

        Notes
        -------
        - train dataを2つに分割します。正解ラベルの割合を保ったまま分割します
        """
        # check condition
        if ratio <= 0.:
            raise ValueError("ratio should be positive")
        
        if ratio >= 1.:
            raise ValueError("ratio should be smaller than 1")

        # get the number of label data
        number_all_train = len(self.Y_train) # number of all data
        number_true = len(self.Y_train[self.Y_train == 0]) # number of data which the label is true
        number_false = len(self.Y_train[self.Y_train == 1]) # number of data which the label is false
        true_ratio = number_true/number_all_train # ratio of true data

        print("- True label data : {0}% ({1}/{2})\n- False label data : {3}% ({4}/{2})"
                .format(round(number_true/number_all_train * 100., 2),
                              number_true, number_all_train,
                        round(number_false/number_all_train * 100., 2),
                              number_false))

        # hold out
        # get number's each data
        number_train = int(number_all_train * (1. - ratio)) # number of train data

        number_train_true = int(number_train * true_ratio) # keep the ratio of true data in train data
        number_train_false = number_train - number_train_true

        # get index
        idx_true = np.where(self.Y_train == 0)[0]
        idx_false = np.where(self.Y_train == 1)[0]

        if shuffle: # if shuffle
            # seed
            random.seed(seed)

            # shuffle the train index
            idx_shuffled_train_true = random.sample([i for i in range(len(idx_true))], number_train_true)
            idx_shuffled_train_false = random.sample([i for i in range(len(idx_false))], number_train_false)

            # get index
            idx_train_true = idx_true[idx_shuffled_train_true]
            idx_train_false = idx_false[idx_shuffled_train_false]

            # shuffle the valid index
            idx_shuffled_valid_true = list(set([i for i in range(len(idx_true))]) - set(idx_shuffled_train_true))
            idx_shuffled_valid_false = list(set([i for i in range(len(idx_false))]) - set(idx_shuffled_train_false))

            idx_valid_true = idx_true[idx_shuffled_valid_true] # get rid of train data
            idx_valid_false = idx_false[idx_shuffled_valid_false]

        else:
            # divide index
            idx_train_true = idx_true[:number_train_true]
            idx_train_false = idx_false[:number_train_false]

            idx_valid_true = idx_true[number_train_true:] # get rid of train data
            idx_valid_false = idx_false[number_train_false:]

        # concat
        idx_train = np.hstack((idx_train_true, idx_train_false))
        idx_valid = np.hstack((idx_valid_true, idx_valid_false))

        # test
        def is_unique(array):
            # test for having same value
            return len(array) == len(set(array.tolist()))

        def compare(array_1, array_2):
            #  test for having same value between two arrays
            return set(array_1.tolist()) and set(array_2.tolist())

        assert is_unique(idx_train), "there are same indexes in idx_train"
        assert is_unique(idx_valid), "there are same indexes in idx_valid"
        assert compare(idx_train, idx_false), "there are same indexes in idx_train and idx_false"

        # set index
        self.X_valid = self.X_train[idx_valid]
        self.Y_valid = self.Y_train[idx_valid]
        self.X_train = self.X_train[idx_train] 
        self.Y_train = self.Y_train[idx_train]

        print("training data info :")
        print("- train(total) : {}".format(len(self.X_train)))
        print("- valid(total) : {}".format(len(self.X_valid)))

        return self.X_train, self.Y_train, self.X_valid, self.Y_valid
    
    def get_test_data(self):
        """return X_test data
        Parameters
        -------------
        X_test : numpy.ndarray
            input data of test 
        """
        return self.X_test

