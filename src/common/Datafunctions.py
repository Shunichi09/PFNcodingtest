import re
import numpy as np
import random

def load_graph_data(path):
    """
    Parameters
    -----------
    path : str
        path for graph data

    Returns
    -----------
    graph_data : numpy.ndarray
        graph data, shape is symmetric
    """

    if not isinstance(path, str):
        raise TypeError("path should be str")

    graph_data = []

    with open(path, "r") as f:
        raw_graph_data = f.readlines()
        
        for raw_graph_row in raw_graph_data[1:]:
            raw_graph_row = raw_graph_row.replace("\n","") # windows
            raw_graph_row = raw_graph_row.split(" ")

            graph_row = list(map(float, raw_graph_row)) # to float
            graph_data.append(graph_row)

    return np.array(graph_data)

def load_label_data(path):
    """
    Parameters
    -------------
    path : str
        path for label data

    Returns
    -----------
    label : int
        data label
    """
    if not isinstance(path, str):
        raise TypeError("path should be str")

    with open(path, "r") as f:
        raw_label_data = f.read()
        label = int(raw_label_data.replace("\n",""))
        
    return label

def write_prediction_data(path, prediction_data):
    """
    Parameters
    -----------
    path : str
        path for data
    prediction_data : numpy.ndarray, shape(N) or shape(N, 1)
        predicted labels
    """
    if not isinstance(path, str):
        raise TypeError("path should be str")
    
    if not isinstance(prediction_data, np.ndarray):
        raise TypeError("predicted data should be numpy.ndarray")

    with open(path, "w") as f:
        prediction_data = list(map(lambda x : str(int(x)) + "\n", prediction_data.flatten()))
        f.writelines(prediction_data)

def natural_number_sort(file_names):
    """ natural sort
    Parameters
    ------------
    file_names : list of str
    
    Returns
    -----------
    sorted_list : list of str

    Notes
    -------
    - file名をデータの数字順に並び替えを行います。
    - この関数は~path/*_name.txtで書かれているpathを想定しています。
    """
    # check condition
    if not isinstance(file_names, list):
        raise TypeError("file_names should be list of str")
    
    if not isinstance(file_names[0], str):
        raise TypeError("file_names should be list of str")
    
    if not ("_" in file_names[0] and ".txt" in file_names[0]):
        raise ValueError("the path name should include _ and .txt, such as ~path/0_name.txt")
    
    if re.search('([0-9])', file_names[0]) is None:
        raise ValueError("the path name should include number, such as ~path/0_name.txt")

    # sort
    convert = lambda text: int(text) if text.isdigit() else None
    number_key = lambda key: [convert(c) for c in re.split('([/_])', key)]

    return sorted(file_names, key=number_key)

def shuffle(X, Y, seed=None):
    """
    Parameters
    -------------
    X : numpy.ndarray, shape(N, data.shape)
    Y : numpy.ndarray, shape(N, data.shape)
    seed : int
        seed of random state
    
    Returns
    ----------
    shuffled_X : numpy.ndarray, shape is same as the X
    shuffled_Y : numpy.ndarray, shape is same as the Y
    """
    if len(X) != len(Y):
        raise ValueError("X and Y should have same row size")
    
    # set seed
    random.seed(seed)
    
    # get size of X
    number_X = len(X)

    idx_shuffled = random.sample([i for i in range(number_X)], number_X)

    shuffled_X = X[idx_shuffled] 
    shuffled_Y = Y[idx_shuffled]

    return shuffled_X, shuffled_Y 