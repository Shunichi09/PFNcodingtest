import numpy as np

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