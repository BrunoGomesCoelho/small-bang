import numpy as np
from sklearn import datasets
from scipy import stats


def check_data(data, warnings=True, col_size=None, row_size=None):
    """
    Does some basic verification on our data, checking various things to make sure
        we have correctly formatted everything.
    """
    if len(data.shape) != 2:
        raise IndexError("Wrongly formated numpy ndarray, expected 2 dimensions")
    if data.shape[1] <= 1:
        raise IndexError("How do you expect to predict something if you have 1 or less columns?")

    check_dimension(data, 0, col_size)
    check_dimension(data, 1, col_size)
    
    # Also warn the user if he tries to do anything too crazy
    if warnings:
        if data.shape[0] < 100:
            print(f"Your train data has {data.shape[0]} rows, this low value might affect the algorithm")


def check_dimension(data, shape_idx, size):
    """
    Checks if a given dimension of our data has the correct amount of elements.
    """
    name = "row" if shape_idx == 0 else "column"
    if size is not None and data.shape[shape_idx] != size:
        raise IndexError(f"Wrong amount of {name}, expected {size}, got {data.shape[shape_idx]}")
    

def preprocess_queries(queries, col_size=None, row_size=None):
    """
    Does some basic verification on our queries (data to be analyzed after a model 
        has already been fit).
    Makes sure it is a numpy ndarray and that it has the correct dimensions, if given.
    """
    if not isinstance(queries, np.ndarray):
        try:
            queries = np.ndarray(queries)
        except:
            raise ValueError("Could not convert internally query to numpy array!")
    check_dimension(queries, 0, row_size)
    check_dimension(queries, 1, col_size)
    return queries


if __name__ == "__main__":
    pass
