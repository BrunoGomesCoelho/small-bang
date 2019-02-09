import numpy as np
def minkowski_distances(vector_x, vector_z, p=2):
    """ Calculates the Minkowski distance for a choosen value p between two vectors.
    
    -----------
    Parameters:
    
    vector_x / vector_z : numpy array
            Both are expected to have the same size and conntain float values
    
    p : int or +/- np.inf
            The value to  raise and then take the root in the minkowski distance formula
    
    Using +/- np.inf for representing a infinity, they are preprocessed to their max/min representation.
    Be careful since very large values of p may take a some time to process, even though we use a vectorized approach
    """
    if p == np.inf:
        return max(np.abs(vector_x - vector_z))
    elif p == - np.inf:
        return min(np.abs(vector_x - vector_z))
    
    if not isinstance(p, int):
        raise TypeError("p is {type(p)}, not a integer!")
    elif p <= 0:
        raise ValueError("p must be a non negative, non zero integer!")
        
    return np.sum(np.abs(vector_x-vector_z)**p) ** (1/p)


def hamming_distance(vector_x, vector_z):
    """ Returns the Hamming distance (amount of different elements) between two vectors
    We divide by the length of the vector to follow sklean's convention
    
    Usefull as a distance metric for categorical data
    ----------
    Parameters
    vector_x / vector_z : numpy array or pd.Series
    """

    # If you want, you can check scipy's hamming distance, but it still gives
    #   different results to sklearn
    #from scipy.spatial.distance import hamming
    #return hamming(vector_x, vector_z)
    return np.sum(~(vector_x == vector_z)) / len(vector_x)

