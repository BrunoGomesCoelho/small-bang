import numpy as np
from sklearn import datasets
from scipy import stats

# For testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
TEST_PERCENT = 0.3
TEST_K = [1, 3, 5, 7]

# To import some functions one directory up
#   A bit of ugly hack, but there doesn't really exist a better way
import sys
sys.path.append("..")
from useful.preprocessing import check_data, check_dimension, preprocess_queries


def knn(data, queries, k=5, preprocess=True, target_idx=None, warnings=True):
    """
    Calculates the K nearest neighbours in our data, given a query, using euclidean distances.
    If various points have the same euclidean distance, 
        we consider the order they appeared in our training data.
    Data and queries are expected to be a numpy ndarray.
    If target_col is not specified, it is assumed to be the last column.
    
    WARNING: In case there is a tie in the voting, 
        the class represented by the lowest value is the one choosen.
    """
    features_count = 0
    if preprocess:
        check_data(data, warnings)
        features_count = data.shape[1] - 1
        queries = preprocess_queries(queries, col_size=features_count)
        
    if target_idx is None:
        target_idx = data.shape[1] - 1
        
    target = data[:, target_idx]
    train_cols = np.delete(data, target_idx, axis=1)[:, 0:target_idx]
    
    # This uses a vectorization trick taken from here: https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # Not very readable, but (supposedly) ~250 times faster
    # The basic idea is to open up the l2 norm (x-y)² = x² - 2xy + y² 
    distances = np.sum(train_cols**2, axis=1) - 2*np.dot(queries, train_cols.T) + np.sum(queries**2, axis=1)[:, np.newaxis]
    
    # Get the classes of the K nearest neighbours 
       # and returns which ever one appears the most, using the lowest value to break ties
    lowest_dist_idxs = distances.argsort(axis=1)[:, :k]
    return stats.mode(target[lowest_dist_idxs], axis=1)[0]


if __name__ == "__main__":
    iris = datasets.load_iris()
    data = np.column_stack((iris.data, iris.target))
    train, test = train_test_split(data, test_size=TEST_PERCENT)

    print(f"Training on {(1-TEST_PERCENT)*100}% of the iris data and "\
            f"checking {len(TEST_K)} values for K")
    for k in TEST_K:
        predictions = knn(train, test[:, 0:4], k=k)
        print(f"Results for K = {k}")
        print(classification_report(test[:, 4], predictions))
        print("\n\n")
