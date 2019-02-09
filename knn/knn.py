import pandas as pd
import numpy as np
from sklearn import datasets
from scipy import stats 

# For comparing with our results
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

TEST_PERCENT = 0.3
TEST_K = [1, 3, 5, 7]

# To import some functions one directory up
#   A bit of ugly hack, but there doesn't really exist a better way
import sys
sys.path.append("..")
from useful.preprocessing import check_data, check_dimension, preprocess_queries
from useful.distances import minkowski_distances, hamming_distance


def euclidean_distance_faster(train_cols, queries):
    """
    This uses a vectorization trick taken from here: https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    Not very readable, but (supposedly) ~250 times faster.
    The basic idea is to open up the l2 norm (x-y)² = x² - 2xy + y² 
    """
    return np.sum(train_cols**2, axis=1) - 2*np.dot(queries, train_cols.T) + np.sum(queries**2, axis=1)[:, np.newaxis]
    

def knn(data, queries, k=5, preprocess=True, target_idx=None, warnings=True, 
        dist="euclidean", dist_param=2):
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

    if dist == "euclidean":
        if dist_param == 2:
            distances = euclidean_distance_faster(train_cols, queries)
        else:
            distances = np.array([[minkowski_distances(row, col, p=dist_param)
                            for row in train_cols] for col in queries])
    elif dist == "hamming":
        distances = np.array([[hamming_distance(row, col)
                            for row in train_cols] for col in queries])

    # Get the classes of the K nearest neighbours 
       # and returns which ever one appears the most, using the lowest value to break ties
    lowest_dist_idxs = distances.argsort(axis=1)[:, :k]
    return stats.mode(target[lowest_dist_idxs], axis=1)[0]


if __name__ == "__main__":
    iris = datasets.load_iris()
    data = np.column_stack((iris.data, iris.target))
    train, test = train_test_split(data, test_size=TEST_PERCENT)
    y_test = test[:, 4]

    print_report = False
    results_dic = {}

    print(f"Training on {(1-TEST_PERCENT)*100}% of the iris data and ", 
            f"checking {len(TEST_K)} values for K")
    print("Also checking scikit-learns knn implementation")

    # Test difference dist metrics, p-values for minskowsky dist and values for k
    for dist in ["euclidean", "hamming"]:
        dist_params = [2, 3, 5, 11, 15] if dist == "euclidean" else [None]
        for dist_param in dist_params:
            for k in TEST_K:
                sci_clf = KNeighborsClassifier(n_neighbors=k, metric=dist, p=dist_param)
                sci_clf.fit(train[:, 0:4], train[:, 4])
                sci_pred = sci_clf.predict(test[:, 0:4])
                
                predictions = knn(train, test[:, 0:4], k=k, dist=dist, 
                        dist_param=dist_param)
                if print_report:
                    print(f"Our results for K = {k}, dist= {dist}")
                    print(classification_report(test[:, 4], predictions))
                    print("\n\n")

                results_dic[k] = (accuracy_score(y_test, predictions),
                                    accuracy_score(y_test, sci_pred))

            print(f"Results for {dist}(p={dist_param})")
            results = pd.DataFrame(results_dic, index=["Our KNN", "Sklearn KNN"])
            results.columns = [f"k={x}" for x in results.columns]
            print(results)
            print("\n\n")

    print("Small differences may happen in case of ties with distance and", 
            "possible rounding errors")
    print("I am not sure why the difference for the hamming distance")
