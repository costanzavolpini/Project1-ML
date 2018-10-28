import numpy as np
from proj1_helpers import *
from helpers import *
from execute_code import replace_set_normalize

def build_poly(x, degree):
    # FEATURE AUGMENTATION
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, degree, m, **args):
    """return the loss of ridge regression (of train and test data)."""
    # get k'th subgroup in test, others in train
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)

    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]

    # form data with polynomial degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    tx_train = replace_set_normalize(tx_train)
    tx_test = replace_set_normalize(tx_test)

    # methods used to calculate weights
    loss, w = m(y_train, tx_train, **args)

    # predict the y given weight and data
    y_train_predicted = predict_labels(w, tx_train)
    y_test_predicted = predict_labels(w, tx_test)

    # calculate the accuracy for train and test data
    accuracy_train = calculate_accuracy(y_train_predicted, y_train)
    accuracy_test = calculate_accuracy(y_test_predicted, y_test)

    return loss, w, accuracy_train, accuracy_test