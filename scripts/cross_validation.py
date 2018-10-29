# -*- coding: utf-8 -*-
import numpy as np
from pre_processing import build_poly, replace_set_normalize
from proj1_helpers import predict_labels
from helpers_functions import calculate_accuracy

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold
        Input:
            y: labels
            k_fold: number of fold that we want to generate
            seed: number to make a random seed
        Output:
            array: indices of the folds
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, degree, m, **args):
    """Apply cross validation given a set and indices of the fold.
        Input:
            y: labels
            x: features
            k_indices: indices of the folds
            k: number of fold that we want to generate
            degree: degree for feature augmentation (build_poly method)
            m: method that we want to run
            args: possible parameters to pass to the method
        Output:
            loss: final loss
            w: final weights
            accuracy_train: final accuracy obtained by train set
            accuracy_test: final accuracy obtained by test set
    """
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