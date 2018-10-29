# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(x):
    """This function returns a sigmoid function"""
    s = 1/(1+np.exp(-x))
    return s



def clean_array(tx):
    """Given a dataset tx (array of array) returns an array where all the missing values are masked"""
    return np.ma.masked_values(tx, -999.) # Mask the array in order to not have -999.



################ CALCULATE ERROR, LOSS ################

def calculate_mse(e, y):
    """This function returns the mean-squared error given the error"""
    return 1/(len(y)) * np.sum(e**2)



def calculate_mae(e):
    """Calculate the mae for vector e"""
    return np.mean(np.abs(e))



def compute_loss(y, tx, w):
    """Calculate the loss using mse or mae"""
    e = y - tx.dot(w)
    return calculate_mse(e, y)



def calculate_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood"""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-5)) + (1 - y).T.dot(np.log(1 - pred + 1e-5))
    return np.squeeze(- loss)

################ END CALCULATE LOSS ################



################ CALCULATE ACCURACY / INITIALIZE WEIGHT ################

def calculate_accuracy(preds, vals):
    """Calculate the accuracy"""
    acc = 0
    for i, v in enumerate(vals):
        if preds[i] == v:
            acc += 1
    accuracy = acc / len(vals)
    return accuracy



def initialize_weight(n):
    return np.random.random(n)*2-1

################ END CALCULATE ACCURACY / INITIALIZE WEIGHT ################



################ MEAN / MEDIAN / NORMALIZATION ################

def find_mean(tx):
    """Given a dataset tx (array of array) returns an array containing the mean for each array"""
    return (clean_array(tx)).mean(axis=0)



def find_median(tx):
    """Given a dataset tx (array of array) returns an array containing the median for each array"""
    return np.ma.median(clean_array(tx), axis=0)

################ END MEAN / MEDIAN / NORMALIZATION ################



################ BATCH ################

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


################ END BATCH ################


################ GRADIENT ################

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err



def compute_gradient_log_likelihood(y, tx, w):
    """compute the gradient of loss"""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
################ END GRADIENT ################



################ LOGISTIC ################

def learning_by_gradient_descent(y, tx, w, gamma):
    """ Do one step of gradient descen using logistic regression. Return the loss and the updated w"""
    loss = calculate_log_likelihood(y, tx, w)
    grad = compute_gradient_log_likelihood(y, tx, w)
    w -= gamma * grad
    return loss, w



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    loss = calculate_log_likelihood(y,tx,w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = compute_gradient_log_likelihood(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

################ END LOGISTIC ################




