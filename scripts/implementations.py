# -*- coding: utf-8 -*-
import numpy as np
from helpers import *

# TODO: Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might have
# kept track of all encountered w for iterative methods, here we only want the last one.

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    if(initial_w == None):
          initial_w = initialize_weight(tx.shape[1])
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    loss = calculate_mse(err, y)
    return loss, w

# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    loss = 0
    w = initial_w
    size = 30

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return loss, w

# Least squares regression using normal equations
def least_squares(y, tx):
    """This function return the optimal weights (using QR), and the mean-squared error"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    try:
        w = np.linalg.solve(a, b)
    except: # case singular matrix
        w = np.linalg.lstsq(a, b)[0]
    loss = compute_loss(y, tx, w)
    return loss, w

# Ridge regression using normal equations //TODO: remove args!
def ridge_regression(y, tx, lambda_, **args):
    """ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return loss, w

# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    losses = []
    w = initial_w

    y = y.copy()
    y[y == -1] = 0

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break

    return loss, w

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    y = y.copy()
    y[y == -1] = 0
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    w = initial_w
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)

    w = w - gamma * gradient
    return loss, w
