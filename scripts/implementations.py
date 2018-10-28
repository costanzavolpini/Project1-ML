# -*- coding: utf-8 -*-
import numpy as np
from helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    Input:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Output:
        loss: final loss
        weight: final weight
    """
    # Define parameters to store w and loss
    if(initial_w == None):
          initial_w = initialize_weight(tx.shape[1])
    losses = []
    w = initial_w
    count = 0
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err, y)
        # gradient w by descent update
        w = w - gamma * grad
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            count += 1
            if count == 10:
                break
    return loss, w



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    Input:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Output:
        loss: final loss
        weight: final weight
    """
    # Define parameters to store w and loss
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    losses = []
    w = initial_w
    size = 30
    count = 0

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=size, num_batches=1):
            # compute a stochastic gradient and loss
            loss = compute_loss(y, tx, w)
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
                count += 1
                if count == 10:
                    break
    return loss, w



def least_squares(y, tx):
    """Least squares regression using normal equations (QR)
    Input:
        y: labels
        tx: features
    Output:
        loss: final loss
        weight: final weight
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    try:
        w = np.linalg.solve(a, b)
    except: # case singular matrix
        w = np.linalg.lstsq(a, b)[0]
    loss = compute_loss(y, tx, w)
    return loss, w



def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
    Input:
        y: labels
        tx: features
        lambda: the regularization parameter
    Output:
        loss: final loss
        weight: final weight
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return loss, w



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD
    Input:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Output:
        loss: final loss
        weight: final weight
    """
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    losses = []
    w = initial_w

    y = y.copy()
    y[y == -1] = 0
    count = 0

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            count += 1
            if count == 10:
                break

    return loss, w



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD.
    Input:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Output:
        loss: final loss
        weight: final weight
    """
    y = y.copy()
    y[y == -1] = 0
    if(initial_w == None):
        initial_w = initialize_weight(tx.shape[1])
    w = initial_w
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)

    w = w - gamma * gradient
    return loss, w
