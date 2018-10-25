import numpy as np

def calculate_mse(e, y):
    """This function returns the mean-squared error given the error"""
    return 1/(len(y)) * np.sum(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e, y)

def calculate_accuracy(preds, vals):
    """Calculate the accuracy"""
    acc = 0
    for i, v in enumerate(vals):
        if preds[i] == v:
            acc += 1
    accuracy = acc / len(vals)
    return accuracy

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

