import numpy as np

# Helper functions
def mse(e):
     """This function returns the mean-squared error given the error"""
    return 1/(len(y)) * np.sum(e**2)


# Linear regression using gradient descent (Yousskit)
#TODO

# Linear regression using stochastic gradient descent (Abranches)
#TODO

# Least squares regression using normal equations (Costanza)
#TODO: check!
def least_squares(y, tx):
    """This function return the optimal weights (using QR), and the mean-squared error"""
    X = np.dot(tx, tx.T)
    b = np.dot(tx, t)
    w_optimal = np.linalg.solve(X, b)
    e = np.subtract(y, np.dot(tx, w_optimal))
    return mse(e), w_optimal

# Ridge regression using normal equations (Costanza)
ridge_regression(y, tx, lambda_) #TODO

# Logistic regression using gradient descent or SGD (Abranches)
#TODO

# Regularized logistic regression using gradient descent or SGD (Yousskit)
#TODO