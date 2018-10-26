import numpy as np

# Given a dataset tx (array of array) returns an array where all the missing values are masked
def clean_array(tx):
    return np.ma.masked_values(tx, -999.) # Mask the array in order to not have -999.

# Given a dataset tx (array of array) returns an array containing the mean for each array
def find_mean(tx):
    return (clean_array(tx)).mean(axis=0)

# Given a dataset tx (array of array) returns an array containing the median for each array
def find_median(tx):
    return np.ma.median(clean_array(tx), axis=0)

# Given a dataset tx (array of array) returns a new dataset tx that instead of missing values contains new values
def replace_missing_values(tx, new_values):
    x = np.copy(tx)
    indices = np.where(x == -999.)
    x[indices] = np.take(new_values, indices[1])
    return x

# Matrix Standardization
# Given a dataset x, subtract the mean and divide by the standard deviation for each dimension.
# After this processing, each dimension has zero mean and unit variance.
def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data
