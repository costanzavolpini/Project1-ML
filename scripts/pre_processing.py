import numpy as np

def clean_array(tx):
    return np.ma.masked_values(tx, -999.) # Mask the array in order to not have -999.

def find_mean(tx):
    return (clean_array(tx)).mean(axis=0)

def find_median(tx):
    return np.ma.median(clean_array(tx), axis=0)

def replace_missing_values(tx, new_values):
    x = np.copy(tx)
    indices = np.where(x == -999.)
    x[indices] = np.take(new_values, indices[1])
    return x