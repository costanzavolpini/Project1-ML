# -*- coding: utf-8 -*-
import numpy as np
from helpers_functions import find_median, find_mean, clean_array

################ CLEAN DATA ################

def replace_missing_values(tx, new_values):
    """Given a dataset tx (array of array) returns a new dataset tx that instead of missing values contains new values"""
    x = np.copy(tx)

    # last column has 0 values consequences of the -999 -> constant values.
    # in order to also replace by median we make this change of 0 to -999.
    x[:, -1][x[:, -1] == 0] = -999.

    indices = np.where(x == -999.)

    x[indices] = np.take(new_values, indices[1])
    return x



def standardize(x):
    """Given a dataset x, subtract the mean and divide by the standard deviation for each dimension. After this processing, each dimension has zero mean and unit variance."""
    """Standardize the original data set."""
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.nanstd(centered_data, axis=0)
    return std_data



def replace_set_mean(tx):
    """ Return a new dataset where missing values are replaced by mean
        Input:
            tx: features
        Output:
            tx: features where missing values are replaced by mean
    """
    means = find_mean(tx)
    tx_replaced_by_mean = replace_missing_values(tx, means)
    return tx_replaced_by_mean



def feature_augmented(*txs):
    """ Return a new dataset(s) with feature augmented
        Input:
            txs: one or more dataset
        Output:
            txs: one or more dataset with feature augmented
    """
    tx_aug = []
    for tx in txs:
        tx = replace_set_normalize(tx)
        tx = np.column_stack([tx, np.exp(tx).clip(max=999), np.log(tx - tx.min(0) + 1e-8)])
        medians = find_median(tx).data
        mask = tx >= medians
        tx_aug.append(np.column_stack([tx*mask, tx*~mask]))
    return tx_aug



def replace_set_median(tx):
    """ Return a new dataset where missing values are replaced by median
        Input:
            tx: features
        Output:
            tx: features where missing values are replaced by median
    """
    means = find_mean(tx)
    tx_replaced_by_mean = replace_missing_values(tx, means)
    return tx_replaced_by_mean



def replace_set_normalize(tx):
    """ Return a new dataset where missing values are replaced by missing values with 0 and before that normalize all values without considering missing values
        Input:
            tx: features
        Output:
            tx: features where missing values are replaced by 0 and normalized
    """
    std_data_tx_with_mask = standardize(clean_array(tx))
    tx_std_data_replaced_by_0 = replace_missing_values(std_data_tx_with_mask, np.full((tx.shape[0], 1), 0))
    return tx_std_data_replaced_by_0


def outlier_removal(array_jet, top_value, bot_value, per_top = False, per_bot = False):
    """ Remove outliers
        Input:
            array_jet: features
            top_value: top percentile
            bot_value: bottom percentile
            per_top = boolean to check if we want to apply top percentile
            per_bot = boolean to check if we want to apply bottom percentile
        Output:
            array_jet: features without outliers
    """
    percentiles_top = np.percentile(array_jet, top_value, axis=1)
    percentiles_bot = np.percentile(array_jet, bot_value, axis=1)

    for col in range(len(array_jet[0])):
        if per_top == True:
            array_jet[:, col][array_jet[:, col] > percentiles_top[col]] = np.ma.median(array_jet[:, col])

        elif per_bot == True:
            array_jet[:, col][array_jet[:, col] < percentiles_bot[col]] = np.ma.median(array_jet[:, col])

    return array_jet

################ END CLEAN DATA ################


################ FEATURE AUGMENTATION ################

def build_poly(x, degree):
    """Feature augmentation. Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def physics_features(input_data):
    """Physics feature augmentation: add new features to the dataset with physical meaning
        Input:
            input_data: features
        Output:
            new_data: new features
    """
    # subtracting the subleading to the leading regarding eta, phi and pt
    raw_phi = input_data[:, 25] - input_data[:, 28]
    raw_eta = input_data[:, 24] - input_data[:, 27]
    raw_pt = input_data[:, 23] - input_data[:, 26]

    # ratio of the raw, sub and leading jet
    ratio_jet_raw = raw_pt / input_data[:, 29]
    ratio_jet_leading = input_data[:, 23] / input_data[:, 29]
    ratio_jet_subleading = input_data[:, 26] / input_data[:, 29]

    # ratio of transverse momentum
    ratio_met = input_data[:, 19] / input_data[:, 21]
    ratio_lep = input_data[:, 16] / input_data[:, 21]
    ratio_tau = input_data[:, 13] / input_data[:, 21]

    new_feat = np.c_[ raw_phi ,raw_eta ,raw_pt ,ratio_jet_raw ,
    ratio_jet_leading,ratio_jet_subleading ,ratio_met ,
    ratio_lep, ratio_tau]

    new_data = np.c_[input_data, new_feat]

    return new_data


def new_features(input_data, polynomial):
    """ Engineered feature augmentation: add new features to the dataset with physical meaning and correlate them
        Input:
            input_data: features
            polynomial: polynomial degree
        Output:
            new_data: new features
            corr_columns: array of features ids that have a correlation higher than 40%
    """
    new_data = physics_features(input_data)

    new_feat_list = []
    corr_columns  = []

    #------------- Feature interaction
    for column_1 in range(len(new_data[0])):
        for column_2 in range(column_1, len(new_data[0])):
            if column_1 != column_2:
                corr = np.corrcoef(new_data[:,column_1], new_data[:,column_2])[0][1]
                # if correlation is higher than 0.4
                if corr > 0.4:
                    feat_intera = (np.power(new_data[:,column_1], polynomial) +
                        np.power(new_data[:,column_2], polynomial) )

                    new_feat_list.append(feat_intera)
                    corr_columns.append([column_1,column_2])

    new_data = np.c_[new_data, np.array(new_feat_list).T]

    return new_data, corr_columns



def new_feat_test(input_data, corr_columns, polynomial):
    """ Engineered feature augmentation: add new features to the dataset with physical meaning looking on the correlation between features
        NB. Since for test set we have more data we do not know if it would affect the correlation, creating untested features
        Input:
            input_data: features
            polynomial: polynomial degree
            corr_columns: array of features ids that have a correlation higher than 40%
        Output:
            new_data: new features
    """
    new_data = physics_features(input_data)

    new_feat_list = []

    for col in corr_columns:
        feat_intera = (np.power(new_data[:, col[0]], polynomial) +
            np.power(new_data[:, col[1]], polynomial) )

        new_feat_list.append(feat_intera)

    new_data = np.c_[new_data, np.array(new_feat_list).T]

    return new_data

################ END FEATURE AUGMENTATION ################

