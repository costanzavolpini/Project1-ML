import numpy as np
from numpy.linalg import eig

def clean_array(tx):
    """Given a dataset tx (array of array) returns an array where all the missing values are masked"""
    return np.ma.masked_values(tx, -999.) # Mask the array in order to not have -999.


def find_mean(tx):
    """Given a dataset tx (array of array) returns an array containing the mean for each array"""
    return (clean_array(tx)).mean(axis=0)


def find_median(tx):
    """Given a dataset tx (array of array) returns an array containing the median for each array"""
    return np.ma.median(clean_array(tx), axis=0)


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


def input_data_PCA(input_data):
	# from numpy.linalg import eig
    #Calculate the mean of each column
    # M = np.mean(input_data.T, axis=1)
    M = find_mean(input_data)

    # center columns by substracting column means
    C = input_data - M

    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)

    # eigendecomposition of covariance matrix
    values, vectors = eig(V)

    # project data
    P = vectors.T.dot(C.T)

    return P.T