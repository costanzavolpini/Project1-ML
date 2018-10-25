import numpy as np

# Function that returns 4 sets looking on jet_num value
def generate_4_sets_looking_on_jetnum(tx, y, ids):
    column_jet_num = tx[:,22]

    # create 4 sets of indexes looking on jet_num
    rows_0_indexes = np.where(column_jet_num == 0)
    rows_1_indexes = np.where(column_jet_num == 1)
    rows_2_indexes = np.where(column_jet_num == 2)
    rows_3_indexes = np.where(column_jet_num == 3)

    # just for check
    assert len(column_jet_num) == (len(rows_0_indexes[0]) + len(rows_1_indexes[0]) + len(rows_2_indexes[0]) + len(rows_3_indexes[0]))

    # subsets looking on jet num
    features_jet_0 = tx[rows_0_indexes, :]
    features_jet_1 = tx[rows_1_indexes, :]
    features_jet_2 = tx[rows_2_indexes, :]
    features_jet_3 = tx[rows_3_indexes, :]

    y_feature_0 = y[rows_0_indexes]
    y_feature_1 = y[rows_1_indexes]
    y_feature_2 = y[rows_2_indexes]
    y_feature_3 = y[rows_3_indexes]

    ids_0 = y[rows_0_indexes]
    ids_1 = y[rows_1_indexes]
    ids_2 = y[rows_2_indexes]
    ids_3 = y[rows_3_indexes]

    # just to check again
    assert len(rows_0_indexes[0] == features_jet_0.shape[1]) & (len(rows_1_indexes[0]) == features_jet_1.shape[1]) & (len(rows_2_indexes[0]) == features_jet_2.shape[1]) & (len(rows_3_indexes[0]) == features_jet_3.shape[1])

    return features_jet_0, features_jet_1, features_jet_2, features_jet_3, y_feature_0, y_feature_1, y_feature_2, y_feature_3, ids_0, ids_1, ids_2, ids_3

def columns_contains_just_missing_values(s):
    columns_to_remove = []
    for i in range (0, 30):
        if(np.all(s[:, i] == -999.)):
            columns_to_remove.append(i)
    return columns_to_remove


def columns_contains_same_value(s):
    columns_to_remove = []
    for i in range (0, 30):
        if(np.all(s[:, i] == s[0][i])):
            columns_to_remove.append(i)
    return columns_to_remove