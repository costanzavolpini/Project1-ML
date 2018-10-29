# -*- coding: utf-8 -*-
import numpy as np

################ HELPER FUNCTION ################

def get_rows_indexes(tx):
	"""Given a dataset return 4 subset looking on jet_num
	Input:
		tx: features
	Output:
		rows_{n}_indexes: indices of rows relative to jet_num n
	"""
	column_jet_num = tx[:,22]
	# create 4 sets of indexes looking on jet_num
	rows_0_indexes = np.where(column_jet_num == 0)
	rows_1_indexes = np.where(column_jet_num == 1)
	rows_2_indexes = np.where((column_jet_num == 2) | (column_jet_num == 3))

	# just for check
	assert len(column_jet_num) == (len(rows_0_indexes[0]) + len(rows_1_indexes[0]) + len(rows_2_indexes[0]))
	return rows_0_indexes, rows_1_indexes, rows_2_indexes



def columns_contains_just_missing_values(s):
	"""Function that returns features containing just missing values
	Input:
		s: feature set
	Output:
		columns_to_remove: array of features "ids" containing just missing values
	"""
	columns_to_remove = []
	for i in range(len(s[0])):
		if(np.all(s[:, i] == -999.)):
			columns_to_remove.append(i)
	return columns_to_remove



def columns_contains_same_value(s):
	"""Function that returns features containing constant value
	Input:
		s: feature set
	Output:
		columns_to_remove: array of features "ids" containing constant value
	"""
	columns_to_remove = []
	for col in range(len(s[0])):
		if len(np.unique(s[:, col])) == 1:
			columns_to_remove.append(col)
	return columns_to_remove

################ END HELPER FUNCTION ################



def generate_3_sets_looking_on_jetnum(tx, y, ids):
	"""Function that returns 3 sets looking on jet_num value
	Input:
		y: labels
		tx: features
		ids: identifiers
	Output:
		features_jet_{n}: dataset of features relative to jet_num n
		y_features_{n}: dataset of labels relative to jet_num n
		ids_{n}: dataset of identifiers relative to jet_num n
	"""
	rows_0_indexes, rows_1_indexes, rows_2_indexes = get_rows_indexes(tx)

	# subsets looking on jet num
	features_jet_0 = tx[rows_0_indexes, :]
	features_jet_1 = tx[rows_1_indexes, :]
	features_jet_2 = tx[rows_2_indexes, :]

	y_feature_0 = y[rows_0_indexes]
	y_feature_1 = y[rows_1_indexes]
	y_feature_2 = y[rows_2_indexes]

	ids_0 = ids[rows_0_indexes]
	ids_1 = ids[rows_1_indexes]
	ids_2 = ids[rows_2_indexes]

	# just to check again
	assert (len(rows_0_indexes[0]) == features_jet_0.shape[1]) & (len(rows_1_indexes[0]) == features_jet_1.shape[1]) & (len(rows_2_indexes[0]) == features_jet_2.shape[1])

	return features_jet_0, features_jet_1, features_jet_2, y_feature_0, y_feature_1, y_feature_2, ids_0, ids_1, ids_2



