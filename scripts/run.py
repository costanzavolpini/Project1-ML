# -*- coding: utf-8 -*-
# TO RUN: python3 run.py
# 0. Imports
import numpy as np
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels
from execute_code import divide_dataset_looking_jetnum_and_remove_features
from execute_code import execute_all_methods, execute_one_method, generate_submission
from implementations import ridge_regression, logistic_regression
from pre_processing import feature_augmented

# 1. Load datasets
# Insert your path
y, tx, ids = load_csv_data("../datas/train.csv", sub_sample=False)
no_y, tx_test, ids_test = load_csv_data("../datas/test.csv", sub_sample=False)

# 2. Make a copy
y_, tx_, ids_, tx_test_, ids_test_ = y.copy(), tx.copy(), ids.copy(), tx_test.copy(), ids_test.copy()

# 3. Feature augmented
tx_ = feature_augmented(tx_)[0]

# 4. Ridge regression method
acc, w = execute_one_method(y_, tx_, ids_, "ridge", False, ridge_regression, lambda_=0)
tx_test_ = feature_augmented(tx_test_)[0]

# # 5. Generate submission
y_test_predicted = predict_labels(w, tx_test_)
create_csv_submission(ids_test_, y_test_predicted, "try")