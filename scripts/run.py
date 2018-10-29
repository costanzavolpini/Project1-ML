# -*- coding: utf-8 -*-
# 0. Imports
import numpy as np
from proj1_helpers import load_csv_data
from execute_code import divide_dataset_looking_jetnum_and_remove_features
from execute_code import execute_one_method, generate_submission
from implementations import ridge_regression

# python3 run.py

# 1. Load datasets
y, tx, ids = load_csv_data("../datas/train.csv", sub_sample=False)
no_y, tx_test, ids_test = load_csv_data("../datas/test.csv", sub_sample=False)

# 2. Divide the train dataset looking on jet_num feature (column 22 of tx)
tx_0, tx_1, tx_2, y_0, y_1, y_2, ids_0, ids_1, ids_2 = divide_dataset_looking_jetnum_and_remove_features(y, tx, ids)

# # 3. Execute all methods on all subsets
# acc0, m0, w0 = execute_all_methods(y_0, tx_0, ids_0, True, 8, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-3)
# print("Selected method ", m0)
# print("with accuracy %f" %(acc0))

# acc1, m1, w1 = execute_all_methods(y_1, tx_1, ids_1, True, 6, lambda_=0.0001, initial_w=None, max_iters=800, gamma=1e-3)
# print("Selected method ", m1)
# print("with accuracy %f" %(acc1))

# acc2, m2, w2 = execute_all_methods(y_2, tx_2, ids_2, True, 2, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-4)
# print("Selected method ", m2)
# print("with accuracy %f" %(acc2))

# acc3, m3, w3 = execute_all_methods(y_3, tx_3, ids_3, True, 10, lambda_=0.0001, initial_w=None, max_iters=800, gamma=1e-8)
# print("Selected method ", m3)
# print("with accuracy %f" %(acc3))

# 3. Execute all methods on all subsets
acc0, w0 = execute_one_method(y_0, tx_0, ids_0, "ridge", True, 7, ridge_regression, lambda_ = 0.01)

acc1, w1 = execute_one_method(y_1, tx_1, ids_1, "ridge", True, 10, ridge_regression, lambda_=0.01)

acc2, w2 = execute_one_method(y_2, tx_2, ids_2, "ridge", True, 8, ridge_regression, lambda_=0.01)

# 4. Divide the test dataset looking on jet_num feature (column 22 of tx)
tx_0_t, tx_1_t, tx_2_t, _, _, _, ids_0_t, ids_1_t, ids_2_t = divide_dataset_looking_jetnum_and_remove_features(no_y, tx_test, ids_test)

# 5. Generate submission
final_prediction = generate_submission(tx_0, tx_1, tx_2, ids_0, ids_1, ids_2, w0, w1, w2, "submission-29-10", [7, 10, 8])

print(acc0, acc1, acc2)
print(np.sum(final_prediction == y)/len(y))