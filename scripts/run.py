# 0. Imports
import numpy as np
from implementations import *
from proj1_helpers import *
from cross_validation import *
from pre_processing import *
from split_jet_num import *
from execute_code import *

# 1. Load datasets
y, tx, ids = load_csv_data("datas/train.csv", sub_sample=False)
no_y, tx_test, ids_test = load_csv_data("datas/test.csv", sub_sample=False)

# 2. Divide the train dataset looking on jet_num feature (column 22 of tx)
tx_0, tx_1, tx_2, tx_3, y_0, y_1, y_2, y_3, ids_0, ids_1, ids_2, ids_3 = divide_dataset_looking_jetnum_and_remove_features(y, tx, ids)

# 3. Execute all methods on all subsets
acc0, m0, w0 = execute_all_methods(y_0, tx_0, ids_0, True, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-3)
print("Selected method ", m0)
print("with accuracy %f" %(acc0))

acc1, m1, w1 = execute_all_methods(y_1, tx_1, ids_1, True, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-3)
print("Selected method ", m1)
print("with accuracy %f" %(acc1))

acc2, m2, w2 = execute_all_methods(y_2, tx_2, ids_2, True, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-3)
print("Selected method ", m2)
print("with accuracy %f" %(acc2))

acc3, m3, w3 = execute_all_methods(y_3, tx_3, ids_3, True, lambda_=0.0001, initial_w=None, max_iters=1000, gamma=1e-3)
print("Selected method ", m3)
print("with accuracy %f" %(acc3))

# 4. Divide the test dataset looking on jet_num feature (column 22 of tx)
tx_0_t, tx_1_t, tx_2_t, tx_3_t, _, _, _, _, ids_0_t, ids_1_t, ids_2_t, ids_3_t = divide_dataset_looking_jetnum_and_remove_features(y_test, tx_test, ids_test)

# 5. Generate submission
generate_submission(tx_0_t, tx_1_t, tx_2_t, tx_3_t, ids_0_t, ids_1_t, ids_2_t, ids_3_t, w0, w1, w2, w3, "submission-28-10-to-submit", 7)
