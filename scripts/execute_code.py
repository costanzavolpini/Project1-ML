import numpy as np
from implementations import *
from proj1_helpers import *
from cross_validation import *
from helpers import *
from pre_processing import *
from split_jet_num import generate_4_sets_looking_on_jetnum, columns_contains_just_missing_values, columns_contains_same_value


def divide_dataset_looking_jetnum_and_remove_features(y, tx, ids):
    """ Divide the dataset looking on jet_num feature (column 22 of tx).
        Input:
            y: labels
            tx: features
            ids: event ids
        Output:
            tx_dropped_columns: features without columns containing just constant values
            features_dropped_n: features without columns containing just constant values where jet_num is equal to n
            y_jet_n: labels with shape compatibles with features_dropped_n
            ids_jet_n: test event ids with shape compatibles with features_dropped_n
    """
    # If PRI_jet_num is zero or one then some features are -999. Divide dataset in 4 looking on jet_num 0, 1, 2 and 3.
    features_jet_0, features_jet_1, features_jet_2, features_jet_3, y_jet_0, y_jet_1, y_jet_2, y_jet_3, ids_jet_0, ids_jet_1, ids_jet_2, ids_jet_3 = generate_4_sets_looking_on_jetnum(tx, y, ids)

    # For each set look how many missing values there are.. in order to detect how many features we want to drop!
    # iterate to find which columns to drop --> Check for constant values, if I feature contains all the same values it is not important.
    columns_to_remove_0 = columns_contains_same_value(features_jet_0[0])
    columns_to_remove_1 = columns_contains_same_value(features_jet_1[0])
    columns_to_remove_2 = columns_contains_same_value(features_jet_2[0])
    columns_to_remove_3 = columns_contains_same_value(features_jet_3[0])

    # remove columns from subset
    features_dropped_0 = np.delete(features_jet_0[0], columns_to_remove_0, axis=1)
    features_dropped_1 = np.delete(features_jet_1[0], columns_to_remove_1, axis=1)
    features_dropped_2 = np.delete(features_jet_2[0], columns_to_remove_2, axis=1)
    features_dropped_3 = np.delete(features_jet_3[0], columns_to_remove_3, axis=1)

    # Columns missing in all subsets
    columns_to_remove = [4, 5, 6, 12, 22, 26, 27, 28]

    tx_dropped_columns = np.delete(tx, columns_to_remove, axis=1)
    return tx_dropped_columns, features_dropped_0, features_dropped_1, features_dropped_2, features_dropped_3, y_jet_0, y_jet_1, y_jet_2, y_jet_3, ids_jet_0, ids_jet_1, ids_jet_2, ids_jet_3


# Execute a method with or without cross_validation
def execute_one_method(y, tx, ids, method_name, cross_validation_flag, m, **args):
    if(cross_validation_flag):
        # can be changed
        seed = 19
        degree = 7
        k_fold = 5

        # store the accuracy of training data and test data
        accuracy_train = []
        accuracy_test = []
        losses = []
        k_indices = build_k_indices(y, k_fold, seed)

        for k in range(k_fold):
            loss, w, single_accuracy_train, single_accuracy_test = cross_validation(y, tx, k_indices, k, degree, m, **args)
            accuracy_train.append(single_accuracy_train)
            accuracy_test.append(single_accuracy_test)
            losses.append(loss)

        # # Just for study the behaviour
        # n = len(accuracy_train)
        # for i in range(n):
        #     print("Iteration: %d) Accuracy train: %f - Accuracy test: %f - Loss: %f\n" % (i, accuracy_train[i], accuracy_test[i], losses[i]))

        mean_accuracy_test = np.mean(accuracy_test)
        min_accuracy_test = np.min(accuracy_test)
        max_accuracy_test = np.max(accuracy_test)

        mean_accuracy_train = np.mean(accuracy_train)
        min_accuracy_train = np.min(accuracy_train)
        max_accuracy_train = np.max(accuracy_train)

        print(method_name)
        print("\nAccuracy test, mean: %f, min value: %f, max value: %f \n" %(mean_accuracy_test, min_accuracy_test, max_accuracy_test))
        print("Accuracy train, mean: %f, min value: %f, max value: %f \n" %(mean_accuracy_train, min_accuracy_train, max_accuracy_train))
        return mean_accuracy_train, method_name, w
    else:
        loss, w = m(y, tx, **args)

        # predict the y given weight and data
        y_predicted = predict_labels(w, tx)

        # calculate the accuracy for train and test data
        accuracy_train = calculate_accuracy(y_predicted, y)
        print(method_name)
        print("\nAccuracy train value: %f \n" %(accuracy_train))
        return accuracy_train, method_name, w

# Return a new dataset where missing values are replaced by mean
def replace_set_mean(tx):
    means = find_mean(tx)
    tx_replaced_by_mean = replace_missing_values(tx, means)
    return tx_replaced_by_mean

# Return a new dataset where missing values are replaced by median
def replace_set_median(tx):
    means = find_mean(tx)
    tx_replaced_by_mean = replace_missing_values(tx, means)
    return tx_replaced_by_mean

# Return a new dataset where missing values are rplaced by missing values with 0 and before that normalize all values without considering missing values
def replace_set_normalize(tx):
    std_data_tx_with_mask = standardize(clean_array(tx))
    tx_std_data_replaced_by_0 = replace_missing_values(std_data_tx_with_mask, np.full((30, 1), 0))
    return tx_std_data_replaced_by_0

# Execute a method with or without cross_validation and with a new datased where missing values are replaced by mean
def execute_one_method_mean(y, tx, ids, method_name, cross_validation_flag, m, **args):
    accuracy, method_name, w = execute_one_method(y, replace_set_mean(tx), ids, replace_set_mean(tx_test), ids_test, method_name, cross_validation_flag, m, **args)
    return accuracy, method_name, w

# Execute a method with or without cross_validation and with a new datased where missing values are replaced by median
def execute_one_method_median(y, tx, ids, method_name, cross_validation_flag, m, **args):
    accuracy, method_name, w = execute_one_method(y, replace_set_median(tx), ids, replace_set_median(tx_test), ids_test, method_name, cross_validation_flag, m, **args)
    return accuracy, method_name, w

# Execute a method with or without cross_validation and with a new datased where missing values are normalized
def execute_one_method_normalized(y, tx, ids, method_name, cross_validation_flag, m, **args):
    accuracy, method_name, w = execute_one_method(y, replace_set_normalize(tx), ids, replace_set_normalize(tx_test), ids_test, method_name, cross_validation_flag, m, **args)
    return accuracy, method_name, w

def execute_all_methods(y, tx, ids, cross_validation_flag, **args):
    accuracy1, method_name1, w1 = execute_one_method(y, tx, ids, "1. LEAST SQUARE", cross_validation_flag, least_squares)
    max_accuracy = accuracy1
    method_name_selected = method_name1
    w_final = w1

    accuracy2, method_name2, w2 = execute_one_method(y, tx, ids, "2. RIDGE REGRESSION", cross_validation_flag, ridge_regression, lambda_=args["lambda_"])
    if(accuracy2 > max_accuracy):
        max_accuracy = accuracy2
        method_name_selected = method_name2
        w_final = w2

    accuracy3, method_name3, w3 = execute_one_method(y, tx, ids, "3. GRADIENT DESCENT", cross_validation_flag, least_squares_GD, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy3 > max_accuracy):
        max_accuracy = accuracy3
        method_name_selected = method_name3
        w_final = w3

    accuracy4, method_name4, w4 = execute_one_method(y, tx, ids, "4. STOCHASTIC GRADIENT", cross_validation_flag, least_squares_SGD, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy4 > max_accuracy):
        max_accuracy = accuracy4
        method_name_selected = method_name4
        w_final = w4

    accuracy5, method_name5, w5 = execute_one_method(y, tx, ids, "5. LOGISTIC REGRESSION", cross_validation_flag, logistic_regression, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy5 > max_accuracy):
        max_accuracy = accuracy5
        method_name_selected = method_name5
        w_final = w5

    return max_accuracy, method_name_selected, w_final
    # ADD OTHER METHODS!!!!!!!!!

def execute_all_methods_median(y, tx, ids, cross_validation_flag, **args):
    # Find median and replace missing values with median
    tx_m = replace_set_median(tx)
    tx_test_m = replace_set_median(tx_test)
    acc, method_selec, w = execute_all_methods(y, tx_m, ids, tx_test_m, ids_test, cross_validation_flag, **args)
    return acc, method_selec, w

def execute_all_methods_mean(y, tx, ids, cross_validation_flag, **args):
    # Find mean and replace missing values with mean
    tx_m = replace_set_mean(tx)
    tx_test_m = replace_set_mean(tx_test)
    acc, method_selec, w = execute_all_methods(y, tx_m, ids, tx_test_m, ids_test, cross_validation_flag, **args)
    return acc, method_selec, w

def execute_all_methods_normalize(y, tx, ids, cross_validation_flag, **args):
    # Normalize replacing missing values with zero
    tx_m = replace_set_normalize(tx)
    tx_test_m = replace_set_normalize(tx_test)
    acc, method_selec, w = execute_all_methods(y, tx_m, ids, tx_test_m, ids_test, cross_validation_flag, **args)
    return acc, method_selec, w

def generate_submission(tx_old, tx0, tx1, tx2, tx3, ids0, ids1, ids2, ids3, w0, w1, w2, w3, name, degree):
    y_test_predicted0 = []
    test_poly0 = build_poly(tx0, degree)
    y_test_predicted0 = predict_labels(w0, test_poly0)

    y_test_predicted1 = []
    test_poly1 = build_poly(tx1, degree)
    y_test_predicted1 = predict_labels(w1, test_poly1)

    y_test_predicted2 = []
    test_poly2 = build_poly(tx2, degree)
    y_test_predicted2 = predict_labels(w2, test_poly2)

    y_test_predicted3 = []
    test_poly3 = build_poly(tx3, degree)
    y_test_predicted3 = predict_labels(w3, test_poly3)

    id_final = np.concatenate((ids0, ids1, ids2, ids3), axis=0)
    y_pred_final = np.concatenate((y_test_predicted0, y_test_predicted1, y_test_predicted2, y_test_predicted3), axis=0)
    create_csv_submission(id_final, y_pred_final, name)
