import numpy as np
from implementations import *
from proj1_helpers import *
from cross_validation import *
from helpers import *
from pre_processing import *
from split_jet_num import generate_4_sets_looking_on_jetnum, columns_contains_same_value


def divide_dataset_looking_jetnum_and_remove_features(y, tx, ids):
    """ Divide the dataset looking on jet_num feature (column 22 of tx).
        Input:
            y: labels
            tx: features
            ids: event ids
        Output:
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

    features_dropped_0 = replace_set_normalize(features_dropped_0)
    features_dropped_1 = replace_set_normalize(features_dropped_1)
    features_dropped_2 = replace_set_normalize(features_dropped_2)
    features_dropped_3 = replace_set_normalize(features_dropped_3)
    return features_dropped_0, features_dropped_1, features_dropped_2, features_dropped_3, y_jet_0, y_jet_1, y_jet_2, y_jet_3, ids_jet_0, ids_jet_1, ids_jet_2, ids_jet_3


def execute_one_method(y, tx, ids, method_name, cross_validation_flag, m, **args):
    """ Execute one method and return the accuracy and weight.
        Input:
            y: labels
            tx: features
            ids: event ids
            method_name: name of the method that we want to run
            cross_validation_flag: boolean flag, if true use cross validation to evaluate
            m: method that we want to run
            args: empty or parameters to pass to the method
        Output:
            accuracy: (mean) accuracy obtained running the method
            w: weights
    """
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

        mean_accuracy_test = np.mean(accuracy_test)
        min_accuracy_test = np.min(accuracy_test)
        max_accuracy_test = np.max(accuracy_test)

        mean_accuracy_train = np.mean(accuracy_train)
        min_accuracy_train = np.min(accuracy_train)
        max_accuracy_train = np.max(accuracy_train)

        print(method_name)
        print("\nAccuracy test, mean: %f, min value: %f, max value: %f \n" %(mean_accuracy_test, min_accuracy_test, max_accuracy_test))
        print("Accuracy train, mean: %f, min value: %f, max value: %f \n" %(mean_accuracy_train, min_accuracy_train, max_accuracy_train))
        return mean_accuracy_test, w
    else:
        loss, w = m(y, tx, **args)

        # predict the y given weight and data
        y_predicted = predict_labels(w, tx)

        # calculate the accuracy for train and test data
        accuracy_train = calculate_accuracy(y_predicted, y)
        print(method_name)
        print("\nAccuracy train value: %f \n" %(accuracy_train))
        return accuracy_train, w



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
    tx_std_data_replaced_by_0 = replace_missing_values(std_data_tx_with_mask, np.full((30, 1), 0))
    return tx_std_data_replaced_by_0



def execute_all_methods(y, tx, ids, cross_validation_flag, **args):
    """ Execute all machine learning baseline and return the accuracy and weight of the best method with the higher accuracy.
        Input:
            y: labels
            tx: features
            ids: event ids
            cross_validation_flag: boolean flag, if true use cross validation to evaluate
            args: empty or parameters to pass to the method
        Output:
            accuracy: (mean) accuracy obtained running the method
            method_name: name of the method with the higher accuracy
            w: weights
    """
    accuracy1, w1 = execute_one_method(y, tx, ids, "1. LEAST SQUARE", cross_validation_flag, least_squares)
    max_accuracy = accuracy1
    method_name_selected = "LEAST SQUARE"
    w_final = w1

    accuracy2, w2 = execute_one_method(y, tx, ids, "2. RIDGE REGRESSION", cross_validation_flag, ridge_regression, lambda_=args["lambda_"])
    if(accuracy2 > max_accuracy):
        max_accuracy = accuracy2
        method_name_selected = "RIDGE REGRESSION"
        w_final = w2

    accuracy3, w3 = execute_one_method(y, tx, ids, "3. GRADIENT DESCENT", cross_validation_flag, least_squares_GD, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy3 > max_accuracy):
        max_accuracy = accuracy3
        method_name_selected = "GRADIENT DESCENT"
        w_final = w3

    accuracy4, w4 = execute_one_method(y, tx, ids, "4. STOCHASTIC GRADIENT", cross_validation_flag, least_squares_SGD, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy4 > max_accuracy):
        max_accuracy = accuracy4
        method_name_selected = "STOCHASTIC GRADIENT"
        w_final = w4

    accuracy5, w5 = execute_one_method(y, tx, ids, "5. LOGISTIC REGRESSION", cross_validation_flag, logistic_regression, initial_w=args["initial_w"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy5 > max_accuracy):
        max_accuracy = accuracy5
        method_name_selected = "LOGISTIC REGRESSION"
        w_final = w5

    accuracy6, w6 = execute_one_method(y, tx, ids, "6. REGULARIZED LOGISTIC REGRESSION", cross_validation_flag, reg_logistic_regression, initial_w=args["initial_w"], lambda_=args["lambda_"], max_iters=args["max_iters"], gamma=args["gamma"])
    if(accuracy6 > max_accuracy):
        max_accuracy = accuracy6
        method_name_selected = "REGULARIZED LOGISTIC REGRESSION"
        w_final = w6

    return max_accuracy, method_name_selected, w_final


def generate_submission(tx0, tx1, tx2, tx3, ids0, ids1, ids2, ids3, w0, w1, w2, w3, name, degree):
    """ Generate a submission given the 4 subset of the TEST data (tx_n and ids_n) and the weights generated by the TRAIN dataset.
        Input:
            tx_n: features of test data
            ids_n: event ids of testa data
            w_n: weights generated by train data
            name: name of the file .csv of the submission
            degree: degree for feature augmentation (to pass to build_poly method)
        Output: the file .csv will be generated
    """
    y_test_predicted0 = []
    test_poly0 = build_poly(tx0, degree)
    test_poly0 = replace_set_normalize(test_poly0)
    y_test_predicted0 = predict_labels(w0, test_poly0)

    y_test_predicted1 = []
    test_poly1 = build_poly(tx1, degree)
    test_poly1 = replace_set_normalize(test_poly1)
    y_test_predicted1 = predict_labels(w1, test_poly1)

    y_test_predicted2 = []
    test_poly2 = build_poly(tx2, degree)
    test_poly2 = replace_set_normalize(test_poly2)
    y_test_predicted2 = predict_labels(w2, test_poly2)

    y_test_predicted3 = []
    test_poly3 = build_poly(tx3, degree)
    test_poly3 = replace_set_normalize(test_poly3)
    y_test_predicted3 = predict_labels(w3, test_poly3)

    # id_final = np.concatenate((ids0, ids1, ids2, ids3), axis=0)
    # y_pred_final = np.concatenate((y_test_predicted0, y_test_predicted1, y_test_predicted2, y_test_predicted3), axis=0)


    jet_num_0 = np.c_[ids0, y_test_predicted0]
    jet_num_1 = np.c_[ids1, y_test_predicted1]
    jet_num_2 = np.c_[ids2, y_test_predicted2]
    jet_num_3 = np.c_[ids3, y_test_predicted3]

    print(jet_num_0)

    final = np.concatenate((jet_num_0, jet_num_1, jet_num_2, jet_num_3), axis=0)
    finallll = list(final)
    finallll.sort(key=lambda x:x[0])
    g = np.array(finallll)
    return g[:, -1]
    # return y_pred_final
    # create_csv_submission(id_final, y_pred_final, name)
