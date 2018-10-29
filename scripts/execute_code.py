# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from split_jet_num import columns_contains_same_value, generate_3_sets_looking_on_jetnum
from pre_processing import replace_set_normalize, build_poly, feature_augmented
from proj1_helpers import predict_labels, create_csv_submission
from cross_validation import build_k_indices, cross_validation
from helpers_functions import calculate_accuracy

################ SPLIT DATASET ON JET_NUM ################

def divide_dataset_looking_jetnum_and_remove_features(y, tx, ids, degree=7):
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
    features_jet_0, features_jet_1, features_jet_2, y_jet_0, y_jet_1, y_jet_2, ids_jet_0, ids_jet_1, ids_jet_2 = generate_3_sets_looking_on_jetnum(tx, y, ids)

    # For each set look how many missing values there are.. in order to detect how many features we want to drop!
    # iterate to find which columns to drop --> Check for constant values, if I feature contains all the same values it is not important.
    columns_to_remove_0 = columns_contains_same_value(features_jet_0[0])
    columns_to_remove_1 = columns_contains_same_value(features_jet_1[0])
    columns_to_remove_2 = columns_contains_same_value(features_jet_2[0])

    # remove columns from subset
    features_dropped_0 = np.delete(features_jet_0[0], columns_to_remove_0, axis=1)
    features_dropped_1 = np.delete(features_jet_1[0], columns_to_remove_1, axis=1)
    features_dropped_2 = np.delete(features_jet_2[0], columns_to_remove_2, axis=1)

    features_dropped_0, features_dropped_1, features_dropped_2 = feature_augmented(features_dropped_0, features_dropped_1, features_dropped_2)

    features_dropped_0 = build_poly(features_dropped_0, degree)
    features_dropped_1 = build_poly(features_dropped_1, degree)
    features_dropped_2 = build_poly(features_dropped_2, degree)

    features_dropped_0 = replace_set_normalize(features_dropped_0)
    features_dropped_1 = replace_set_normalize(features_dropped_1)
    features_dropped_2 = replace_set_normalize(features_dropped_2)

    return features_dropped_0, features_dropped_1, features_dropped_2, y_jet_0, y_jet_1, y_jet_2, ids_jet_0, ids_jet_1, ids_jet_2

################ END SPLIT DATASET ON JET_NUM ################



################ EXECUTE METHOD(S) ################

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

################ END EXECUTE METHOD(S) ################



################ GENERATE SUBMISSION  ################

def generate_submission(tx0, tx1, tx2, ids0, ids1, ids2, w0, w1, w2, name):
    """ Generate a submission given the 3 subsets of the TEST data (tx_n and ids_n) and the weights generated by the TRAIN dataset.
        Input:
            tx_n: features of test data
            ids_n: event ids of testa data
            w_n: weights generated by train data
            name: name of the file .csv of the submission
        Output: the file .csv will be generated
    """
    y_test_predicted0 = []
    y_test_predicted0 = predict_labels(w0, tx0)

    y_test_predicted1 = []
    y_test_predicted1 = predict_labels(w1, tx1)

    y_test_predicted2 = []
    y_test_predicted2 = predict_labels(w2, tx2)

    jet_num_0 = np.c_[ids0, y_test_predicted0]
    jet_num_1 = np.c_[ids1, y_test_predicted1]
    jet_num_2 = np.c_[ids2, y_test_predicted2]

    final_pred = np.concatenate((jet_num_0, jet_num_1, jet_num_2), axis=0)
    final_pred_list = list(final_pred)
    final_pred_list.sort(key=lambda x:x[0])
    result = np.array(final_pred_list)
    create_csv_submission(result[:, 0], result[:, -1], name)
    return result[:, -1]

################ END GENERATE SUBMISSION  ################