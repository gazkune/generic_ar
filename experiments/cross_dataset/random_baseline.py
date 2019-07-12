# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:43:52 2019
@author: gazkune
"""
from __future__ import print_function

from collections import Counter
from collections import defaultdict
import sys
import os
from pprint import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial import distance

import numpy as np
import json

from cross_dataset_formatter import CrossDatasetFormatter
sys.path.append('..')
from utils import Utils

# BEGIN CONFIGURATION VARIABLES
# Dataset
TRAIN_DATASET = 'tapia_s1' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
TEST_DATASET = 'kasterenC' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
DATASETS = [TRAIN_DATASET, TEST_DATASET]
# Directories of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select the number of predictions to calculate
N_PREDS = 5
# Select between weighted random (using train distro) or not
WEIGHTED = True
# Select the number of runs
RUNS = 100
# END CONFIGURATION VARIABLES


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
# TODO: Change this to better address different experiments and models
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'cross_lstm-reg-' + DAYTIME + '-' + NONES + '-' + TRAIN_DATASET + '-' + TEST_DATASET

def main(argv):   
    """" Main function
    """
    # Make an instance of the class Utils
    utils = Utils()

    # Obtain the file number
    maxnumber = utils.find_file_maxnumber(RESULTS)
    filenumber = maxnumber + 1
    print('file number: ', filenumber)
    
    # 1: Load data (X and y_emb)
    print('Loading and formatting data')
    cross_dataset_formatter = CrossDatasetFormatter(DATASETS, BASE_INPUT_DIR, DAYTIME, NONES, OP)
    X_seq_up, y_onehot_up, common_embedding_matrix, common_activity_to_int, common_int_to_activity, common_activity_to_emb = cross_dataset_formatter.reformat_datasets()
    
    # Common data structures
    print("---------------------------------")
    print("Common data structures info:")
    print("Embedding matrices:")
    print("   Embedding matrix shape for training: " + str(cross_dataset_formatter.embedding_weights[0].shape))
    print("   Embedding matrix shape for testing: " + str(cross_dataset_formatter.embedding_weights[1].shape))
    print("   Common embedding matrix shape: " + str(cross_dataset_formatter.common_embedding_matrix.shape))
    print("Activity to int:")
    print("   Activities in training: " + str(len(cross_dataset_formatter.activity_to_int_dicts[0].keys())))
    print("   Activities in testing: " + str(len(cross_dataset_formatter.activity_to_int_dicts[1].keys())))
    print("   Common activities: " + str(len(cross_dataset_formatter.common_activity_to_int.keys())))

    # y one hot to train (to obtain activity distributions)
    y_train_onehot = y_onehot_up[0]
    # y indices to train (for auxiliary tasks)
    y_train_index = np.argmax(y_train_onehot, axis=1)
    # y embeddings to train
    filename = BASE_INPUT_DIR + TRAIN_DATASET + '/complete/' + DAYTIME + '_' + NONES + '/' + TRAIN_DATASET + '_' + OP  + '_60_y_embedding.npy'
    print("File name for y embedding (train): " + filename)
    y_train_emb = np.load(filename)

    # y one hot to test
    y_test_onehot = y_onehot_up[1]
    # y indices to test (for auxiliary tasks)
    y_test_index = np.argmax(y_test_onehot, axis=1)
    # y embeddings to test
    filename = BASE_INPUT_DIR + TEST_DATASET + '/complete/' + DAYTIME + '_' + NONES + '/' + TEST_DATASET + '_' + OP  + '_60_y_embedding.npy'
    print("File name for y embedding (test): " + filename)
    y_test_emb = np.load(filename)    
    
    print("y labels:")
    print('   y train embedding shape:', y_train_emb.shape)
    print('   y test embedding shape:', y_test_emb.shape)
    print('   y train one hot shape:', y_train_onehot.shape)
    print('   y test one hot shape:', y_test_onehot.shape)
    print('   y train index shape:', y_train_index.shape)
    print('   y test index shape:', y_test_index.shape)    
    
    train_distro_int = dict(sorted(Counter(y_train_index).items())) # sort by key
    test_distro_int = dict(sorted(Counter(y_test_index).items())) # sort by key
    train_distro_name = {}
    for key in train_distro_int:
        train_distro_name[common_int_to_activity[key]] = train_distro_int[key]
    
    test_distro_name = {}
    for key in test_distro_int:
        test_distro_name[common_int_to_activity[key]] = test_distro_int[key]

    print("Activity distribution for training:")
    pprint(train_distro_name)
    print("Activity distribution for testing:")
    pprint(test_distro_name)
    
    pprint(test_distro_int)

    metrics_per_fold = utils.init_metrics_per_fold()
    # We will handle top-k accuracies separately (for k > 1)
    ks = [3, 5]
    topk_accuracies = {}
    for k in ks:
        topk_accuracies["acc_at_" + str(k)] = []

    for run in range(RUNS):
        # Generate predictions
        ypred_index = np.array(len(y_test_index))
        ypred_emb = []
        if WEIGHTED == True:
            # Use the training distribution for random predictions
            values = train_distro_int.values()
            total = sum(values)
            probs = np.array(values) / float(total)
            ypred_index = np.random.choice(len(train_distro_int.keys()), len(y_test_index), p=probs)
            # Convert predictions to embeddings
            # We have to use the embeddings of the training set: activity_to_emb_dicts[0] (activity_name:embedding relation)        
            for i in range(len(ypred_index)):
                emb = cross_dataset_formatter.activity_to_emb_dicts[0][cross_dataset_formatter.int_to_activity_dicts[0][str(ypred_index[i])]]
                ypred_emb.append(emb)        

        else:
            # Completely random predictions
            ypred_index = np.random.choice(len(test_distro_int.keys()), len(y_test_index))
            # Convert predictions to embeddings
            # We have to use common_activity_to_emb (activity index : embedding relation)        
            for i in range(len(ypred_index)):
                emb = common_activity_to_emb[ypred_index[i]]
                ypred_emb.append(emb)
        
        ypred_emb = np.array(ypred_emb)        

        print("ypred_emb shape: " + str(ypred_emb.shape))
        print("ypred_index shape: " + str(ypred_index.shape))

        ypreds = obtain_class_predictions(ypred_emb, cross_dataset_formatter.activity_to_emb_dicts[1], cross_dataset_formatter.common_activity_to_int,
                                        cross_dataset_formatter.common_int_to_activity, N_PREDS)

        # Calculate metrics
        ypreds1 = ypreds[:, 0]
        ytrue = y_test_index

        #Dictionary with the values for the metrics (precision, recall and f1)
        #metrics = utils.calculate_evaluation_metrics(ytrue, ypreds1)
        metrics = utils.calculate_evaluation_metrics(ytrue, ypreds1)        
        metrics_per_fold = utils.update_metrics_per_fold(metrics_per_fold, metrics)
        # Calculate top-k accuracy (k=3 and k=5)
        for k in ks:
            acc_at_k = utils.calculate_accuracy_at_k(ytrue, ypreds, k)        
            topk_accuracies["acc_at_" + str(k)].append(acc_at_k)

    # Calculate the mean and std for the metrics obtained for each run
    metrics_per_fold = utils.calculate_aggregate_metrics_per_fold(metrics_per_fold)
    # Calculate the mean and std for the top-k accuracies obtained for each run
    keys = topk_accuracies.keys()
    for key in keys:
        newkey = "mean_acc_at_" + str(key)
        topk_accuracies[newkey] = np.mean(topk_accuracies[key])
        newkey = "std_acc_at_" + str(key)
        topk_accuracies[newkey] = np.std(topk_accuracies[key])

    # Print only mean metrics
    for key in metrics_per_fold:
        if 'mean' in key:
            print(key)
            print(metrics_per_fold[key])
    print("----------------------------")
    for key in topk_accuracies:
        if 'mean' in key:
            print(key)
            print(topk_accuracies[key])
    
        

    
def obtain_class_predictions(yp, activity_dict, activity_to_int_dict, int_to_activity_dict, k=1):
    """Obtains the class predicted from the embeddings using the closest embedding from activity_dict
            
    Usage example:    
        
    Parameters
    ----------
        yp : array, shape = [n_samples, EMBEDDING DIMENSION]
            Word embeddings predicted by the regressor for given test inputs.
        
        activity_dict: dictionary, {class name, word embedding}
            The word embeddings for the activity classes.
        
        activity_to_int_dict: dictionary, {class name, index}
            The activity index for every activity name in the dataset. 
        
        int_to_activity_dict: dictionary, {index, class name}
            The activity name for every activity index in the dataset. 
        k: int
            How many predictions per sample
           
    Returns
    -------
        ypreds : array, shape = [n_samples, k]
            Array with the k activity indices obtained from the predictions of the regressor stored in yp    
    """    

    print('Transforming regression predictions to classes')

    # Simple approach: use fors and check one by one
    
    def closest_activity(pred, activity_dict):
        min_dist = 100.0
        activity = ""
        for key in activity_dict:
            dist = distance.cosine(pred, activity_dict[key])
            if dist < min_dist: 
                min_dist = dist
                activity = key
        return activity, min_dist
    
    def closest_k_activities(pred, activity_dict, k):
        activities = ["!!!"]*k # Build an array of empty activity names (strings)
        min_dists = [1000000.0]*k # Build an array of distances
        for key in activity_dict: # TODO: Ignore 'None' activities that are in activity_dict (if 'no_nones')
            dist = distance.cosine(pred, activity_dict[key])
            i = 0
            inserted = False
            while i < k and not inserted:
                if dist < min_dists[i]:
                    activities.insert(i, key) # The other activities are displaced
                    min_dists.insert(i, dist)
                    # Remove the last element of the lists
                    activities.pop(-1)
                    min_dists.pop(-1)
                    #activities[i] = key
                    #min_dists[i] = dist
                    inserted = True                    

                i += 1

        return activities, min_dists

    ypred = []
    for i in range(len(yp)):        
        activities, dists = closest_k_activities(yp[i], activity_dict, k)        
        acti_indices = np.full(len(activities), -1)
        i = 0
        for act_name in activities:
            try:
                acti_indices[i] = activity_to_int_dict[act_name]
            except KeyError:
                print("Activities: " + str(activities))
                sys.exit()
            
            i += 1     

        ypred.append(acti_indices)        
        
    ypred = np.array(ypred)    
    unique_act_indices = np.unique(ypred)
    #unique_act_names = [int_to_activity_dict[str(x)] for x in unique_act_indices]
    print("Predicted unique activities:")
    print(unique_act_indices)    

    return ypred


if __name__ == "__main__":
   main(sys.argv)