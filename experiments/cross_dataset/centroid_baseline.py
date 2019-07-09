# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:56:52 2019
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

import numpy as np
import json

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from scipy.spatial import distance

sys.path.append('..')
from utils import Utils

# BEGIN CONFIGURATION VARIABLES
# Dataset
# Dataset
DATASET = 'kasterenC' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
# Directory of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/' + DATASET + '/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select the number of predictions to calculate
N_PREDS = 5
# END CONFIGURATION VARIABLES

# Directory where X, Y and Embedding files are stored
INPUT_DIR = BASE_INPUT_DIR + 'complete/' + DAYTIME + '_' + NONES + '/'

# File where the embedding matrix weights are stored to initialize the embedding layer of the network
EMBEDDING_WEIGHTS = INPUT_DIR + DATASET + '_' + OP + '_60_embedding_weights.npy'
# File where action sequences are stored
X_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_x.npy'
# File where activity labels for the corresponding action sequences are stored in word embedding format (for regression)
Y_EMB_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_y_embedding.npy'
Y_INDEX_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_y_index.npy'


# To convert the predicted embedding by the regressor to a class we need the json file with that association
ACTIVITY_EMBEDDINGS = BASE_INPUT_DIR + 'word_' + OP + '_activities.json'
# To know the indices of activity names
ACTIVITY_TO_INT = BASE_INPUT_DIR + 'activity_to_int_' + NONES + '.json'
INT_TO_ACTIVITY = BASE_INPUT_DIR + 'int_to_activity_' + NONES + '.json'


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
# TODO: Change this to better address different experiments and models
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'centroid-baseline-' + DAYTIME + '-' + NONES + '-' + DATASET


def main(argv):
    """" Main function
    Take into account that in this baseline there is no training phase. We test directly in the target dataset.
    This is the flow of actions of this main
    0: Initial steps
    1: Load data    
    2: Calculate predictions using the centroids of X sequences and their similarity to y embeddings
    5: Store the generated learning curves and metrics with the best model (ModelCheckpoint? If results get worse with epochs, use EarlyStopping. Validation data?)
    6: Calculate the metrics obtained and store
    """
    # 0: Initial steps
    print_configuration_info()            
    # fix random seed for reproducibility
    np.random.seed(7)
    # Make an instance of the class Utils
    utils = Utils()

    # Obtain the file number
    maxnumber = utils.find_file_maxnumber(RESULTS)
    filenumber = maxnumber + 1
    print('file number: ', filenumber)
    
    # 1: Load data (X and y_emb)
        # 1: Load data (X and y_emb)
    print('Loading data')

    # Load activity_dict where every activity name has its associated word embedding
    with open(ACTIVITY_EMBEDDINGS) as f:
        activity_dict = json.load(f)
    
    # Load the activity indices
    with open(ACTIVITY_TO_INT) as f:
        activity_to_int_dict = json.load(f)
    
    # Load the index to activity relations    
    with open(INT_TO_ACTIVITY) as f:
        int_to_activity = json.load(f)

    # Load embedding matrix, X and y sequences (for y, load both, the embedding and index version)
    embedding_matrix = np.load(EMBEDDING_WEIGHTS)    
    X = np.load(X_FILE)
    y_emb = np.load(Y_EMB_FILE)
    # We need the following two lines for StratifiedKFold
    y_index_one_hot = np.load(Y_INDEX_FILE) 
    y_index = np.argmax(y_index_one_hot, axis=1)

    # We need an activity_index:embedding relation
    # Build it using INT_TO_ACTIVITY and ACTIVITY_EMBEDDINGS files
    activity_index_to_embedding = {}
    for key in int_to_activity:
        activity_index_to_embedding[key] = activity_dict[int_to_activity[key]]

    max_sequence_length = X.shape[1]
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = embedding_matrix.shape[1]
    
    print('X shape:', X.shape)
    print('y shape:', y_emb.shape)
    print('y index shape:', y_index.shape)
    
    print('max sequence length:', max_sequence_length)
    print('features per action:', embedding_matrix.shape[0])
    print('Action max length:', ACTION_MAX_LENGTH)

    # 2: Calculate predictions using the centroids of X sequences and their similarity to y embeddings
    i = 0
    yp = [] # We will store here the predictions per sample (the controids)
    for x_sample in X:
        # x_sample holds a sequence of indices to the embedding matrix (ignore 0 indices)        
        non_zero_x_sample = x_sample[np.nonzero(x_sample)]               
        
        # Now non_zero_x_sample has the indices in x_sample which are not 0                        
        embs = embedding_matrix[non_zero_x_sample.astype(int)]        

        # Calculate the centroid of embs
        centroid = np.mean(embs, axis=0)        
        # Add the predictions
        yp.append(centroid)        
        
    # yp has all the centroids for all the samples. Now obtain the activity clasess
    ypreds = obtain_class_predictions(yp, activity_dict, activity_to_int_dict, int_to_activity, N_PREDS)
    
    # Calculate the metrics        
    ytrue = y_index
    print("ytrue shape: ", ytrue.shape)
    print("ypreds shape: ", ypreds.shape)
    sys.exit()
    
    ypreds1 = ypreds[:, 0]
    # Plot non-normalized confusion matrix -> Conf option SAVE
    
    results_file_root = RESULTS + str(filenumber).zfill(2) + '-' + DATASET + '-' + EXPERIMENT_ID
    labels = []
    for i in range(len(int_to_activity.keys())):
        labels.append(int_to_activity[str(i)])
    print("Classes for the heatmap (" + str(len(labels)) + ")")
    print(labels)    
    utils.plot_heatmap(ytrue, ypreds1, classes=labels,
                      title='Confusion matrix (centroid baseline), without normalization: ' + DATASET,
                      path=results_file_root + '-cm.png')

    # Plot normalized confusion matrix
    utils.plot_heatmap(ytrue, ypreds1, classes=labels, normalize=True,
                      title='Normalized confusion matrix (centroid baseline): ' + DATASET,
                      path=results_file_root + '-cm-normalized.png')

        
    #Dictionary with the values for the metrics (precision, recall and f1)
    metrics = utils.calculate_evaluation_metrics(ytrue, ypreds1)
    # Calculate top-k accuracy (k=3 and k=5)
    k = 3
    acc_at_k = utils.calculate_accuracy_at_k(ytrue, ypreds, k)
    key = "acc_at_" + str(k)
    metrics[key] = acc_at_k
    k = 5
    acc_at_k = utils.calculate_accuracy_at_k(ytrue, ypreds, k)
    key = "acc_at_" + str(k)
    metrics[key] = acc_at_k

    metrics_filename = RESULTS + str(filenumber).zfill(2) + '-' + DATASET + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
       json.dump(metrics, fp, indent=4)
    print("Metrics saved in " + metrics_filename)
    # print(metrics)


    

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



def print_configuration_info():
    """ Dummy function to print configuration parameters expressed as global variables in the script
    """
    print("Selected dataset:", DATASET)        
    print("Dataset base directory:", BASE_INPUT_DIR)    
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)     
    print("Selected action/activity representation:", OP)    
    print("Experiment ID:", EXPERIMENT_ID)      
    print("Number of predictions:", N_PREDS)


if __name__ == "__main__":
   main(sys.argv)