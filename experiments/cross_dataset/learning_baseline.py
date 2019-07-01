# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:56:52 2019
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

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Input, Dropout, TimeDistributed
from keras.layers import LSTM#, CuDNNLSTM
from keras import backend as K # For Adrian's approach to the cosine similarity calculation


import numpy as np
import json

from scipy.spatial import distance

from cross_dataset_formatter import CrossDatasetFormatter
sys.path.append('..')
from utils import Utils

# BEGIN CONFIGURATION VARIABLES
# Dataset
TRAIN_DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
TEST_DATASET = 'kasterenB' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
DATASETS = [TRAIN_DATASET, TEST_DATASET]
# Directories of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select imbalance data treatment
TREAT_IMBALANCE = False
# Select the number of epochs for training
EPOCHS = 130
# Select batch size
BATCH_SIZE = 256
# Select dropout value
DROPOUT = 0.1
# Select loss function
LOSS = 'cosine_proximity' # 'cosine_proximity' # 'mean_squared_error'
# Select the number of predictions to calculate
N_PREDS = 5
# END CONFIGURATION VARIABLES


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
# TODO: Change this to better address different experiments and models
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'learning-baseline-' + DAYTIME + '-' + NONES + '-' + TRAIN_DATASET + '-' + TEST_DATASET

# File name for best model weights storage
WEIGHTS_FILE_ROOT = '-weights.hdf5'   


def main(argv):   
    """" Main function
    """
    print("main")

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
    
    """
    # Adrian's approach, to be tested
    # Build [activity_num, action_dim = 300] np.array using activity_dict and activity_to_int    
    y = np.empty([len(int_to_activity_dict), 300])
    for i in xrange(y.shape[0]):
        activity = int_to_activity_dict[str(i)]
        y[i] = activity_dict[activity]

    #m = K.matmul(yp, K.transpose(y, [1, 0])) -> produces error
    m = np.matmul(yp, np.transpose(y))
    # Using this operation we have a [yp.shape[0], activity_num] matrix where each cell is the cosine similarity between a row in yp and an activity
    # Use ypreds = np.argmax(m, axis=1) to retrieve the column (activity) with the maximum similarity to the prediction in yp
    ypreds = np.argmax(m, axis=1)

    return ypreds
    """

def print_configuration_info():
    """ Dummy function to print configuration parameters expressed as global variables in the script
    """
    print("Selected train dataset:", TRAIN_DATASET)    
    print("Selected test dataset:", TEST_DATASET)    
    print("Dataset base directory:", BASE_INPUT_DIR)    
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)     
    print("Selected action/activity representation:", OP)
    print("Number of epochs: ", EPOCHS)        
    print("Experiment ID:", EXPERIMENT_ID)
    print("Treat imbalance data:", TREAT_IMBALANCE)    
    print("Batch size:", BATCH_SIZE)
    print("Dropout:", DROPOUT)
    print("Loss:", LOSS)
    print("Number of predictions:", N_PREDS)



if __name__ == "__main__":
   main(sys.argv)
