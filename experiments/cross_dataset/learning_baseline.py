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

from keras.utils import np_utils
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
TRAIN_DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
TEST_DATASET = 'kasterenB' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
DATASETS = [TRAIN_DATASET, TEST_DATASET]
# Directories of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select segmentation period (0: perfect segmentation)
DELTA = 60
# Select imbalance data treatment
TREAT_IMBALANCE = False
# Select the number of epochs for training
EPOCHS = 54
# Select batch size
BATCH_SIZE = 512
# Select dropout value
DROPOUT = 0.2
# Select loss function
LOSS = 'categorical_crossentropy'
# Select the number of predictions to calculate
N_PREDS = 5
# END CONFIGURATION VARIABLES


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'learning-baseline-' + DAYTIME + '-' + NONES + '-' + TRAIN_DATASET + '-' + TEST_DATASET

# File name for best model weights storage
WEIGHTS_FILE_ROOT = '-weights.hdf5'   


def main(argv):   
    """" Main function
    This is the flow of actions of this main
    0: Initial steps
    1: Load data and reformat for cross dataset experiments (class CrossDatasetFormatter)    
    2: Build the LSTM model (embedding layer frozen)
    3: Test managing imbalanced data in the training set (SMOTE?)
    4: Train the model with the (imbalance-corrected) training set and use the test set to validate (TODO: consult this better)
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
    print('Loading and formatting data')
    cross_dataset_formatter = CrossDatasetFormatter(DATASETS, BASE_INPUT_DIR, DAYTIME, NONES, OP, DELTA)
    # We will use stemmed activities
    cross_dataset_formatter.set_stemmer(True)
    # Use only the needed methods
    cross_dataset_formatter.build_common_activity_to_int_dict()
    cross_dataset_formatter.build_common_action_to_int_dict()
    cross_dataset_formatter.update_x_sequences_no_rep()
    cross_dataset_formatter.update_y_onehot()
    
    cross_dataset_formatter.save_common_activity_int_dicts("cross_activity_int/") 

    # Use the special methods for one-hot encoding
    cross_dataset_formatter.update_x_sequences_no_rep()

    num_common_actions = len(cross_dataset_formatter.common_action_to_int.keys()) + 1 # take into account 0 action which is not mapped to any
    

    # Common data structures
    print("---------------------------------")
    print("Common data structures info:")    
    print("Activity to int:")
    print("   Activities in training: " + str(len(cross_dataset_formatter.activity_to_int_dicts[0].keys())))
    print("   Activities in testing: " + str(len(cross_dataset_formatter.activity_to_int_dicts[1].keys())))
    print("   Common stemmed activities: " + str(len(cross_dataset_formatter.stemmed_activity_to_int.keys())))    
    print("Action to int:")
    print("   Actions in training: " + str(len(cross_dataset_formatter.action_to_int_dicts[0].keys())))
    print("   Actions in testing: " + str(len(cross_dataset_formatter.int_to_action_dicts[1].keys())))
    print("   Common actions: " + str(num_common_actions))
    
    # X sequences to train        
    X_train = np_utils.to_categorical(cross_dataset_formatter.X_seq_no_rep[0], num_classes = num_common_actions) # 0 corresponds to TRAIN_DATASET. We convert it to one hot encoding
    # y one hot to train
    y_train_onehot = cross_dataset_formatter.y_stemmed_onehot_updated[0]
    # y indices to train (for auxiliary tasks)
    y_train_index = np.argmax(y_train_onehot, axis=1)    

    # X sequences to test
    X_test = np_utils.to_categorical(cross_dataset_formatter.X_seq_no_rep[1], num_classes = num_common_actions) # 1 corresponds to TEST_DATASET. We convert it to one hot encoding
    # y one hot to test
    y_test_onehot = cross_dataset_formatter.y_stemmed_onehot_updated[1]
    # y indices to test (for auxiliary tasks)
    y_test_index = np.argmax(y_test_onehot, axis=1)

    max_sequence_length = X_train.shape[1]    
    TOTAL_ACTIVITIES = y_train_onehot.shape[1]
    
    print("X sequences:")
    print('   X train shape:', X_train.shape)
    print('   X test shape:', X_test.shape)
    print("y labels:")    
    print('   y train one hot shape:', y_train_onehot.shape)
    print('   y test one hot shape:', y_test_onehot.shape)
    print('   y train index shape:', y_train_index.shape)
    print('   y test index shape:', y_test_index.shape)
    
    print('max sequence length (train, test): ' + str(max_sequence_length) + ", " + str(X_test.shape[1]))    
    
    train_distro_int = Counter(y_train_index)
    test_distro_int = Counter(y_test_index)
    train_distro_name = {}
    for key in train_distro_int:
        train_distro_name[cross_dataset_formatter.stemmed_int_to_activity[key]] = train_distro_int[key]
    
    test_distro_name = {}
    for key in test_distro_int:
        test_distro_name[cross_dataset_formatter.stemmed_int_to_activity[key]] = test_distro_int[key]


    print("Activity distribution for training:")
    pprint(train_distro_name)
    print("Activity distribution for testing:")
    pprint(test_distro_name)

    # 2: Build the LSTM model (embedding layer frozen)
    print('Building model...')
    sys.stdout.flush()
        
    model = Sequential()            
        
    model.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, num_common_actions)))
    model.add(Dense(TOTAL_ACTIVITIES))
    model.add(Activation('softmax'))        
    model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print('Model built')
    print(model.summary())
    sys.stdout.flush()    

    # 4: Train the model
    print('Training...')        
    sys.stdout.flush()
    
    # Define the callbacks to be used (ModelCheckpoint)    
    weights_file = WEIGHTS + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + WEIGHTS_FILE_ROOT
    modelcheckpoint = ModelCheckpoint(weights_file, monitor='loss', save_best_only=True, verbose=0)
    callbacks = [modelcheckpoint]
    history = model.fit(X_train, y_train_onehot, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, callbacks=callbacks)
    
    # 5: Store the generated learning curves and metrics with the best model (ModelCheckpoint?) -> Conf option SAVE
    plot_filename = PLOTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID    
    utils.plot_training_info(['loss'], True, history.history, plot_filename)
    print("Plots saved in " + PLOTS)
    
    print("Training finished")

    model.load_weights(weights_file)
    yp = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    print("yp shape: " + str(yp.shape))
    print("   sample: " + str(yp[0]))
    # Select the N_PREDS highest values
    unsorted_ypreds = np.argpartition(yp, -N_PREDS, axis=1)[:, -N_PREDS:] # These indices are not sorted
    ypreds = []
    for i in range(len(yp)):        
        yp_i = yp[i]
        unsorted_ypreds_i = unsorted_ypreds[i]
        ypreds.append(np.flip(unsorted_ypreds_i[np.argsort(yp_i[unsorted_ypreds_i])], axis=0))

    ypreds = np.array(ypreds)
    
    print("ypreds shape: " + str(ypreds.shape))
    print("   sample: " + str(ypreds[0]))
    
    ypreds1 = np.argmax(yp, axis=1)

    ytrue = y_test_index

    # Plot non-normalized confusion matrix -> Conf option SAVE    
    results_file_root = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID
    labels = []
    for i in cross_dataset_formatter.stemmed_int_to_activity:
        labels.append(cross_dataset_formatter.stemmed_int_to_activity[i])
    print("Classes for the heatmap (" + str(len(labels)) + ")")
    print(labels)
    utils.plot_heatmap(ytrue, ypreds1, classes=labels,
                       title='Confusion matrix, without normalization: ' + TRAIN_DATASET + '-' + TEST_DATASET,
                       path=results_file_root + '-cm.png')

    # Plot normalized confusion matrix
    utils.plot_heatmap(ytrue, ypreds1, classes=labels, normalize=True,
                       title='Normalized confusion matrix: ' + TRAIN_DATASET + '-' + TEST_DATASET,
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

    metrics_filename = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)
    print("Metrics saved in " + metrics_filename)    


def print_configuration_info():
    """ Dummy function to print configuration parameters expressed as global variables in the script
    """
    print("Selected train dataset:", TRAIN_DATASET)    
    print("Selected test dataset:", TEST_DATASET)    
    print("Dataset base directory:", BASE_INPUT_DIR)    
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)         
    print("Number of epochs: ", EPOCHS)        
    print("Experiment ID:", EXPERIMENT_ID)
    print("Treat imbalance data:", TREAT_IMBALANCE)    
    print("Batch size:", BATCH_SIZE)
    print("Dropout:", DROPOUT)
    print("Loss:", LOSS)
    print("Number of predictions:", N_PREDS)

if __name__ == "__main__":
   main(sys.argv)
