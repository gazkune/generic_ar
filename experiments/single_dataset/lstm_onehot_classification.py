# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:23:52 2019
@author: gazkune
"""
from __future__ import print_function

from collections import Counter
from collections import defaultdict
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Input, Dropout, TimeDistributed
from keras.layers import LSTM#, CuDNNLSTM
from keras import backend as K # For Adrian's approach to the cosine similarity calculation


import numpy as np
import json

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE, RandomOverSampler

from scipy.spatial import distance

sys.path.append('..')
from utils import Utils

# BEGIN CONFIGURATION VARIABLES
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
# Select the muber of folds in the cross-validation process
FOLDS = 10
# Select imbalance data treatment
TREAT_IMBALANCE = False
# Select the number of epochs for training
EPOCHS = 3
# Select batch size
BATCH_SIZE = 256
# Select dropout value
DROPOUT = 0.7
# Select loss function
LOSS = 'categorical_crossentropy' # 

# Select whether intermediate plots and results should be saved
SAVE = False
# END CONFIGURATION VARIABLES

# Directory where X, Y and Embedding files are stored
INPUT_DIR = BASE_INPUT_DIR + 'complete/' + DAYTIME + '_' + NONES + '/'

# File where action sequences are stored (action index sequences)
X_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_x.npy'
# File where activity labels for the corresponding action sequences are stored in one hot vector format 
Y_INDEX_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_y_index.npy'

# To know the indices of activity names
ACTIVITY_TO_INT = BASE_INPUT_DIR + 'activity_to_int_' + NONES + '.json'
INT_TO_ACTIVITY = BASE_INPUT_DIR + 'int_to_activity_' + NONES + '.json'

# ID for the experiment which is being run -> used to store the files with
# appropriate naming
# TODO: Change this to better address different experiments and models
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'lstm-onehot-class-' + DAYTIME + '-' + NONES

# File name for best model weights storage
WEIGHTS_FILE_ROOT = '_lstm-onehot-classification-weights.hdf5'   


def main(argv):   
    """" Main function
    
    This is the flow of actions of this main
    0: Initial steps
    1: Load data (X and y_emb) and needed dictionaries (activity-to-int, etc.)    
    2: Generate K partitions of the dataset (KFold cross-validation)
    3: For each partition (train, test):
       3.1: Build the LSTM model
       3.2: Manage imbalanced data in the training set (SMOTE?)
       3.3: Train the model with the imbalance-corrected training set and use the test set to validate
       3.4: Store the generated learning curves and metrics with the best model (ModelCheckpoint? 
               If results get worse with epochs, use EarlyStopping)
    4: Calculate the mean and std for the metrics obtained for each partition and store
    """
    # 0: Initial steps
    print_configuration_info()        
    # fix random seed for reproducibility
    np.random.seed(7)
    # Make an instance of the class Utils
    utils = Utils()

    # Obtain the file number
    maxnumber = utils.find_file_maxnumber(RESULTS + DATASET + '/')
    filenumber = maxnumber + 1
    print('file number: ', filenumber)
    
    # 1: Load data (X and y_emb)
    print('Loading data')
    
    # Load the activity indices
    with open(ACTIVITY_TO_INT) as f:
        activity_to_int_dict = json.load(f)
    
    # Load the index to activity relations    
    with open(INT_TO_ACTIVITY) as f:
        int_to_activity = json.load(f)

    # Load X and y sequences (for y, load both, the embedding and index version)    
    X = np.load(X_FILE)
    X_onehot = np_utils.to_categorical(X)    
    # We need the following two lines for StratifiedKFold and classification
    y_index_onehot = np.load(Y_INDEX_FILE) 
    y_index = np.argmax(y_index_onehot, axis=1)

    max_sequence_length = X.shape[1]
    action_feature_length = X_onehot.shape[2]
    #total_activities = y_train.shape[1]    
    TOTAL_ACTIVITIES = y_index_onehot.shape[1]
    
    print('X shape:', X_onehot.shape)    
    print('y index shape:', y_index.shape)
    
    print('max sequence length:', max_sequence_length)
    print('Total activities:', TOTAL_ACTIVITIES)
    

    # 2: Generate K partitions of the dataset (KFold cross-validation)        
    # TODO: Decide between KFold or StratifiedKFold
    # if StratifiedKFold    
    skf = StratifiedKFold(n_splits=FOLDS)
        
    # if KFold
    #kf = KFold(n_splits = FOLDS)

    fold = 0
    # 4: For each partition (train, test):
    metrics_per_fold = utils.init_metrics_per_fold()

    best_epochs = []
    
    #for train, test in kf.split(X):
    for train, test in skf.split(X, y_index):
        print("%d Train: %s,  test: %s" % (fold, len(train), len(test)))        
        X_train = X_onehot[train]
        X_train_index = X[train] # Needed for ROS
        y_train = y_index_onehot[train]
        y_train_index = y_index[train]
        X_val = X_onehot[test]
        y_val = y_index_onehot[test]        
        print('Activity distribution %s' % Counter(y_index))        

        #   3.1: Build the LSTM model
        print('Building model...')
        sys.stdout.flush()
        
        model = Sequential()            
        
        model.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, action_feature_length)))            
        model.add(Dense(TOTAL_ACTIVITIES))
        model.add(Activation('softmax'))        
        model.compile(loss=LOSS, optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
        print('Model built')
        print(model.summary())
        sys.stdout.flush()        
        
        #   3.2: Manage imbalanced data in the training set (SMOTE?) -> Conf option TREAT_IMBALANCE
        # NOTE: We may have a problem with SMOTE, since there are some classes with only 1-3 samples and SMOTE needs n_samples < k_neighbors (~5)
        # NOTE: RandomOverSampler could do the trick, however it generates just copies of current samples
        # TODO: Think about a combination between RandomOverSampler for n_samples < 5 and SMOTE?
        # TODO: First attempt without imbalance management
        if(TREAT_IMBALANCE == True):
            ros = RandomOverSampler(random_state=42) # sampling_strategy={4:10, 12:10, 14:10, 8:10, 13:10}
            print('Original dataset samples for training %s' % len(y_train_index))
            print('Original dataset shape for training %s' % Counter(y_train_index))
            X_train_index_res, y_train_index_res = ros.fit_resample(X_train_index, y_train_index)
            print('Resampled dataset samples for training %s' % len(y_train_index_res))
            print('Resampled dataset shape for training %s' % Counter(y_train_index_res))
            y_train_res = np_utils.to_categorical(y_train_index_res)
            X_train_res = np_utils.to_categorical(X_train_index_res)
            
            print("y_train_res shape: ", y_train_res.shape)
        else:
            X_train_res = X_train
            y_train_res = y_train
        
        #   3.3: Train the model with the imbalance-corrected training set and use the test set to validate
        print('Training...')        
        sys.stdout.flush()
        # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
        # TODO: Do we need EarlyStopping here?
        #earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
        # TODO: improve file naming for multiple architectures
        weights_file = WEIGHTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(fold) + WEIGHTS_FILE_ROOT
        modelcheckpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0)
        callbacks = [modelcheckpoint]        
        history = model.fit(X_train_res, y_train_res, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)
        #   3.4: Store the generated learning curves and metrics with the best model (ModelCheckpoint?) -> Conf option SAVE
        plot_filename = PLOTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(fold)
        #plot_training_info(['loss'], True, history.history, plot_filename)
        if SAVE == True:
            utils.plot_training_info(['loss'], True, history.history, plot_filename)
            print("Plots saved in " + PLOTS + DATASET + '/')
        print("Training finished")
                
        # Print the best val_loss
        min_val_loss = min(history.history['val_loss'])
        min_val_loss_index = history.history['val_loss'].index(min_val_loss) 
        print("Validation loss: " + str(min_val_loss)+ " (epoch " + str(history.history['val_loss'].index(min_val_loss))+")")
        best_epochs.append(min_val_loss_index)
        model.load_weights(weights_file)
        yp = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
        # yp has the activity predictions (one-hot vectors)        
        ypreds = np.argmax(yp, axis=1)

        # Calculate the metrics        
        ytrue = np.argmax(y_val, axis=1)
        print("ytrue shape: ", ytrue.shape)
        print("ypreds shape: ", ypreds.shape)       
        
        # Plot non-normalized confusion matrix -> Conf option SAVE
        if SAVE == True:
            results_file_root = RESULTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(fold)
            utils.plot_heatmap(ytrue, ypreds, classes=activity_to_int_dict.keys(),
                              title='Confusion matrix, without normalization, fold ' + str(fold),
                              path=results_file_root + '-cm.png')

            # Plot normalized confusion matrix
            utils.plot_heatmap(ytrue, ypreds, classes=activity_to_int_dict.keys(), normalize=True,
                              title='Normalized confusion matrix, fold ' + str(fold),
                              path=results_file_root + '-cm-normalized.png')

        
        #Dictionary with the values for the metrics (precision, recall and f1)
        metrics = utils.calculate_evaluation_metrics(ytrue, ypreds)        
        metrics_per_fold = utils.update_metrics_per_fold(metrics_per_fold, metrics)
        # Update fold counter
        fold += 1

    # 5: Calculate the mean and std for the metrics obtained for each partition and store (always)    
    metrics_per_fold = utils.calculate_aggregate_metrics_per_fold(metrics_per_fold)    
    metrics_filename = RESULTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics_per_fold, fp, indent=4)
    print("Metrics saved in " + metrics_filename)
    print("Avg best epoch: " + str(np.mean(best_epochs)) + ", min: " + str(min(best_epochs)) + ", max: " + str(max(best_epochs)))
    #print(metrics_per_fold)

def print_configuration_info():
    """ Dummy function to print configuration parameters expressed as global variables in the script
    """
    print("Selected dataset:", DATASET)    
    print("Dataset base directory:", BASE_INPUT_DIR)    
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)     
    print("Selected action/activity representation:", OP)
    print("Number of epochs: ", EPOCHS)
    print("Number of folds for cross-validation: ", FOLDS)
    print("Input directory for data files:", INPUT_DIR)        
    print("Action sequences (X) file:", X_FILE)        
    print("Activity to int mappings:", ACTIVITY_TO_INT)
    print("Int to activity mappings:", INT_TO_ACTIVITY)    
    print("Experiment ID:", EXPERIMENT_ID)
    print("Treat imbalance data:", TREAT_IMBALANCE)
    print("Save intermediate plots:", SAVE)
    print("Batch size:", BATCH_SIZE)
    print("Dropout:", DROPOUT)
    print("Loss:", LOSS)    


if __name__ == "__main__":
   main(sys.argv)
