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

from utils import Utils

# BEGIN CONFIGURATION VARIABLES
# Dataset
DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
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
TREAT_IMBALANCE = True
# Select the number of epochs for training
EPOCHS = 300
# Select batch size
BATCH_SIZE = 256
# Select dropout value
DROPOUT = 0.1
# Select loss function
LOSS = 'categorical_crossentropy' # 

# Select whether intermediate plots and results should be saved
SAVE = False
# END CONFIGURATION VARIABLES

# Directory where X, Y and Embedding files are stored
INPUT_DIR = BASE_INPUT_DIR + 'complete/' + DAYTIME + '_' + NONES + '/'

# File where the embedding matrix weights are stored to initialize the embedding layer of the network
EMBEDDING_WEIGHTS = INPUT_DIR + DATASET + '_' + OP + '_60_embedding_weights.npy'
# File where action sequences are stored
X_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_x.npy'
# File where activity labels for the corresponding action sequences are stored in word embedding format (for regression)
#Y_EMB_FILE = INPUT_DIR + DATASET + '_' + OP + '_60_y_embedding.npy'
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
EXPERIMENT_ID = 'lstm-class-' + DAYTIME + '-' + NONES

# File name for best model weights storage
WEIGHTS_FILE_ROOT = '_lstm-classification-weights.hdf5'   


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
    #y_emb = np.load(Y_EMB_FILE)
    # We need the following two lines for StratifiedKFold
    y_index_one_hot = np.load(Y_INDEX_FILE) 
    y_index = np.argmax(y_index_one_hot, axis=1)

    # To use oversampling methods in imbalance-learn, we need an activity_index:embedding relation
    # Build it using INT_TO_ACTIVITY and ACTIVITY_EMBEDDINGS files
    activity_index_to_embedding = {}
    for key in int_to_activity:
        activity_index_to_embedding[key] = activity_dict[int_to_activity[key]]


    max_sequence_length = X.shape[1] # TODO: change this to fit the maximum sequence length of all the datasets
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = embedding_matrix.shape[1]
    TOTAL_ACTIVITIES = y_index_one_hot.shape[1]
    
    print('X shape:', X.shape)    
    print('y index shape:', y_index.shape)
    
    print('max sequence length:', max_sequence_length)
    print('features per action:', embedding_matrix.shape[0])
    print('Action max length:', ACTION_MAX_LENGTH)     
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
    
    #for train, test in kf.split(X):
    for train, test in skf.split(X, y_index):
        print("%d Train: %s,  test: %s" % (fold, len(train), len(test)))        
        X_train = X[train]
        y_train = y_index_one_hot[train]
        y_train_index = y_index[train]
        X_val = X[test]
        y_val = y_index_one_hot[test]        
        print('Activity distribution %s' % Counter(y_index))        

        #   3.1: Build the LSTM model
        print('Building model...')
        sys.stdout.flush()
        
        model = Sequential()
    
        model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
        # Change input shape when using embeddings
        model.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, embedding_matrix.shape[1])))    
        # For regression use a linear dense layer with embedding_matrix.shape[1] size (300 in this case)
        # TODO: consider the need of normalization before calculating the loss (we may use a Lambda layer with L2 norm)        
        model.add(Dense(TOTAL_ACTIVITIES))
        model.add(Activation('softmax'))
        # TODO: check different regression losses; cosine_proximity could be the best one for us?        
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
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
            X_train_res, y_train_index_res = ros.fit_resample(X_train, y_train_index)
            print('Resampled dataset samples for training %s' % len(y_train_index_res))
            print('Resampled dataset shape for training %s' % Counter(y_train_index_res))
            y_train_res = np_utils.to_categorical(y_train_index_res)
            
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
        print("Validation loss: " + str(min_val_loss)+ " (epoch " + str(history.history['val_loss'].index(min_val_loss))+")")        
        model.load_weights(weights_file)
        yp = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
        # yp has the activity predictions (one-hot vectors)        
        ypreds = np.argmax(yp, axis=1)

        # Calculate the metrics        
        ytrue = np.argmax(y_val, axis=1)
        print("ytrue shape: ", ytrue.shape)
        print("ypreds shape: ", ypreds.shape)
    
        # Use scikit-learn metrics to calculate confusion matrix, accuracy, precision, recall and F-Measure
        """
        cm = confusion_matrix(ytrue, ypreds)
    
        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=3, linewidth=1000, suppress=True)
        
        # Save also the cm to a txt file
        results_file_root = RESULTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(fold)
        np.savetxt(results_file_root + '-cm.txt', cm, fmt='%.0f')   
        
        np.savetxt(results_file_root+'-cm-normalized.txt', cm_normalized, fmt='%.3f')
        print("Confusion matrices saved in " + RESULTS + DATASET + '/')
        """
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
    print("Embedding matrix file:", EMBEDDING_WEIGHTS)
    print("Action sequences (X) file:", X_FILE)    
    print("Word embedding file for activities:", ACTIVITY_EMBEDDINGS)    
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