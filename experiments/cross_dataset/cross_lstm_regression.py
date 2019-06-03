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
TRAIN_DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
TEST_DATASET = 'kasterenB' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
# Directories of formatted datasets
TRAIN_BASE_INPUT_DIR = '../../formatted_datasets/' + TRAIN_DATASET + '/'
TEST_BASE_INPUT_DIR = '../../formatted_datasets/' + TEST_DATASET + '/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select imbalance data treatment
TREAT_IMBALANCE = False
# Select the number of epochs for training
EPOCHS = 10
# Select batch size
BATCH_SIZE = 512
# Select dropout value
DROPOUT = 0.8
# Select loss function
LOSS = 'cosine_proximity' # 'cosine_proximity' # 'mean_squared_error'
# END CONFIGURATION VARIABLES

# Directories where X, Y and Embedding files are stored
TRAIN_INPUT_DIR = TRAIN_BASE_INPUT_DIR + 'complete/' + DAYTIME + '_' + NONES + '/'
TEST_INPUT_DIR = TEST_BASE_INPUT_DIR + 'complete/' + DAYTIME + '_' + NONES + '/'

# File where the training  embedding matrix weights are stored to initialize the embedding layer of the network
EMBEDDING_WEIGHTS_TRAIN = TRAIN_INPUT_DIR + TRAIN_DATASET + '_' + OP + '_60_embedding_weights.npy'
# File where the testing embedding matrix weights are stored to initialize the embedding layer of the network
EMBEDDING_WEIGHTS_TEST = TEST_INPUT_DIR + TEST_DATASET + '_' + OP + '_60_embedding_weights.npy'
# File where action sequences are stored
X_TRAIN_FILE = TRAIN_INPUT_DIR + TRAIN_DATASET + '_' + OP + '_60_x.npy'
X_TEST_FILE = TEST_INPUT_DIR + TEST_DATASET + '_' + OP + '_60_x.npy'
# File where activity labels for the corresponding action sequences are stored in word embedding format (for regression)
Y_TRAIN_EMB_FILE = TRAIN_INPUT_DIR + TRAIN_DATASET + '_' + OP + '_60_y_embedding.npy'
Y_TRAIN_INDEX_FILE = TRAIN_INPUT_DIR + TRAIN_DATASET + '_' + OP + '_60_y_index.npy'
Y_TEST_EMB_FILE = TEST_INPUT_DIR + TEST_DATASET + '_' + OP + '_60_y_embedding.npy'
Y_TEST_INDEX_FILE = TEST_INPUT_DIR + TEST_DATASET + '_' + OP + '_60_y_index.npy'

# To convert the predicted embedding by the regressor to a class we need the json file with that association
TRAIN_ACTIVITY_EMBEDDINGS = TRAIN_BASE_INPUT_DIR + 'word_' + OP + '_activities.json'
TEST_ACTIVITY_EMBEDDINGS = TEST_BASE_INPUT_DIR + 'word_' + OP + '_activities.json'
# To know the indices of activity names
TRAIN_ACTIVITY_TO_INT = TRAIN_BASE_INPUT_DIR + 'activity_to_int_' + NONES + '.json'
TRAIN_INT_TO_ACTIVITY = TRAIN_BASE_INPUT_DIR + 'int_to_activity_' + NONES + '.json'
TEST_ACTIVITY_TO_INT = TEST_BASE_INPUT_DIR + 'activity_to_int_' + NONES + '.json'
TEST_INT_TO_ACTIVITY = TEST_BASE_INPUT_DIR + 'int_to_activity_' + NONES + '.json'

# ID for the experiment which is being run -> used to store the files with
# appropriate naming
# TODO: Change this to better address different experiments and models
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
EXPERIMENT_ID = 'cross_lstm-reg-' + DAYTIME + '-' + NONES + '-' + TRAIN_DATASET + '-' + TEST_DATASET

# File name for best model weights storage
WEIGHTS_FILE_ROOT = '-weights.hdf5'   


def main(argv):   
    """" Main function
    
    This is the flow of actions of this main
    0: Initial steps
    1: Load data (X and y_emb) and needed dictionaries (activity-to-int, etc.) for the training dataset
    2: Load data (X and y_emb) and needed dictionaries (activity-to-int, etc.) for the testing dataset
    3: Reformat X_train or X_test with zero padding to have the same sequence length
        3*: Concatenate both datasets' embedding matrices to use them as network input
    4: Build the LSTM model (embedding layer frozen)
    5: Test managing imbalanced data in the training set (SMOTE?)
    6: Train the model with the (imbalance-corrected) training set and use the test set to validate (TODO: consult this better)
    7: Store the generated learning curves and metrics with the best model (ModelCheckpoint? If results get worse with epochs, use EarlyStopping. Validation data?)
    8: Calculate the metrics obtained and store
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
    
    # 1: Load train data (X and y_emb)
    print('Loading train data')
    #########################################################
    # TRAIN DATA
    #########################################################
    # Load activity_dict where every activity name has its associated word embedding
    with open(TRAIN_ACTIVITY_EMBEDDINGS) as f:
        activity_dict_train = json.load(f)
    
    # Load the activity indices
    with open(TRAIN_ACTIVITY_TO_INT) as f:
        activity_to_int_dict_train = json.load(f)
    
    # Load the index to activity relations    
    with open(TRAIN_INT_TO_ACTIVITY) as f:
        int_to_activity_train = json.load(f)
    
    # Load embedding matrix, X and y sequences (for y, load both, the embedding and index version)
    embedding_matrix = np.load(EMBEDDING_WEIGHTS_TRAIN)    
    X_train = np.load(X_TRAIN_FILE)
    y_train_emb = np.load(Y_TRAIN_EMB_FILE) 
    # We need the following two lines for StratifiedKFold
    y_train_index_one_hot = np.load(Y_TRAIN_INDEX_FILE) 
    y_train_index = np.argmax(y_train_index_one_hot, axis=1)

    # To use oversampling methods in imbalance-learn, we need an activity_index:embedding relation
    # Build it using INT_TO_ACTIVITY and ACTIVITY_EMBEDDINGS files
    activity_index_to_embedding_train = {}
    for key in int_to_activity_train:
        activity_index_to_embedding_train[key] = activity_dict_train[int_to_activity_train[key]]


    max_sequence_length = X_train.shape[1] # TODO: change this to fit the maximum sequence length of all the datasets
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = embedding_matrix.shape[1]
    
    print('X train shape:', X_train.shape)
    print('y train shape:', y_train_emb.shape)
    print('y train index shape:', y_train_index.shape)
    
    print('max sequence length:', max_sequence_length)
    print('features per action:', embedding_matrix.shape[0])
    print('Action max length:', ACTION_MAX_LENGTH)
    
    # 2: Load test data (X and y_emb)
    #########################################################
    # TEST DATA
    #########################################################
    print('***************************************')
    print('Loading test data')
    # Load activity_dict where every activity name has its associated word embedding
    with open(TEST_ACTIVITY_EMBEDDINGS) as f:
        activity_dict_test = json.load(f)
    
    # Load the activity indices
    with open(TEST_ACTIVITY_TO_INT) as f:
        activity_to_int_dict_test = json.load(f)
    
    # Load the index to activity relations    
    with open(TEST_INT_TO_ACTIVITY) as f:
        int_to_activity_test = json.load(f)
    
    # Load embedding matrix
    embedding_matrix_test = np.load(EMBEDDING_WEIGHTS_TEST)
    
    # Load X and y sequences (for y, load both, the embedding and index version)
    X_test = np.load(X_TEST_FILE)
    y_test_emb = np.load(Y_TEST_EMB_FILE) 
    # We need the following two lines for StratifiedKFold
    y_test_index_one_hot = np.load(Y_TEST_INDEX_FILE) 
    y_test_index = np.argmax(y_test_index_one_hot, axis=1)

    # To use oversampling methods in imbalance-learn, we need an activity_index:embedding relation
    # Build it using INT_TO_ACTIVITY and ACTIVITY_EMBEDDINGS files
    activity_index_to_embedding_test = {}
    for key in int_to_activity_test:
        activity_index_to_embedding_test[key] = activity_dict_test[int_to_activity_test[key]]


    max_sequence_length = X_test.shape[1] # TODO: change this to fit the maximum sequence length of all the datasets
    #total_activities = y_train.shape[1]
        
    print('X test shape:', X_test.shape)
    print('y test shape:', y_test_emb.shape)
    print('y test index shape:', y_test_index.shape)
    
    print('max sequence length:', max_sequence_length)    
    print('Action max length:', ACTION_MAX_LENGTH)
    
    # 3: Reformat X_train or X_test with zero padding to have the same sequence length
    X_train, X_test = reformat_action_sequences(X_train, X_test)
    max_sequence_length = X_train.shape[1]
    print("After reformatting:")
    print('X train shape:', X_train.shape)
    print('X test shape:', X_test.shape)
    
    print('Activity distribution for training %s' % Counter(y_train_index))
    print('Activity distribution for training %s' % Counter(y_test_index))

    # 3*: Concatenate embedding matrices to be able to use both dataset actions as inputs to the network
    print("Embedding matrix train shape:", embedding_matrix.shape)
    print("Embedding matrix test shape:", embedding_matrix_test.shape)
    embedding_matrix = np.concatenate((embedding_matrix, embedding_matrix_test), axis=0)
    print("Concatenated embedding matrix shape:", embedding_matrix.shape)

    # 4: Build the LSTM model (embedding layer frozen)
    print('Building model...')
    sys.stdout.flush()
        
    model = Sequential()
    
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    # Change input shape when using embeddings
    model.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, embedding_matrix.shape[1])))    
    # For regression use a linear dense layer with embedding_matrix.shape[1] size (300 in this case)    
    model.add(Dense(embedding_matrix.shape[1]))
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    model.compile(loss=LOSS, optimizer='adam', metrics=['cosine_proximity', 'mse', 'mae'])
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
        y_train_res = []
        for j in y_train_index_res:
            y_train_res.append(activity_index_to_embedding_train[str(y_train_index_res[j])])
        y_train_res = np.array(y_train_res)
        print("y_train_res shape: ", y_train_res.shape)
    else:
        X_train_res = X_train
        y_train_res = y_train_emb
        
    #   3.3: Train the model with the imbalance-corrected training set and use the test set to validate
    print('Training...')        
    sys.stdout.flush()
    
    # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
    # TODO: Do we need EarlyStopping here?
    #earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    # TODO: improve file naming for multiple architectures
    weights_file = WEIGHTS + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-' + WEIGHTS_FILE_ROOT
    modelcheckpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [modelcheckpoint]
    history = model.fit(X_train_res, y_train_res, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test_emb), shuffle=True, callbacks=callbacks)
    #history = model.fit(X_train_res, y_train_res, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)
    #   3.4: Store the generated learning curves and metrics with the best model (ModelCheckpoint?) -> Conf option SAVE
    plot_filename = PLOTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID
    #plot_training_info(['loss'], True, history.history, plot_filename)
    utils.plot_training_info(['loss'], True, history.history, plot_filename)
    print("Plots saved in " + PLOTS)
    
    print("Training finished")
    
    # Print the best val_loss
    min_val_loss = min(history.history['val_loss'])
    print("Validation loss: " + str(min_val_loss)+ " (epoch " + str(history.history['val_loss'].index(min_val_loss))+")")        
    model.load_weights(weights_file)
    yp = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    # yp has the embedding predictions of the regressor network
    # Obtain activity labels from embedding predictions
    ypreds = obtain_class_predictions(yp, activity_dict_test, activity_to_int_dict_test, int_to_activity_test)

    # Calculate the metrics        
    ytrue = y_test_index
    print("ytrue shape: ", ytrue.shape)
    print("ypreds shape: ", ypreds.shape)    
    
    # Plot non-normalized confusion matrix -> Conf option SAVE
    
    results_file_root = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID
    utils.plot_heatmap(ytrue, ypreds, classes=activity_to_int_dict_test.keys(),
                       title='Confusion matrix, without normalization: ' + TRAIN_DATASET + '-' + TEST_DATASET,
                       path=results_file_root + '-cm.png')

    # Plot normalized confusion matrix
    utils.plot_heatmap(ytrue, ypreds, classes=activity_to_int_dict_test.keys(), normalize=True,
                       title='Normalized confusion matrix: ' + TRAIN_DATASET + '-' + TEST_DATASET,
                       path=results_file_root + '-cm-normalized.png')

        
    #Dictionary with the values for the metrics (precision, recall and f1)
    metrics = utils.calculate_evaluation_metrics(ytrue, ypreds)           
    metrics_filename = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)
    print("Metrics saved in " + metrics_filename)
    #print(metrics_per_fold)


def reformat_action_sequences(X_train, X_test):
    """As X_train and X_test may have different sequence lengths, this function pads the shortest one

    Usage exmaple:

    Parameters
    ----------
        X_train : array, shape = [n_samples1, max_sequence_length1]
                Action sequences for training
        X_test : array, shape = [n_samples2, max_sequence_length2]
                Action sequences for testing
    Returns
    -------
        X_train_ref : array, shape = [n_samples1, max{max_sequence_length1, max_sequence_length2}]
                Reformatted action sequences for training
        X_train_ref : array, shape = [n_samples2, max{max_sequence_length1, max_sequence_length2}]
                Reformatted action sequences for testing
    """
    train_seq_length = X_train.shape[1]
    test_seq_length = X_test.shape[1]
    if train_seq_length > test_seq_length:
        X_test_ref = pad_sequences(X_test, maxlen=train_seq_length, dtype='float32')
        return X_train, X_test_ref
    elif test_seq_length > train_seq_length:
        X_train_ref = pad_sequences(X_train, maxlen=test_seq_length, dtype='float32')
        return X_train_ref, X_test
    else:
        return X_train, X_test

def obtain_class_predictions(yp, activity_dict, activity_to_int_dict, int_to_activity_dict):
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
           
    Returns
    -------
        ypreds : array, shape = [n_samples, int]
            Array with the activity indices obtained from the predictions of the regressor stored in yp    
    """

    print('Transforming regression predictions to classes')

    # Simple approach: use fors and check one by one
    
    def closest_activity(pred, activity_dict):
        min_dist = 100
        activity = ""
        for key in activity_dict:
            dist = distance.cosine(pred, activity_dict[key])
            if dist < min_dist: 
                min_dist = dist
                activity = key
        return activity, min_dist

    ypred = []
    for i in xrange(len(yp)):
        activity, dist = closest_activity(yp[i], activity_dict)
        ypred.append(activity_to_int_dict[activity])

    return np.array(ypred)
    
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
    print("Train dataset base directory:", TRAIN_BASE_INPUT_DIR)
    print("Test dataset base directory:", TEST_BASE_INPUT_DIR)
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)     
    print("Selected action/activity representation:", OP)
    print("Number of epochs: ", EPOCHS)    
    print("Input directory for train data files:", TRAIN_INPUT_DIR)
    print("Input directory for test data files:", TEST_INPUT_DIR)
    print("Embedding matrix file for training:", EMBEDDING_WEIGHTS_TRAIN)
    print("Train action sequences (X) file:", X_TRAIN_FILE)
    print("Test action sequences (X) file:", X_TEST_FILE)
    print("Train label (y) file:", Y_TRAIN_EMB_FILE)
    print("Test label (y) file:", Y_TEST_EMB_FILE)
    print("Word embedding file for train activities:", TRAIN_ACTIVITY_EMBEDDINGS)    
    print("Word embedding file for test activities:", TEST_ACTIVITY_EMBEDDINGS)
    print("Activity to int mappings for training:", TRAIN_ACTIVITY_TO_INT)
    print("Int to activity mappings for training:", TRAIN_INT_TO_ACTIVITY)
    print("Activity to int mappings for testing:", TEST_ACTIVITY_TO_INT)
    print("Int to activity mappings for testing:", TEST_INT_TO_ACTIVITY)
    print("Experiment ID:", EXPERIMENT_ID)
    print("Treat imbalance data:", TREAT_IMBALANCE)    
    print("Batch size:", BATCH_SIZE)
    print("Dropout:", DROPOUT)
    print("Loss:", LOSS)    


if __name__ == "__main__":
   main(sys.argv)
