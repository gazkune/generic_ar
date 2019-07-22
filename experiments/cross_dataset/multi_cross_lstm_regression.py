# -*- coding: utf-8 -*-
"""
Created on Wed July 17 11:43:52 2019
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

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from scipy.spatial import distance

from cross_dataset_formatter import CrossDatasetFormatter
sys.path.append('..')
from utils import Utils

# BEGIN CONFIGURATION VARIABLES
# Dataset
TRAIN_DATASET = ['kasterenB', 'kasterenC', 'tapia_s1'] # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
TEST_DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia_s1'
DATASETS = TRAIN_DATASET + [TEST_DATASET]
# Directories of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'with_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
# Select segmentation period (0: perfect segmentation)
DELTA = 0
# Select imbalance data treatment
TREAT_IMBALANCE = False
# Select the number of epochs for training
EPOCHS = 300
# Select batch size
BATCH_SIZE = 512
# Select dropout value
DROPOUT = 0.5
# Select loss function
LOSS = 'cosine_proximity' # 'cosine_proximity' # 'mean_squared_error'
# Select the number of predictions to calculate
N_PREDS = 5
# END CONFIGURATION VARIABLES


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
RESULTS = 'results/'
PLOTS = 'plots/'
WEIGHTS = 'weights/'
TRAINING_STR = ""
for dataset in TRAIN_DATASET:
    TRAINING_STR += "+" + dataset
TRAINING_STR = TRAINING_STR[1:]
EXPERIMENT_ID = TRAINING_STR + '-' + TEST_DATASET + '-multi_cross_lstm-reg-' + DAYTIME + '-' + NONES

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
    print('Loading and formatting data for development')
    cdf_dev = CrossDatasetFormatter(TRAIN_DATASET, BASE_INPUT_DIR, DAYTIME, NONES, OP, DELTA)
    X_up, y_onehot_up, dev_embedding_matrix, dev_activity_to_int, dev_int_to_activity, dev_activity_to_emb = cdf_dev.reformat_datasets()

     # X sequences to train    
    X = X_up[0]
    for x in X_up[1:]:
        X = np.concatenate((X, x), axis=0)

    # y one hot to train
    y_onehot = y_onehot_up[0]
    for y in y_onehot_up[1:]: 
        y_onehot = np.concatenate((y_onehot, y), axis=0)
        
    # y indices (for auxiliary tasks)
    y_index = np.argmax(y_onehot, axis=1)
    # y embeddings
    filename = BASE_INPUT_DIR + TRAIN_DATASET[0] + '/complete/' + DAYTIME + '_' + NONES + '/' + TRAIN_DATASET[0] + '_' + OP  + '_' + str(DELTA) + '_y_embedding.npy'
    print("File name for y embedding (train): " + filename)
    y_emb = np.load(filename)
    for dataset in TRAIN_DATASET[1:]:
        filename = BASE_INPUT_DIR + dataset + '/complete/' + DAYTIME + '_' + NONES + '/' + dataset + '_' + OP  + '_' + str(DELTA) + '_y_embedding.npy'
        print("File name for y embedding (train): " + filename)
        y_emb = np.concatenate((y_emb, np.load(filename)), axis=0)
    # Common data structures for dev
    print("---------------------------------")
    print("Common data structures info for dev:")
    print("Embedding matrices:")
    print("   Individual embedding matrix shape: ")
    for emb in cdf_dev.embedding_weights:
        print("     " + str(emb.shape))    
    print("   Common embedding matrix shape: " + str(cdf_dev.common_embedding_matrix.shape))    
        
    
    # Build a development set stratifying from X and y_index    
    indices = np.arange(X.shape[0])
    _, _, _, _, train_index, dev_index = train_test_split(X, y_index, indices, test_size=0.1)
    # print(len(train_index))
    # print(len(dev_index))
    # print("Training distro: " + str(Counter(y_train_index[train_index])))
    # print("Dev distro: " + str(Counter(y_train_index[dev_index])))
    # sys.exit()

    max_sequence_length = X.shape[1]
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = dev_embedding_matrix.shape[1]
    
    print("X sequences:")
    print('   X train shape:', X[train_index].shape)
    print('   X dev shape:', X[dev_index].shape)
    print("y labels:")
    print('   y train embedding shape:', y_emb[train_index].shape)
    print('   y dev embedding shape:', y_emb[dev_index].shape)
    print('   y train one hot shape:', y_onehot[train_index].shape)
    print('   y dev one hot shape:', y_onehot[dev_index].shape)
    print('   y train index shape:', y_index[train_index].shape)
    print('   y dev index shape:', y_index[dev_index].shape)
    
    print('max sequence length: ' + str(max_sequence_length))    
    print('Action max length:', ACTION_MAX_LENGTH)
    
    # 2: Build the LSTM model (embedding layer frozen)
    print('Building model...')
    sys.stdout.flush()
        
    devmodel = Sequential()
    
    devmodel.add(Embedding(input_dim=dev_embedding_matrix.shape[0], output_dim=dev_embedding_matrix.shape[1], weights=[dev_embedding_matrix], input_length=max_sequence_length, trainable=False))
    # Change input shape when using embeddings
    devmodel.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, dev_embedding_matrix.shape[1])))
    # For regression use a linear dense layer with embedding_matrix.shape[1] size (300 in this case)    
    devmodel.add(Dense(dev_embedding_matrix.shape[1]))
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    devmodel.compile(loss=LOSS, optimizer='adam', metrics=['cosine_proximity', 'mse', 'mae'])
    print('Model built')
    print(devmodel.summary())
    sys.stdout.flush()       
    
        
    # 4: Train the model and use the dev set to validate
    print('Training...')        
    sys.stdout.flush()
    
    # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
    # TODO: Do we need EarlyStopping here?
    #earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    weights_file_dev = WEIGHTS + str(filenumber).zfill(2) + '-dev-' + EXPERIMENT_ID + WEIGHTS_FILE_ROOT
    modelcheckpoint = ModelCheckpoint(weights_file_dev, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [modelcheckpoint]
    history = devmodel.fit(X[train_index], y_emb[train_index], validation_data=(X[dev_index], y_emb[dev_index]), batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, callbacks=callbacks)   
    
    # 5: Store the generated learning curves and metrics with the best model (ModelCheckpoint?)
    plot_filename = PLOTS + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID    
    utils.plot_training_info(['loss'], True, history.history, plot_filename)
    print("Plots saved in " + PLOTS)

    # Print the best val_loss
    min_val_loss = min(history.history['val_loss'])
    min_val_loss_index = history.history['val_loss'].index(min_val_loss) # NOTE: This variable is very important for the next training!
    epochs_testing = min_val_loss_index + 1 # NOTE: We will use this variable to train the final model
    print("Validation loss: " + str(min_val_loss)+ " (epoch " + str(history.history['val_loss'].index(min_val_loss))+")")
    
    print("Development training finished")

    devmodel.load_weights(weights_file_dev)
    test_model(devmodel, X[dev_index], y_index[dev_index], cdf_dev, utils, filenumber, dev=True)    
    

    # Train a new model with the whole training set and epoch number of development experiment
    # 1: Load data (X and y_emb)
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print('Loading and formatting data for testing')
    cdf_test = CrossDatasetFormatter(DATASETS, BASE_INPUT_DIR, DAYTIME, NONES, OP, DELTA)
    X_up, y_onehot_up, common_embedding_matrix, common_activity_to_int, common_int_to_activity, common_activity_to_emb = cdf_test.reformat_datasets()

    # X sequences to train    
    X_train = X_up[0]
    for x in X_up[1:-1]: # The last element of the list is for testing
        X_train = np.concatenate((X_train, x), axis=0)

    # y one hot to train
    y_train_onehot = y_onehot_up[0]
    for y in y_onehot_up[1:-1]: # The last element of the list is for testing
        y_train_onehot = np.concatenate((y_train_onehot, y), axis=0)

    # y indices to train (for auxiliary tasks)
    y_train_index = np.argmax(y_train_onehot, axis=1)
    # y embeddings to train
    filename = BASE_INPUT_DIR + TRAIN_DATASET[0] + '/complete/' + DAYTIME + '_' + NONES + '/' + TRAIN_DATASET[0] + '_' + OP  + '_' + str(DELTA) + '_y_embedding.npy'
    print("File name for y embedding (train): " + filename)
    y_train_emb = np.load(filename)
    for dataset in TRAIN_DATASET[1:]:
        filename = BASE_INPUT_DIR + dataset + '/complete/' + DAYTIME + '_' + NONES + '/' + dataset + '_' + OP  + '_' + str(DELTA) + '_y_embedding.npy'
        print("File name for y embedding (train): " + filename)
        y_train_emb = np.concatenate((y_train_emb, np.load(filename)), axis=0)

    # X sequences to test
    X_test = X_up[-1] # -1 corresponds to TEST_DATASET
    # y one hot to test
    y_test_onehot = y_onehot_up[-1]
    # y indices to test (for auxiliary tasks)
    y_test_index = np.argmax(y_test_onehot, axis=1)
    # y embeddings to test
    filename = BASE_INPUT_DIR + TEST_DATASET + '/complete/' + DAYTIME + '_' + NONES + '/' + TEST_DATASET + '_' + OP  + '_' + str(DELTA) + '_y_embedding.npy'
    print("File name for y embedding (test): " + filename)
    y_test_emb = np.load(filename)

    max_sequence_length = X_train.shape[1]
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = common_embedding_matrix.shape[1]
    
    print("X sequences:")
    print('   X train shape:', X_train.shape)
    print('   X test shape:', X_test.shape)
    print("y labels:")
    print('   y train embedding shape:', y_train_emb.shape)
    print('   y test embedding shape:', y_test_emb.shape)
    print('   y train one hot shape:', y_train_onehot.shape)
    print('   y test one hot shape:', y_test_onehot.shape)
    print('   y train index shape:', y_train_index.shape)
    print('   y test index shape:', y_test_index.shape)
    
    print('max sequence length (train, test): ' + str(max_sequence_length) + ", " + str(X_test.shape[1]))
    print('features per action:', common_embedding_matrix.shape[0])
    print('Action max length:', ACTION_MAX_LENGTH)          
    
    train_distro_int = Counter(y_train_index)
    test_distro_int = Counter(y_test_index)
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

    # 2: Build the LSTM model (embedding layer frozen)
    print('Building model...')
    sys.stdout.flush()
        
    model = Sequential()
    
    model.add(Embedding(input_dim=common_embedding_matrix.shape[0], output_dim=common_embedding_matrix.shape[1], weights=[common_embedding_matrix], input_length=max_sequence_length, trainable=False))
    # Change input shape when using embeddings
    model.add(LSTM(512, return_sequences=False, recurrent_dropout=DROPOUT, dropout=DROPOUT, input_shape=(max_sequence_length, common_embedding_matrix.shape[1])))
    # For regression use a linear dense layer with embedding_matrix.shape[1] size (300 in this case)    
    model.add(Dense(common_embedding_matrix.shape[1]))
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    model.compile(loss=LOSS, optimizer='adam', metrics=['cosine_proximity', 'mse', 'mae'])
    print('Model built')
    print(model.summary())
    sys.stdout.flush()

    # 4: Train the model and use the dev set to validate
    print('Training...')        
    sys.stdout.flush()
    
    weights_file = WEIGHTS + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + WEIGHTS_FILE_ROOT
    modelcheckpoint = ModelCheckpoint(weights_file, monitor='loss', save_best_only=True, verbose=0)
    callbacks = [modelcheckpoint]
    history = model.fit(X_train, y_train_emb, batch_size=BATCH_SIZE, epochs=epochs_testing, shuffle=True, callbacks=callbacks)

    print("Test training finished")

    model.load_weights(weights_file)
    test_model(model, X_test, y_test_index, cdf_test, utils, filenumber, dev=False)
    print("Best development epoch: " + str(epochs_testing))

def test_model(model, X, ytrue, cdf, utils, filenumber, dev):
    """
    Function to test a model on a given input set (dev or test), obtain confusion matrices and results and store them
    Parameters
    ----------
        model : keras model            
        
        X : a numpy array representing a sequence of action indices (num_samples x action_embedding_size)            
        
        ytrue: a numpy array with true labels (activity indices)            
        
        cdf: instance of the class CrossDatasetFormatter used to frame the data
            
        utils: instance of the class Utils (to calculate and store results/metrics)

        filenumber: integer. The ID for the files to be stored.

        dev: boolean. If True, "dev" string will be used to store the results/metrics
            
           
    Returns
    -------
        None
    """
    yp = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    # yp has the embedding predictions of the regressor network
    # Obtain activity labels from embedding predictions
    if dev == True:
        ypreds = obtain_class_predictions(yp, cdf.activity_name_to_embedding, cdf.common_activity_to_int,
                                            cdf.common_int_to_activity, N_PREDS)
    else:
        ypreds = obtain_class_predictions(yp, cdf.activity_to_emb_dicts[-1], cdf.common_activity_to_int,
                                            cdf.common_int_to_activity, N_PREDS)

    # Calculate the metrics    
    print("ytrue shape: ", ytrue.shape)
    print("ypreds shape: ", ypreds.shape)    
    
    ypreds1 = ypreds[:, 0]
    # Plot non-normalized confusion matrix -> Conf option SAVE
    development = ""
    if dev == True:
        development = "dev-"
    
    results_file_root = RESULTS + str(filenumber).zfill(2) + '-' + development + EXPERIMENT_ID
    labels = [] 
    for i in cdf.common_int_to_activity:
        labels.append(cdf.common_int_to_activity[i])
    print("Classes for the heatmap (" + str(len(labels)) + ")")
    print(labels)
    utils.plot_heatmap(ytrue, ypreds1, classes=labels,
                       title='Confusion matrix, without normalization: ' + TRAINING_STR + '-' + TEST_DATASET + development,
                       path=results_file_root + '-cm.png')

    # Plot normalized confusion matrix
    utils.plot_heatmap(ytrue, ypreds1, classes=labels, normalize=True,
                       title='Normalized confusion matrix: ' + TRAINING_STR + '-' + TEST_DATASET + development,
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

    metrics_filename = RESULTS + str(filenumber).zfill(2) + '-' + development + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)
    print("Metrics saved in " + metrics_filename)    


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
                print("ERROR! Activities: " + str(activities))                
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
    print("Selected delta:", DELTA)
    print("Number of epochs: ", EPOCHS)        
    print("Experiment ID:", EXPERIMENT_ID)
    print("Treat imbalance data:", TREAT_IMBALANCE)    
    print("Batch size:", BATCH_SIZE)
    print("Dropout:", DROPOUT)
    print("Loss:", LOSS)
    print("Number of predictions:", N_PREDS)



if __name__ == "__main__":
   main(sys.argv)
