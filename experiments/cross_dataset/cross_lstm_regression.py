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

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE, RandomOverSampler

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
EPOCHS = 1
# Select batch size
BATCH_SIZE = 512
# Select dropout value
DROPOUT = 0.8
# Select loss function
LOSS = 'cosine_proximity' # 'cosine_proximity' # 'mean_squared_error'
# END CONFIGURATION VARIABLES


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

    # X sequences to train
    X_train = X_seq_up[0] # 0 corresponds to TRAIN_DATASET
    # y one hot to train
    y_train_onehot = y_onehot_up[0]
    # y indices to train (for auxiliary tasks)
    y_train_index = np.argmax(y_train_onehot, axis=1)
    # y embeddings to train
    filename = BASE_INPUT_DIR + TRAIN_DATASET + '/complete/' + DAYTIME + '_' + NONES + '/' + TRAIN_DATASET + '_' + OP  + '_60_y_embedding.npy'
    print("File name for y embedding (train): " + filename)
    y_train_emb = np.load(filename)

    # X sequences to test
    X_test = X_seq_up[1] # 1 corresponds to TEST_DATASET
    # y one hot to test
    y_test_onehot = y_onehot_up[1]
    # y indices to test (for auxiliary tasks)
    y_test_index = np.argmax(y_test_onehot, axis=1)
    # y embeddings to test
    filename = BASE_INPUT_DIR + TEST_DATASET + '/complete/' + DAYTIME + '_' + NONES + '/' + TEST_DATASET + '_' + OP  + '_60_y_embedding.npy'
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
        
    #3: Manage imbalanced data in the training set (SMOTE?) -> Conf option TREAT_IMBALANCE
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
            y_train_res.append(common_activity_to_emb[str(y_train_index_res[j])])
        y_train_res = np.array(y_train_res)
        print("y_train_res shape: ", y_train_res.shape)
    else:
        X_train_res = X_train
        y_train_res = y_train_emb
        
    # 4: Train the model with the imbalance-corrected training set and use the test set to validate
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
    
    # 5: Store the generated learning curves and metrics with the best model (ModelCheckpoint?) -> Conf option SAVE
    plot_filename = PLOTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID    
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
    #ypreds = obtain_class_predictions(yp, activity_dict_test, activity_to_int_dict_test, int_to_activity_test)
    ypreds = obtain_class_predictions(yp, cross_dataset_formatter.activity_to_emb_dicts[1], 
                                        cross_dataset_formatter.common_activity_to_int, cross_dataset_formatter.common_int_to_activity)

    # Calculate the metrics        
    ytrue = y_test_index
    print("ytrue shape: ", ytrue.shape)
    print("ypreds shape: ", ypreds.shape)    
    
    # Plot non-normalized confusion matrix -> Conf option SAVE
    
    results_file_root = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID
    labels = cross_dataset_formatter.common_activity_to_int.keys()
    print("Classes for the heatmap (" + str(len(labels)) + ")")
    print(labels)
    utils.plot_heatmap(ytrue, ypreds, classes=labels,
                       title='Confusion matrix, without normalization: ' + TRAIN_DATASET + '-' + TEST_DATASET,
                       path=results_file_root + '-cm.png')

    # Plot normalized confusion matrix
    utils.plot_heatmap(ytrue, ypreds, classes=labels, normalize=True,
                       title='Normalized confusion matrix: ' + TRAIN_DATASET + '-' + TEST_DATASET,
                       path=results_file_root + '-cm-normalized.png')

        
    #Dictionary with the values for the metrics (precision, recall and f1)
    metrics = utils.calculate_evaluation_metrics(ytrue, ypreds)           
    metrics_filename = RESULTS + str(filenumber).zfill(2) + '-' + TRAIN_DATASET + '-' + TEST_DATASET + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics, fp, indent=4)
    print("Metrics saved in " + metrics_filename)
    #print(metrics_per_fold)



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
    for i in range(len(yp)):
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


if __name__ == "__main__":
   main(sys.argv)
