# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:30:52 2018
@author: gazkune
"""
from __future__ import print_function

from collections import Counter
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from scipy.spatial import distance

# BEGIN CONFIGURATION VARIABLES
# Dataset
DATASET = 'kasterenA' # Select between 'kasterenA', 'kasterenB', 'kasterenC' and 'tapia'
# Directory of formatted datasets
BASE_INPUT_DIR = '../../formatted_datasets/' + DATASET + '/'
# Select between 'with_time' and 'no_time'
DAYTIME = 'no_time'
# Select between 'with_nones' and 'no_nones'
NONES = 'no_nones'
# Select between 'avg' and 'sum' for action/activity representation
OP = 'sum'
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
EXPERIMENT_ID = DAYTIME + '-' + NONES


# File name for best model weights storage
WEIGHTS_FILE_ROOT = '_lstm-regression-weights.hdf5'

    
def load_model(model_file, weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    
def check_activity_distribution(y_np, unique_activities):
    activities = []
    for activity_np in y_np:
        index = activity_np.tolist().index(1.0)
        activities.append(unique_activities[index])
    print(Counter(activities))
    
    
"""
Function to plot accurary and loss during training
"""

def plot_training_info(metrics, save, history, filename):
    # summarize history for accuracy
    if 'accuracy' in metrics:
        
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        lgd = plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(filename + '-accuracy.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        lgd = plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(filename + '-loss.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
        

    
def calculate_evaluation_metrics(y_gt, y_preds):
    
    """Calculates the evaluation metrics (precision, recall and F1) for the
    predicted examples. It calculates the micro, macro and weighted values
    of each metric.
            
    Usage example:
        y_gt = ['make_coffe', 'brush_teeth', 'wash_hands']
        y_preds = ['make_coffe', 'wash_hands', 'wash_hands']
        metrics = calculate_evaluation_metrics (y_ground_truth, y_predicted)
        
    Parameters
    ----------
        y_gt : array, shape = [n_samples]
            Classes that appear in the ground truth.
        
        y_preds: array, shape = [n_samples]
            Predicted classes. Take into account that the must follow the same
            order as in y_ground_truth
           
    Returns
    -------
        metric_results : dict
            Dictionary with the values for the metrics (precision, recall and 
            f1)    
    """
        
    metric_types =  ['micro', 'macro', 'weighted']
    metric_results = {
        'precision' : {},
        'recall' : {},
        'f1' : {},
        'acc' : -1.0        
    }
            
    for t in metric_types:
        metric_results['precision'][t] = metrics.precision_score(y_gt, y_preds, average = t)
        metric_results['recall'][t] = metrics.recall_score(y_gt, y_preds, average = t)
        metric_results['f1'][t] = metrics.f1_score(y_gt, y_preds, average = t)
        metric_results['acc'] = metrics.accuracy_score(y_gt, y_preds) 
                
    return metric_results


#def check_activities_train_test(y_train_code, y_test_code):

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
    print("Selected dataset:", DATASET)    
    print("Dataset base directory:", BASE_INPUT_DIR)    
    print("Daytime option:", DAYTIME)    
    print("Nones option:", NONES)     
    print("Selected action/activity representation:", OP)    
    print("Input directory for data files:", INPUT_DIR)    
    print("Embedding matrix file:", EMBEDDING_WEIGHTS)
    print("Action sequences (X) file:", X_FILE)
    print("Label (y) file:", Y_EMB_FILE)
    print("Word embedding file for activities:", ACTIVITY_EMBEDDINGS)    
    print("Activity to int mappings:", ACTIVITY_TO_INT)
    print("Int to activity mappings:", INT_TO_ACTIVITY)    
    print("Experiment ID:", EXPERIMENT_ID)

# Main function
def main(argv):   
    # This is the flow of actions of this main
    # 0: Initial steps
    # 1: Load data (X and y_emb) and needed dictionaries (activity-to-int, etc.)    
    # 2: Generate K partitions of the dataset (KFold cross-validation)
    # 3: For each partition (train, test):
    #   3.1: Build the LSTM model
    #   3.2: Manage imbalanced data in the training set (SMOTE?)
    #   3.3: Train the model with the imbalance-corrected training set and use the test set to validate
    #   3.4: Store the best the generated learning curves and metrics with the best model (ModelCheckpoint? 
    #           If results get worse with epochs, use EarlyStopping)
    # 4: Calculate the mean and std for the metrics obtained for each partition and store

    # 0: Initial steps
    print_configuration_info()        
    # fix random seed for reproducibility
    np.random.seed(7)

    # Obtain the file number
    contents = os.listdir(RESULTS + DATASET + '/')
    maxnumber = -1
    for f in contents:
        parts = f.split('-')
        if int(parts[0]) > maxnumber:
            maxnumber = int(parts[0])

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
    y_emb = np.load(Y_EMB_FILE)
    # We need the following two lines for StratifiedKFold
    y_index_one_hot = np.load(Y_INDEX_FILE) 
    y_index = np.argmax(y_index_one_hot, axis=1)      

    max_sequence_length = X.shape[1] # TODO: change this to fit the maximum sequence length of all the datasets
    #total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = embedding_matrix.shape[1]
    
    print('X shape:', X.shape)
    print('y shape:', y_emb.shape)
    print('y index shape:', y_index.shape)
    
    print('max sequence length:', max_sequence_length)
    print('features per action:', embedding_matrix.shape[0])
    print('Action max length:', ACTION_MAX_LENGTH)     

    # 2: Generate K partitions of the dataset (KFold cross-validation)        
    # TODO: Decide between KFold or StratifiedKFold
    # if StratifiedKFold
    """
    skf = StratifiedKFold(n_splits=5)
    i = 0
    for train, test in skf.split(X, y_index):
        print("%d Train: %s,  test: %s" % (i, len(train), len(test)))
        i = i + 1
    """
    # if KFold
    kf = KFold(n_splits = 10)
    i = 0
    # 4: For each partition (train, test):
    metrics_per_fold = {} # This dict is to store the metrics of each fold
    # Initialize the dictionary with empty lists
    metric_names = ['precision', 'recall', 'f1']
    metric_variants = ['micro', 'weighted', 'macro']
    metrics_per_fold['acc'] = []

    for metric in metric_names:
        metrics_per_fold[metric] = {}
        for variant in metric_variants:
            metrics_per_fold[metric][variant] = []            
    
    for train, test in kf.split(X):
        print("%d Train: %s,  test: %s" % (i, len(train), len(test)))        
        X_train = X[train]
        y_train = y_emb[train]
        X_val = X[test]
        y_val = y_emb[test]
        y_val_index = y_index_one_hot[test]

        #   3.1: Build the LSTM model
        print('Building model...')
        sys.stdout.flush()
        batch_size = 1024 # TODO: Check the actual number of samples for training!
        model = Sequential()
    
        model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
        # Change input shape when using embeddings
        model.add(LSTM(512, return_sequences=False, recurrent_dropout=0.8, dropout=0.8, input_shape=(max_sequence_length, embedding_matrix.shape[1])))    
        # For regression use a linear dense layer with embedding_matrix.shape[1] size (300 in this case)
        # TODO: consider the need of normalization before calculating the loss (we may use a Lambda layer with L2 norm)
        model.add(Dense(embedding_matrix.shape[1]))
        # TODO: check different regression losses; cosine_proximity could be the best one for us? 
        # NOTE: accuracy makes no sense in this regression scenario
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
        print('Model built')
        print(model.summary())
        sys.stdout.flush()

        
        #   3.2: Manage imbalanced data in the training set (SMOTE?)
        # TODO: First attempt without imbalance management

        #   3.3: Train the model with the imbalance-corrected training set and use the test set to validate
        print('Training...')        
        sys.stdout.flush()
        # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
        # TODO: Do we need EarlyStopping here?
        #earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
        # TODO: improve file naming for multiple architectures
        weights_file = WEIGHTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(i) + WEIGHTS_FILE_ROOT
        modelcheckpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0)
        callbacks = [modelcheckpoint]
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=300, validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)
        #   3.4: Store the generated learning curves and metrics with the best model (ModelCheckpoint?)
        plot_filename = PLOTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(i)
        plot_training_info(['loss'], True, history.history, plot_filename)
        print("Training finished")
        print("Plots saved in " + PLOTS + DATASET + '/')
        
        # Print the best val_loss    
        print('Validation loss:', min(history.history['val_loss']))
        model.load_weights(weights_file)
        yp = model.predict(X_val, batch_size=batch_size, verbose=1)
        # yp has the embedding predictions of the regressor network
        # Obtain activity labels from embedding predictions
        ypreds = obtain_class_predictions(yp, activity_dict, activity_to_int_dict, int_to_activity)

        # Calculate the metrics
        # TODO: tidy up the following code!!!
        #print(ypreds)
        #print("y_val shape:", y_val.shape)
        #print(y_val)
        #print("y_val activity indices:")
        #ytrue = np.array(y_orig[val_limit:]) # the original way
        ytrue = np.argmax(y_val_index, axis=1)
        #print(ytrue)

        # Verify y_true and y_preds activities
        #print("Unique activities in y_true: ", np.unique(ytrue))
        #print("Unique activities in y_preds: ", np.unique(ypreds))
    
        # Use scikit-learn metrics to calculate confusion matrix, accuracy, precision, recall and F-Measure    
        cm = confusion_matrix(ytrue, ypreds)
    
        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=3, linewidth=1000, suppress=True)
        #print('Confusion matrix')
        #print(cm)
        # Save also the cm to a txt file
        results_file_root = RESULTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-fold' + str(i)
        np.savetxt(results_file_root + '-cm.txt', cm, fmt='%.0f')
    
        #print('Normalized confusion matrix')
        #print(cm_normalized)
        np.savetxt(results_file_root+'-cm-normalized.txt', cm_normalized, fmt='%.3f')
        print("Confusion matrices saved in " + RESULTS + DATASET + '/')
    
        #Dictionary with the values for the metrics (precision, recall and f1)    
        metrics = calculate_evaluation_metrics(ytrue, ypreds)
        metrics_per_fold['acc'].append(metrics['acc'])
        for metric in metric_names:
            for variant in metric_variants:
                metrics_per_fold[metric][variant].append(metrics[metric][variant])

        #print("Scikit metrics")
        #print('accuracy: ', metrics['acc'])
        #print('precision:', metrics['precision'])
        #print('recall:', metrics['recall'])
        #print('f1:', metrics['f1'])

        # Update fold counter
        i += 1

    # 5: Calculate the mean and std for the metrics obtained for each partition and store
    metrics_per_fold['mean_acc'] = np.mean(np.array(metrics_per_fold['acc']))
    metrics_per_fold['std_acc'] = np.std(np.array(metrics_per_fold['acc']))    
    for metric in metric_names:
        metrics_per_fold['mean_' + metric] = {}
        metrics_per_fold['std_' + metric] = {}
        for variant in metric_variants:
            metrics_per_fold['mean_' + metric][variant] = np.mean(np.array(metrics_per_fold[metric][variant]))
            metrics_per_fold['std_' + metric][variant] = np.std(np.array(metrics_per_fold[metric][variant]))

    metrics_filename = RESULTS + DATASET + '/' + str(filenumber).zfill(2) + '-' + EXPERIMENT_ID + '-complete-metrics.json'
    with open(metrics_filename, 'w') as fp:
        json.dump(metrics_per_fold, fp, indent=4)
    print("Metrics saved in " + metrics_filename)
    #print(metrics_per_fold)

if __name__ == "__main__":
   main(sys.argv)