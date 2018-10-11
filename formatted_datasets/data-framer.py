# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:38:48 2017

@author: gazkune
"""

from collections import Counter
import sys
from copy import deepcopy

import pandas as pd

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from gensim.models import KeyedVectors

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np

import datetime
import json


# Directory of datasets
DIR = '../datasets/'
DATASET = 'kasterenA'
# Choose the specific dataset
CSV = DIR + DATASET + '/kasterenA_groundtruth.csv'

# ACTION_VECTORS = DIR + 'action2vec/actions_vectors.json'
# Word2Vec model
#WORD2VEC_MODEL = DIR + 'action2vec/continuous_complete_numeric_200_10.model' # d=200, win=10
WORD2VEC_MODEL = '../word_models/GoogleNews-vectors-negative300.bin.gz' # d=300

# Number of dimensions of an action vector
ACTION_DIM = 300 # Make coherent with selected WORD2VEC_MODEL

# Options for action representation
OP = 'sum' # avg is another option
DELTA = 60 # size of the sliding window for action segmentation in seconds

OUTPUT_ROOT_NAME = DATASET + '_' + OP + '_' + str(DELTA) # make coherent with WORD2VEC_MODEL

# To create training, validation and test set, we can load previously generated X and y files
READ_PREVIOUS_XY = False
PREVIOUS_X = OUTPUT_ROOT_NAME + '_x.npy'
PREVIOUS_Y = OUTPUT_ROOT_NAME + '_y.npy'

# We have to define temporal slots of a day
# For that purpose use TEMPORAL_DICT
# TODO: make it reusable for other scripts (store in a JSON?)
TEMPORAL_DICT = {'morning': {'day_change': False,'start': datetime.time(hour=9), 'end': datetime.time(hour=12)}, 
                 'afternoon': {'day_change': False, 'start': datetime.time(hour=12), 'end': datetime.time(hour=19)}, 
                 'evening': {'day_change': False, 'start': datetime.time(hour=19), 'end': datetime.time(hour=22)}, 
                 'night': {'day_change': True, 'start': datetime.time(hour=22), 'end': datetime.time(hour=9)}}

"""
Function to return the day period given the exact timestampt
Input:
    t -> pandas.tslib.Timestamp (or maybe datetime.time?) representing a time of the day
Output:
    day_period -> a string representing the moment of a day (morning, afternoon, evening, night)    
"""
def obtain_day_period(t):
    # Convert pandas.tslib.Timestamp to datetime.time
    tt = t.to_pydatetime().time()
    for key in TEMPORAL_DICT:
        if TEMPORAL_DICT[key]['day_change'] == True:
            # This day period has 00:00 inside
            if tt > TEMPORAL_DICT[key]['start'] or tt < TEMPORAL_DICT[key]['end']:
                return key
        if tt > TEMPORAL_DICT[key]['start'] and tt <= TEMPORAL_DICT[key]['end']:
            return key

"""
Function which implements the data framing to use an embedding layer
Input:
    df -> Pandas DataFrame with timestamp, action and activity
    activity_to_int -> dict with the mappings between activities and integer indices
    delta -> integer to control the segmentation of actions for sequence generation
Output:
    X -> array with action index sequences
    y -> array with activity labels as integers
    tokenizer -> instance of Tokenizer class used for action/index convertion
    max_sequence_length -> the length of a sequence

"""
def prepare_embeddings(df, activity_to_int, delta = 0):
    # Numpy array with all the actions of the dataset
    actions = df['action'].values
    print "prepare_embeddings: actions length:", len(actions)
    
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    # Very important to remove '.' and '_' from filters, since they are used
    # in action names (T003_21.5)
    tokenizer = Tokenizer(lower=False, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(actions)
    action_index = tokenizer.word_index
    print "prepare_embeddings: action_index:"
    print action_index.keys()
    
    # Build new list with action indices    
    trans_actions = np.zeros(len(actions))
    for i in xrange(len(actions)):
        #print "prepare_embeddings: action:", actions[i]        
        trans_actions[i] = action_index[actions[i]]

    #print trans_actions
    X = []
    y = []
    # Depending on delta, we generate sequences in different ways
    if delta == 0:
        # Each sequence is composed by the actions of that
        # activity instance
        current_activity = ""
        actionsdf = []
        aux_actions = []
        i = 0
        ACTIVITY_MAX_LENGTH = 0
        for index in df.index:        
            if current_activity == "":
                current_activity = df.loc[index, 'activity']
            
        
            if current_activity != df.loc[index, 'activity']:
                y.append(activity_to_int[current_activity])
                X.append(actionsdf)            
                #print current_activity, aux_actions
                current_activity = df.loc[index, 'activity']
                # reset auxiliary variables
                actionsdf = []
                aux_actions = []
        
            #print 'Current action: ', action
            actionsdf.append(np.array(trans_actions[i]))
            aux_actions.append(trans_actions[i])
            i = i + 1
        
        # Append the last activity
        y.append(activity_to_int[current_activity])
        X.append(actionsdf)
        if len(actionsdf) > ACTIVITY_MAX_LENGTH:
            ACTIVITY_MAX_LENGTH = len(actionsdf)
    else:
        
        print 'prepare_embeddings: delta value =', delta
        
        current_index = df.index[0]
        last_index = df.index[len(df) - 1]
        i = 0
        DYNAMIC_MAX_LENGTH = 0
        while current_index < last_index:
            current_time = df.loc[current_index, 'timestamp']
            #print 'prepare_embeddings: inside while', i
            #print 'prepare_embeddings: current time', current_time
            i = i + 1            
            
            """
            if i % 10 == 0:
                print '.',
            """
            actionsdf = []
            
            #auxdf = df.iloc[np.logical_and(df.index >= current_index, df.index < current_index + pd.DateOffset(seconds=delta))]
            auxdf = df.loc[np.logical_and(df.timestamp >= current_time, df.timestamp < current_time + pd.DateOffset(seconds=delta))]
            
            #print 'auxdf'
            #print auxdf
                        
            #first = df.index.get_loc(auxdf.index[0])
            first = auxdf.index[0]
            #last = df.index.get_loc(auxdf.index[len(auxdf)-1])
            last = auxdf.index[len(auxdf)-1]
            #print 'First:', first, 'Last:', last
            #actionsdf.append(np.array(trans_actions[first:last]))
            
            # TODO: using 'first' and 'last' we have to add the time period of a the day at the beginning of 'actionsdf'
            
            if first == last:
                actionsdf.append(np.array(trans_actions[first]))
            else:
                for j in xrange(first, last+1):            
                    actionsdf.append(np.array(trans_actions[j]))
            
            if len(actionsdf) > DYNAMIC_MAX_LENGTH:
                print " "
                DYNAMIC_MAX_LENGTH = len(actionsdf)
                print "MAX LENGTH =", DYNAMIC_MAX_LENGTH
                print 'First:', auxdf.loc[first, 'timestamp'], 'Last:', auxdf.loc[last, 'timestamp']
                print 'first index:', first, 'last index:', last
                print 'Length:', len(auxdf)
                #print auxdf
                #print actionsdf
                
                
            X.append(actionsdf)
            # Find the dominant activity in the time slice of auxdf
            activity = auxdf['activity'].value_counts().idxmax()
            y.append(activity_to_int[activity])
            
            # Update current_index            
            #pos = df.index.get_loc(auxdf.index[len(auxdf)-1])
            #current_index = df.index[pos+1]
            if last < last_index:
                current_index = last + 1
            else:
                current_index = last_index
            
                

    # Pad sequences
    max_sequence_length = 0
    if delta != 0:
        X = pad_sequences(X, maxlen=DYNAMIC_MAX_LENGTH, dtype='float32')
        max_sequence_length = DYNAMIC_MAX_LENGTH
    else:            
        X = pad_sequences(X, maxlen=ACTIVITY_MAX_LENGTH, dtype='float32')
        max_sequence_length = ACTIVITY_MAX_LENGTH
    
    return X, y, tokenizer, max_sequence_length

# Function to create the embedding matrix, which will be used to initialize
# the embedding layer of the network
def create_embedding_matrix(tokenizer, action_dict):    
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, ACTION_DIM))
    unknown_words = {}    
    for action, i in action_index.items():
        try:            
            embedding_vector = action_dict[action]
            embedding_matrix[i] = embedding_vector            
        except Exception as e:
            #print type(e) exceptions.KeyError
            if action in unknown_words:
                unknown_words[action] += 1
            else:
                unknown_words[action] = 1
    print "Number of unknown tokens: " + str(len(unknown_words))
    print unknown_words
    
    return embedding_matrix



def create_store_naive_datasets(X, y):
    print "Naive strategy"
    total_examples = len(X)
    train_per = 0.6
    val_per = 0.2
    # test_per = 0.2 # Not needed
    
    train_limit = int(train_per * total_examples)
    val_limit = train_limit + int(val_per * total_examples)    
    X_train = X[0:train_limit]
    X_val = X[train_limit:val_limit]
    X_test = X[val_limit:]
    y_train = y[0:train_limit]
    y_val = y[train_limit:val_limit]
    y_test = y[val_limit:]    
    print '  Total examples:', total_examples
    print '  Train examples:', len(X_train), len(y_train) 
    print '  Validation examples:', len(X_val), len(y_val)
    print '  Test examples:', len(X_test), len(y_test)
    sys.stdout.flush()  
    X_train = np.array(X_train)
    y_train = np.array(y_train)    
    print '  Activity distribution for training:'
    y_train_code = np.array([np.argmax(y_train[x]) for x in xrange(len(y_train))])
    print Counter(y_train_code)

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print '  Activity distribution for validation:'
    y_val_code = np.array([np.argmax(y_val[x]) for x in xrange(len(y_val))])
    print Counter(y_val_code)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)    

    print '  Activity distribution for testing:'
    y_test_code = np.array([np.argmax(y_test[x]) for x in xrange(len(y_test))])
    print Counter(y_test_code)

    # Save training, validation and test sets using numpy serialization
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_x_train.npy', X_train)
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_x_val.npy', X_val)
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_x_test.npy', X_test)
    
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_y_train.npy', y_train)
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_y_val.npy', y_val)
    np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_y_test.npy', y_test)
    
    print "  Formatted data saved"
    
def create_store_stratified_datasets(X, y):
    print "Stratified strategy"
    
    # Create the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # Generate indices for training and testing set
    # As sss.split() is a generator, we must use next()
    train_index, test_index = sss.split(X, y).next()
    
    # Generate X_train, y_train, X_test and y_test
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Now, generate the validation sets from the training set
    # For validation set we keep using 20% of the training data
    train_index, val_index = sss.split(X_train, y_train).next()
    X_val = X_train[val_index]
    y_val = y_train[val_index]
    X_train = X_train[train_index]
    y_train = y_train[train_index]
    
    # Print activity distributions to make sure everything is allright
    print '  X_train shape:', X_train.shape
    print '  y_train shape:', y_train.shape
    y_train_code = np.array([np.argmax(y_train[x]) for x in xrange(len(y_train))])
    print '  Activity distribution for training:'    
    print Counter(y_train_code)
    print '  X_val shape:', X_val.shape
    print '  y_val shape:', y_val.shape
    y_val_code = np.array([np.argmax(y_val[x]) for x in xrange(len(y_val))])
    print '  Activity distribution for training:'    
    print Counter(y_val_code)
    print '  X_test shape:', X_test.shape
    print '  y_test shape:', y_test.shape
    y_test_code = np.array([np.argmax(y_test[x]) for x in xrange(len(y_test))])
    print '  Activity distribution for training:'    
    print Counter(y_test_code)
    
    # Save the generated datasets in the corresponding files
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_x_train.npy', X_train)
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_x_val.npy', X_val)
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_x_test.npy', X_test)
    
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_y_train.npy', y_train)
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_y_val.npy', y_val)
    np.save(OUTPUT_ROOT_NAME + '_stratified_' + str(DELTA) + '_y_test.npy', y_test)
    
def sum_action_representation(action, model):
    # Function to represent an action suming the embeddings of constituente words
    words = action.split('_')
    embedding = np.zeros(ACTION_DIM) 
    for word in words:
        if word != 'to': # word 'to' is not in the model (??)
            embedding = embedding + model[word]
    
    return embedding


def avg_action_representation(action, model):
    # Function to represent an action averaging the embeddings of constituente words
    words = action.split('_')
    embedding = np.zeros(ACTION_DIM) 
    for word in words:
        if word != 'to': # word 'to' is not in the model (??)
            embedding = embedding + model[word]
    
    embedding = embedding / len(words)
    return embedding
        
    
def build_action_representation(df, model):
    # Translate actions to neural embeddings depending on the value of variable OP (sum, avg)
    # For each action we have to separate conforming words split by '_'
    # translate those words to word vectors using the WORD_MODEL and represent
    # the final embedding depending on OP
    # For that purpose, we build a dict where each action name has its n-d embedding 
    

    # Numpy array with unique actions of the dataset
    actions = df['action'].unique()
    
    # Dict for action-embedding relationship
    action_dict = {}
    
    for action in actions:
        embedding = np.zeros(ACTION_DIM)
        if OP == 'sum':
            embedding = sum_action_representation(action, model)
        if OP == 'avg':
            embedding = avg_action_representation(action, model)
        
        action_dict[action] = embedding
        
    return action_dict
        
    

# Main function
def main(argv):
    
    if READ_PREVIOUS_XY == False:
        # Load dataset from csv file
        df = pd.read_csv(CSV, parse_dates=[[0, 1]], header=None, sep=' ')        
        df.columns = ['timestamp', 'sensor', 'action', 'event', 'activity']    
    
        #df = df[0:1000] # reduce dataset for tests    
        unique_activities = df['activity'].unique()
        print "Unique activities:"
        print unique_activities

        total_activities = len(unique_activities)        
    
        # Generate the dict to transform activities to integer numbers
        activity_to_int = dict((c, i) for i, c in enumerate(unique_activities))
        # Generate the dict to transform integer numbers to activities
        int_to_activity = dict((i, c) for i, c in enumerate(unique_activities))
    
        # TODO: save those two dicts in a file
        with open(DATASET+"_activity_to_int.json", 'w') as fp:
            json.dump(activity_to_int, fp, indent=4)
        
        with open(DATASET+"_int_to_activity.json", 'w') as fp:
            json.dump(int_to_activity, fp, indent=4)        
        
        print df.head(10)        
        
        # Translate actions to neural embeddings depending on the value of variable OP (sum, avg)
        # For each action we have to separate conforming words split by '_'
        # translate those words to word vectors using the WORD_MODEL and represent
        # the final embedding depending on OP
        # For that purpose, we build a dict where each action name has its n-d embedding
        # First of all, load WORD_MODEL
        print "Loading", WORD2VEC_MODEL, "model"
        model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)
        print "Model loaded"
        
        # action_dict holds a word vector (dependind on OP variable) for each action in df
        action_dict = build_action_representation(df, model)
        
        # TODO: Generate temporal representations
        # function obtain_day_period(t) returns a day period in TEMPORAL_DICT given a Pandas timestamp
                
        # Prepare sequences using action indices
        # Each action will be an index which will point to an action vector
        # in the weights matrix of the Embedding layer of the network input
        # Use 'delta' to establish slicing time; if 0, slicing done on activity type basis    
        X, y, tokenizer, max_sequence_length = prepare_embeddings(df, activity_to_int, delta=DELTA)
    
        # Create the embedding matrix for the embedding layer initialization
        embedding_matrix = create_embedding_matrix(tokenizer)    
    
        print 'max sequence length:', max_sequence_length
        print 'X shape:', X.shape
    
        print 'embedding matrix shape:', embedding_matrix.shape
    
    
    
        # Keep original y (with activity indices) before transforming it to categorical
        y_orig = deepcopy(y)
        # Tranform class labels to one-hot encoding
        y = np_utils.to_categorical(y)
        print 'y shape:', y.shape
    
        # Save X, y and embedding_matrix using numpy serialization
        np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_x.npy', X)
        np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_y.npy', y)
        np.save(OUTPUT_ROOT_NAME + '_' + str(DELTA) + '_embedding_weights.npy', embedding_matrix)
        
    else:
    
        X = np.load(PREVIOUS_X)
        y= np.load(PREVIOUS_Y)
        # Prepare training, validation and testing datasets
        # We implement two strategies for this:    
        # 1: Naive datasets, using only the percentages
        # This strategy preserves the original sequences and time dependencies
        # It can be useful for stateful LSTMs
        #createStoreNaiveDatasets(X, y)
    
        # 2: Stratified datasets, making sure all three sets have the same percentage of classes
        # This strategy may break the time dependencies amongst sequences
        create_store_stratified_datasets(X, y)
    
    
    
if __name__ == "__main__":
   main(sys.argv)
    