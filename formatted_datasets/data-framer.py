# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:38:48 2017

@author: gazkune
"""
from __future__ import print_function
import sys
from copy import deepcopy

import pandas as pd

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np

import datetime
import json
import random


# Directory of original datasets 
DIR = '../datasets/'
DATASET = 'tapia'
# Choose the specific dataset
#CSV = DIR + DATASET + '/' + DATASET + '_groundtruth.csv'
CSV = DIR + DATASET + '/' + 'mit_s1-m.csv'

# Number of dimensions of an action vector
WORD_DIM = 300 # Make coherent with selected WORD2VEC_MODEL

# Option for action representation
OP = 'sum' # 'sum' and 'avg' are the current options

# Option for sequence segmentation using its duration
DELTA = 60 # size of the sliding window for action segmentation in seconds (Kasteren et al use 60 seconds)

# Option to include (or not) day time period in each of the sequences (ex: [morning, frontdoor, frontdoor])
DAYTIME = 'with_time' # select between 'no_time' and 'with_time'

# Option to include (or not) activities of type 'None' 
NONES = 'no_nones' # select between 'no_nones' and 'with_nones'

OUTPUT_DIR = DATASET + '/complete/' + DAYTIME + '_' + NONES
SUBDESCR = "s1" # As Tapia dataset has two subjects, we use this variable to distinguish (for the rest of datasets use "")
OUTPUT_ROOT_NAME = ""
if SUBDESCR == "":
    OUTPUT_ROOT_NAME = DATASET + '_' + OP + '_' + str(DELTA)
else:
    OUTPUT_ROOT_NAME = DATASET + '_' + SUBDESCR + '_' + OP + '_' + str(DELTA)

# We have to define temporal slots of a day
# For that purpose use TEMPORAL_DICT
# TODO: make it reusable for other scripts (store in a JSON?)
TEMPORAL_DICT = {'morning': {'day_change': False,'start': datetime.time(hour=9), 'end': datetime.time(hour=12)}, 
                 'afternoon': {'day_change': False, 'start': datetime.time(hour=12), 'end': datetime.time(hour=19)}, 
                 'evening': {'day_change': False, 'start': datetime.time(hour=19), 'end': datetime.time(hour=22)}, 
                 'night': {'day_change': True, 'start': datetime.time(hour=22), 'end': datetime.time(hour=9)}}

"""
Function to return the day period given the exact timestamp
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
    y_index -> array with activity labels as integers
    y_embedding -> array with activity labels as word embeddings
    tokenizer -> instance of Tokenizer class used for action/index convertion
    max_sequence_length -> the length of a sequence

"""
def prepare_embeddings(df, action_dict, activity_dict, temporal_dict, activity_to_int, delta = 0):
    # Numpy array with all the actions of the dataset    
    actions = df['action'].values
    print("prepare_embeddings: actions length:", len(actions))
    
    # We have to add also the day-time periods inside temporal_dict
    dayperiods = np.array(temporal_dict.keys())
    words = np.append(actions, dayperiods)
    
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    # Very important to remove '.' and '_' from filters, since they are used
    # in action names (plates_cupboard)
    tokenizer = Tokenizer(lower=False, filters='!"#$%&()*+,/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(words)
    word_index = tokenizer.word_index
    print("prepare_embeddings: action_index:")
    print(word_index.keys())
    
    # Build new list with action indices
    # The sequences to be trained will be action sequences
    # Here we transform actions to indices using the word_index, where
    # day periods are also present. However, day periods will be integrated
    # afterwards (every action sequence will have a day period)    
    trans_actions = np.zeros(len(actions))
    for i in range(len(actions)):
        #print "prepare_embeddings: action:", actions[i]        
        trans_actions[i] = word_index[actions[i]]

    #print trans_actions
    
    # X is for action sequences and their day period
    X = []
    
    # y is for activity labels
    # we have two y: 
    # 1) y_index, where activity labels are stored as activity indices
    # 2) y_embedding: activities as stored using ther word embeddings (from activity_dict)
    y_index = []
    y_embedding = []
    
    # Depending on delta, we generate sequences in different ways
    if delta == 0:
        # TODO: this piece of code is taken from another problem -> review it before using!!!
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
                
                y_index.append(activity_to_int[current_activity])
                y_embedding.append(activity_dict[current_activity])
                
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
        y_index.append(activity_to_int[current_activity])
        y_embedding.append(activity_dict[current_activity])
        X.append(actionsdf)
        if len(actionsdf) > ACTIVITY_MAX_LENGTH:
            ACTIVITY_MAX_LENGTH = len(actionsdf)
    else:
        
        print("prepare_embeddings: delta value = ", delta)
        
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
            
            # Using 'first' and 'last' we have to add the time period of a day at the beginning of 'actionsdf'
            first_period = obtain_day_period(auxdf.loc[first, 'timestamp'])
            last_period = obtain_day_period(auxdf.loc[last, 'timestamp'])
            action_seq_period = ''
            if first_period == last_period:                
                action_seq_period = first_period
            else:
                # decide day period for this action sequence with majority voting                
                action_seq_period = auxdf['dayperiod'].value_counts().idxmax()                
            
            if first == last:
                actionsdf.append(np.array(trans_actions[first]))
            else:
                for j in xrange(first, last+1):            
                    actionsdf.append(np.array(trans_actions[j]))
            
            if action_seq_period == '':
                print("prepare_embeddings: problem computing day period of an action sequence")
                print(auxdf)
                sys.exit()
                
            # Append the day period for the action sequence depending on the DAYTIME variable
            # We have to append the index generated by word_index
            if DAYTIME == 'with_time':
                actionsdf.append(word_index[action_seq_period])                
            
            if len(actionsdf) > DYNAMIC_MAX_LENGTH:
                print(" ")
                DYNAMIC_MAX_LENGTH = len(actionsdf)
                print("MAX LENGTH = ", DYNAMIC_MAX_LENGTH)
                print("First: ", auxdf.loc[first, 'timestamp'], "Last: ", auxdf.loc[last, 'timestamp'])
                print("first index: ", first, "last index: ", last)
                print("Length: ", len(auxdf))
                #print auxdf
                #print actionsdf
                
                
            X.append(actionsdf)
            
            # There are two strategies to store the activity of a given action sequence:
            # 1: store the activity index (as it's already done)
            # 2: store the word embedding of the activity using activity_dict (for regression problems)
            
            # Implementation of strategy 1
            # Find the dominant activity in the time slice of auxdf
            activity = auxdf['activity'].value_counts().idxmax()
            y_index.append(activity_to_int[activity])
            
            # Implementation of strategy 2
            # Use the dominant activity label in the action sequence, as in strategy 1
            y_embedding.append(activity_dict[activity])
            
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
    
    return X, y_index, y_embedding, tokenizer, max_sequence_length

# Function to create the embedding matrix, which will be used to initialize
# the embedding layer of the network
def create_embedding_matrix(tokenizer, action_dict, temporal_dict):    
    # TODO: We currently add the day period words to the Embedding matrix. Check whether this has any influence on
    # when we are not using them for action sequence generation (in principle I assume it won't have any effect)
    element_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(element_index) + 1, WORD_DIM))
    unknown_words = {}    
    for element, i in element_index.items():
        if element in action_dict:
            embedding_vector = action_dict[element]
            embedding_matrix[i] = embedding_vector 
        elif element in temporal_dict:
            embedding_vector = temporal_dict[element]
            embedding_matrix[i] = embedding_vector
        else:
            if element in unknown_words:
                unknown_words[element] += 1
            else:
                unknown_words[element] = 1       
        
    print("Number of unknown tokens: " + str(len(unknown_words)))
    print(unknown_words)
    
    return embedding_matrix


def test_data_framing(num_samples, X, y, tokenizer, int_to_activity):
    print(" ")
    print("########################")
    print("Testing X and y: ")
    num_samples = 10
    for i in range(num_samples):
        print("Test ", i)
        sample = random.randint(0, X.shape[0]-1)
        print("Sample number: ", sample)
        seq = X[sample]
        print("Sequence: ", seq)
        for element in seq:
            if int(element) != 0:
                # 0 is introduced in the padding process, so do not print it
                print(tokenizer.word_index.keys()[tokenizer.word_index.values().index(int(element))], end=" ")
        
        print("| Activity: ", int_to_activity[y[sample]])
        print(".................................")


def print_configuration_info():
    print("Selected dataset: ", DATASET)
    print("Selected operation for action/activity representation: ", OP)
    print("Selected delta for sequence segmentation: ", DELTA)
    print("Daytime: ", DAYTIME)
    print("Nones: ", NONES)
    print("Output directory: ", OUTPUT_DIR)
    print("Output file root name: ", OUTPUT_ROOT_NAME)


# Main function
def main(argv):

    # Print some 'configuration' info just in case
    print_configuration_info()    
    
    # Load dataset from csv file
    df = pd.read_csv(CSV, parse_dates=[[0, 1]], header=None, sep=' ')        
    df.columns = ['timestamp', 'sensor', 'action', 'event', 'activity']    
    
    #df = df[0:1000] # reduce dataset for tests
    # Remove None type activities depending on the value of the variable NONES
    if NONES == 'no_nones':
        df = df.loc[np.logical_and(df['activity'] != 'None', df['activity'] != 'none')]
        df.reset_index(inplace=True, drop=True) 

    unique_activities = df['activity'].unique()
    print("Unique activities:")
    print(unique_activities)

    total_activities = len(unique_activities)    
    
    # Generate the dict to transform activities to integer numbers
    activity_to_int = dict((c, i) for i, c in enumerate(unique_activities))
    # Generate the dict to transform integer numbers to activities
    int_to_activity = dict((i, c) for i, c in enumerate(unique_activities))
    
    # Save those two dicts in a file
    with open(DATASET+"/activity_to_int_"+NONES+".json", 'w') as fp:
        json.dump(activity_to_int, fp, indent=4)
        
    with open(DATASET+"/int_to_activity_"+NONES+".json", 'w') as fp:
        json.dump(int_to_activity, fp, indent=4)                 
    
    # Load word embeddings and store them in action_dict, activity_dict and temporal_dict
    with open(DATASET+'/word_'+OP+'_actions.json') as f:
        action_dict = json.load(f)
        
    with open(DATASET+'/word_'+OP+'_activities.json') as f:
        activity_dict = json.load(f)
        
    with open(DATASET+'/word_'+OP+'_temporal.json') as f:
        temporal_dict = json.load(f)
            
    # Remember that the embeddings in those dictionaries are of type list 
    # convert them to numpy arrays
               
    # Generate temporal representations
    # function obtain_day_period(t) returns a day period in TEMPORAL_DICT given a Pandas timestamp
    # we will add a new column to df to store the day period of a given action
    dayperiods = []
    for i in df.index:
        p = obtain_day_period(df.loc[i, 'timestamp'])
        dayperiods.append(p)
    
    df['dayperiod'] = dayperiods
    
    print(df.head(10))    
    
    # Prepare sequences using action indices
    # Each action will be an index which will point to an action vector
    # in the weights matrix of the Embedding layer of the network input
    # Use 'delta' to establish slicing time; if 0, slicing done on activity type basis    
    X, y_index, y_embedding, tokenizer, max_sequence_length = prepare_embeddings(df, action_dict, activity_dict, temporal_dict, activity_to_int, delta=DELTA)
    
    # Create the embedding matrix for the embedding layer initialization
    embedding_matrix = create_embedding_matrix(tokenizer, action_dict, temporal_dict)
    
    print("tokenizer:")
    print(tokenizer.word_index)
    action_to_int = tokenizer.word_index
    int_to_action = {v: k for k, v in action_to_int.iteritems()}
    print(int_to_action)

    # Save those two dicts in a file
    with open(DATASET+"/action_to_int_"+NONES+".json", 'w') as fp:
        json.dump(action_to_int, fp, indent=4)
        
    with open(DATASET+"/int_to_action_"+NONES+".json", 'w') as fp:
        json.dump(int_to_action, fp, indent=4)    
    
    print("max sequence length: ", max_sequence_length)
    print("X shape: ", X.shape)
    
    print("embedding matrix shape: ", embedding_matrix.shape)
    
    # Keep original y (with activity indices) before transforming it to categorical
    y_orig = deepcopy(y_index)
    # Tranform class labels to one-hot encoding
    y_index = np_utils.to_categorical(y_index)
    y_embedding = np.array(y_embedding)
    print("y_index shape: ", y_index.shape)
    print("y_embedding shape: ", y_embedding.shape)
    
    
    # test some samples from X and y to see whether they make sense or not
    test_data_framing(10, X, y_orig, tokenizer, int_to_activity)
    
    # Save X, y and embedding_matrix using numpy serialization
    np.save(OUTPUT_DIR + '/' + OUTPUT_ROOT_NAME + '_x.npy', X)
    np.save(OUTPUT_DIR + '/' + OUTPUT_ROOT_NAME + '_y_index.npy', y_index)
    np.save(OUTPUT_DIR + '/' + OUTPUT_ROOT_NAME + '_y_embedding.npy', y_embedding)
    np.save(OUTPUT_DIR + '/' + OUTPUT_ROOT_NAME + '_embedding_weights.npy', embedding_matrix)
    
if __name__ == "__main__":
   main(sys.argv)
    