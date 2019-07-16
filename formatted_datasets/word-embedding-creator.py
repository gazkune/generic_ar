#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 07:55:44 2018

@author: gazkune
"""

from collections import Counter
import sys
from copy import deepcopy

import pandas as pd

from gensim.models import KeyedVectors

import numpy as np
import datetime
import json

# Directory of datasets
DIR = '../datasets/'
DATASET = 'tapia_s1' # Currently: 'kasterenA', 'kasterenB', 'kasterenC', 'tapia_s1'
# Choose the specific dataset: '/kasterenA_groundtruth.csv', '/kasterenB_groundtruth.csv', '/kasterenC_groundtruth.csv', '/mit_s1-m.csv'
CSV = DIR + DATASET + '/mit_s1-m.csv'

# Word2Vec model
WORD2VEC_MODEL = '../word_models/GoogleNews-vectors-negative300.bin.gz' # d=300

# Number of dimensions of an action vector
ACTION_DIM = 300 # Make coherent with selected WORD2VEC_MODEL

# Options for action representation
OP = 'sum' # 'sum' and 'avg' are the current options

# Option to use the location name of a given sensor for its representation (specially designed for Tapia dataset, where
# many sensors share the same name but use an int ID to distinguish their location)
USE_LOCATION = True
LOCATION_FILE = DIR + DATASET + '/sensors_s1.csv'

# Root name for output files
OUTPUT_ROOT = 'word_' + OP + '_'

# We have to define temporal slots of a day
# For that purpose use TEMPORAL_DICT
# TODO: make it reusable for other scripts (store in a JSON?)
TEMPORAL_DICT = {'morning': {'day_change': False,'start': datetime.time(hour=9), 'end': datetime.time(hour=12)}, 
                 'afternoon': {'day_change': False, 'start': datetime.time(hour=12), 'end': datetime.time(hour=19)}, 
                 'evening': {'day_change': False, 'start': datetime.time(hour=19), 'end': datetime.time(hour=22)}, 
                 'night': {'day_change': True, 'start': datetime.time(hour=22), 'end': datetime.time(hour=9)}}

# List of word to be ignored
IGNORE_WORD_LIST = ['to', 'pir', '-', 'a']

def is_integer(word):
    # Function to chech whether a given string represents an integer number
    try:
        int(word)
        return True
    except ValueError:
        return False
        
def sum_action_representation(action, model, locations_df = None):
    # Function to represent an action/activity suming the embeddings of constituente words
    words = action.split('_')
    embedding = np.zeros(ACTION_DIM)
    sensor_id = -1 # specially thought for Tapia datasets
    for word in words:
        word = word.lower()
        if word not in IGNORE_WORD_LIST: # word 'to' is not in the model (??); the word 'pir' has a totally different meaning
            if word == 'cutlary': # KasterenB contains 'cutlary' when it should be 'cutlery'
                embedding = embedding + model['cutlery']
            elif is_integer(word):
                if USE_LOCATION == True:
                    print("Use location of " + word)
                    # TODO: use sensors_s1.csv file to assign locations to sensors
                    sensor_id = int(word)

            else:   
                embedding = embedding + model[word]
        
    if USE_LOCATION == True and sensor_id > 0:        
        auxdf = locations_df.loc[locations_df["id"] == sensor_id]
        location = auxdf.iloc[0]["location"]
        # A location in sensors_s1 is "Office/study" -> Code to cope with it (and similar situations)
        location = location.split("/")[0]
        embedding = embedding + model[location]

    return embedding


def avg_action_representation(action, model, locations_df = None):
    # Function to represent an action/activity averaging the embeddings of constituente words
    words = action.split('_')
    embedding = np.zeros(ACTION_DIM) 
    for word in words:
        if word not in IGNORE_WORD_LIST: # word 'to' is not in the model (??); the word 'pir' has a totally different meaning
            if word == 'cutlary': # KasterenB contains 'cutlary' when it should be 'cutlery'
                embedding = embedding + model['cutlery']
            elif is_integer(word):
                if USE_LOCATION == True:
                    print("Use location of " + word)
                    # TODO: use sensors_s1.csv file to assign locations to sensors
            else:   
                embedding = embedding + model[word]
    
    embedding = embedding / len(words)
    return embedding
        
    
def build_action_representation(df, model, locations_df = None):
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
            embedding = sum_action_representation(action, model, locations_df).tolist() # in order to serialize with JSON
        if OP == 'avg':
            embedding = avg_action_representation(action, model, locations_df).tolist() # in order to serialize with JSON
        
        action_dict[action.lower()] = embedding
        
    return action_dict


def build_temporal_representation(model):
    # Dict for tempral-embedding relationship
    temporal_dict = {}
    
    for key in TEMPORAL_DICT:
        embedding = model[key]
        temporal_dict[key.lower()] = embedding.tolist() # in order to serialize with JSON
        
    return temporal_dict

def build_activity_representation(unique_activities, model):
    # Dict for activity-embedding relationship
    activity_dict = {}
    
    for activity in unique_activities:
        embedding = np.zeros(ACTION_DIM)
        if OP == 'sum':
            embedding = sum_action_representation(activity, model).tolist() # in order to serialize with JSON
        if OP == 'avg':
            embedding = avg_action_representation(activity, model).tolist() # in order to serialize with JSON
        
        activity_dict[activity.lower()] = embedding
        
    return activity_dict
    
# Main function
def main(argv):
    # Load dataset from csv file
    df = pd.read_csv(CSV, parse_dates=[[0, 1]], header=None, sep=' ')        
    df.columns = ['timestamp', 'sensor', 'action', 'event', 'activity']    
    
    #df = df[0:1000] # reduce dataset for tests    
    unique_activities = df['activity'].unique()
    print "Unique activities:"
    print unique_activities

    total_activities = len(unique_activities)
  
    print df.head(10)

    locations_df = None
    if USE_LOCATION == True:
        locations_df = pd.read_csv(LOCATION_FILE, header=None, sep=',')
        locations_df.columns = ['id', 'location', 'sensor']        

        
    # Translate actions to neural embeddings depending on the value of variable OP (sum, avg)
    # For each action we have to separate conforming words split by '_'
    # translate those words to word vectors using the WORD_MODEL and represent
    # the final embedding depending on OP
    # For that purpose, we build a dict where each action name has its n-d embedding
    # First of all, load WORD_MODEL
    print "Loading", WORD2VEC_MODEL, "model"
    model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)
    print "Model loaded"
        
    # action_dict holds a word vector (depending on OP variable) for each action in df
    action_dict = build_action_representation(df, model, locations_df)
    
    # Build another dictionary for temporal concepts
    temporal_dict = build_temporal_representation(model)
    
    # Build another dictionary for activity representations (depending on OP variable)
    activity_dict = build_activity_representation(unique_activities, model)
    
    # Store the three dictionaries as json files
    with open(DATASET+'/'+OUTPUT_ROOT+"actions.json", 'w') as fp:
        json.dump(action_dict, fp, indent=4)
        print DATASET+'/'+OUTPUT_ROOT+"actions.json stored"
        
    with open(DATASET+'/'+OUTPUT_ROOT+"temporal.json", 'w') as fp:
        json.dump(temporal_dict, fp, indent=4)
        print DATASET+'/'+OUTPUT_ROOT+"temporal.json stored"
        
    with open(DATASET+'/'+OUTPUT_ROOT+"activities.json", 'w') as fp:
        json.dump(activity_dict, fp, indent=4)
        print DATASET+'/'+OUTPUT_ROOT+"activities.json stored"
        
        
        
if __name__ == "__main__":
   main(sys.argv)