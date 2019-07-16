# -*- coding: utf-8 -*-
"""
Created on Tue Jun 04 09:15:52 2019
@author: gazkune
"""

from __future__ import print_function

from collections import Counter
from collections import defaultdict
import sys
import os

from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from nltk.stem import PorterStemmer

import numpy as np
import json
import random


class CrossDatasetFormatter:
    def __init__(self, datasets, base_input_dir, daytime, nones, op, delta):
        """Constructor        
        """
        # Attribute to decide the use of word stemming
        self.stem = False

        self.datasets = datasets
        self.base_input_dir = base_input_dir
        self.daytime = daytime
        self.nones = nones
        self.op = op
        self.delta = delta
        # Load X files
        self.X_sequences = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_' + str(delta) + '_x.npy'            
            self.X_sequences.append(np.load(filename))       

        # Load y files (only index)
        self.y_onehot = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_' + str(delta) + '_y_index.npy'            
            self.y_onehot.append(np.load(filename))       

        # Load embedding files
        self.embedding_weights = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_' + str(delta) + '_embedding_weights.npy'
            print("File name: " + filename)
            self.embedding_weights.append(np.load(filename))       

        # Load activity_to_int and int_to_activity files
        self.activity_to_int_dicts = []
        self.int_to_activity_dicts = []
        for dataset in datasets:
            filename_ai = base_input_dir + dataset + '/activity_to_int_' + nones + '.json'
            filename_ia = base_input_dir + dataset + '/int_to_activity_' + nones + '.json'            
            with open(filename_ai) as f:
                self.activity_to_int_dicts.append(json.load(f))
            with open(filename_ia) as f:
                self.int_to_activity_dicts.append(json.load(f))

        # To use oversampling methods in imbalance-learn, we need an activity_index:embedding relation
        # For that purpose, load individual activity to emb dicts (filenames "word_sum_activities.json")
        self.activity_to_emb_dicts = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/word_' + op + '_activities.json'
            with open(filename) as f:
                self.activity_to_emb_dicts.append(json.load(f))

        # Load action_to_int and int_to_action files
        self.action_to_int_dicts = []
        self.int_to_action_dicts = []
        for dataset in datasets:
            filename_ai = base_input_dir + dataset + '/action_to_int_' + nones + '.json'
            filename_ia = base_input_dir + dataset + '/int_to_action_' + nones + '.json'            
            with open(filename_ai) as f:
                self.action_to_int_dicts.append(json.load(f))
            with open(filename_ia) as f:
                self.int_to_action_dicts.append(json.load(f))

    def reformat_datasets(self):
        """ Function to reformat datasets, which means:
        1) build_common_activity_to_int_dict
        2) build_common_action_to_int_dict
        2) build_common_embedding_matrix
        3) update_x_sequences
        4) update_y_onehot
        5) build common activity name to embedding dict
        """
        print("Reformatting datasets for cross dataset usage")
        self.build_common_activity_to_int_dict()
        self.build_common_action_to_int_dict()
        self.build_common_embedding_matrix()
        self.update_x_sequences()        
        self.update_y_onehot()
        self.build_common_activity_to_emb()

        return self.X_seq_updated, self.y_onehot_updated, self.common_embedding_matrix, self.common_activity_to_int, self.common_int_to_activity, self.activity_index_to_embedding


    def set_stemmer(self, stem):
        self.stem = stem


    def build_common_activity_to_int_dict(self):
        """Function to fuse individual activity_to_int dicts
        """
        self.common_activity_to_int = {}
        index = 0
        counter = 0        
        for i in range(len(self.activity_to_int_dicts)):
            for key in self.activity_to_int_dicts[i].keys():
                #print("Activity: " + key)
                counter += 1
                if not key in self.common_activity_to_int:
                    self.common_activity_to_int[key] = index
                    index += 1
                
        
        # Now build the int_to_activity dict
        self.common_int_to_activity = {v: k for k, v in self.common_activity_to_int.iteritems()}
        
        if self.stem == True:
            # We will also build a stemmed activity to int dict
            porter = PorterStemmer()
            self.stemmed_activity_to_int = {}
            index = 0
            counter = 0        
            for i in range(len(self.activity_to_int_dicts)):
                for key in self.activity_to_int_dicts[i].keys():
                    #print("Activity: " + key)
                    counter += 1
                    # Build the stemmed key
                    words = key.split("_")
                    new_key = ""
                    for word in words:
                        new_key += "_" + porter.stem(word)

                    # Remove first character ('_')
                    new_key = new_key[1:]
                    if not new_key in self.stemmed_activity_to_int:
                        self.stemmed_activity_to_int[new_key] = index
                        index += 1
                
        
            # Now build the int_to_activity dict
            self.stemmed_int_to_activity = {v: k for k, v in self.stemmed_activity_to_int.iteritems()}


    def save_common_activity_int_dicts(self, folder):
        """Function to save the generated dictionaries in json format
        """
        datasets = ""
        for dataset in self.datasets:
            datasets = datasets + "_" + dataset

        # Store common activity to int and int to activity
        with open(folder + datasets + "_activity_to_int_"+ self.nones +".json", 'w') as fp:
            json.dump(self.common_activity_to_int, fp, indent=4)
        
        with open(folder + datasets + "_int_to_activity_" + self.nones + ".json", 'w') as fp:
            json.dump(self.common_int_to_activity, fp, indent=4)

        if self.stem == True:
            # Store stemmed activity to int and int to activity
            with open(folder + datasets + "_stemmed_activity_to_int_"+ self.nones +".json", 'w') as fp:
                json.dump(self.stemmed_activity_to_int, fp, indent=4)
        
            with open(folder + datasets + "_stemmed_int_to_activity_" + self.nones + ".json", 'w') as fp:
                json.dump(self.stemmed_int_to_activity, fp, indent=4)

    def build_common_action_to_int_dict(self):
        """Function to fuse individual action_to_int dicts
        """
        self.common_action_to_int = {}
        index = 1 # Remember the 0 position of embedding matrix is for unknown actions
        for i in range(len(self.action_to_int_dicts)):        
            for key in self.action_to_int_dicts[i].keys():                
                if not key in self.common_action_to_int:
                    self.common_action_to_int[key] = index
                    index += 1    
        
        # Now build the int_to_activity dict
        self.common_int_to_action = {v: k for k, v in self.common_action_to_int.iteritems()}
        
    def save_common_action_int_dicts(self, folder):
        """Function to save the generated dictionaries in json format
        """
        datasets = ""
        for dataset in self.datasets:
            datasets = datasets + "_" + dataset

        with open(folder + datasets + "_action_to_int_"+ self.nones +".json", 'w') as fp:
            json.dump(self.common_action_to_int, fp, indent=4)
        
        with open(folder + datasets + "_int_to_action_" + self.nones + ".json", 'w') as fp:
            json.dump(self.common_int_to_action, fp, indent=4)
    

    def build_common_embedding_matrix(self):
        """Function to fuse all embbeding matrices
        Take into account that all individual embedding matrices have the first element reserved for unknown words (0 vector)
        NOTE: this approach repeats embedding vectors for repeated actions in differente datasets (e.g. the embedding for
        "front_door" may appear twice)
        """
        self.common_embedding_matrix = self.embedding_weights[0]        
        for i in range(1, len(self.embedding_weights)):           
            self.common_embedding_matrix = np.concatenate((self.common_embedding_matrix, self.embedding_weights[i][1:]), axis=0)    


    def update_x_sequences(self):
        """Function to update X_sequences with the new indices of common embedding matrix and the new sequence length
        extracted from the maximum length of the datasets
        """
        # First step: recalculate the indices of X_sequences to match the new common_embedding_matrix
        # The first X of the list is already correct
        # All the 0s should remain 0, since they refer to the unknown word of the embedding matrix
        self.X_seq_updated = []
        i = 0
        displacement = 0
        for x in self.X_sequences:            
            # Generate a mask for values greater than 0
            boolean_mask = x > 0
            mask = boolean_mask.astype(int)
            # Calculate the displacement using embedding_weights shape        
            if i > 0:            
                displacement += self.embedding_weights[i-1].shape[0] - 1 # The number of rows - 1
            
            x_updated = x + displacement*mask
            i += 1
            self.X_seq_updated.append(x_updated)

        # Second step: use the maximum length of X_sequences (max number of actions per activity), to pad all the other sequences
        maxlen = 0
        for x in self.X_sequences:
            maxlen = max(maxlen, x.shape[1])
            
        for i in range(len(self.X_seq_updated)):        
            self.X_seq_updated[i] = pad_sequences(self.X_seq_updated[i], maxlen=maxlen, dtype='float32')
    
    def test_view_x_seq_updated(self):
        """Function to test and view whether X_seq_updated is well formatted
        """
        for i in range(len(self.X_sequences)):
            X = self.X_sequences[i]
            X_up = self.X_seq_updated[i]
            sample = random.randint(0, X.shape[0]-1)
            print("Sample of X " + str(i))
            print(X[sample])
       
            for j in X[sample]:
                j = j.astype(int)
                if j != 0:
                    emb_matrix = self.embedding_weights[i]
                    emb = emb_matrix[j]
                    print("Corresponding embedding:")
                    print(emb)
       
            print("Sample of X updated " + str(i))
            print(X_up[sample])
       
            for k in X_up[sample]:
                k = k.astype(int)
                if k != 0:                
                    emb = self.common_embedding_matrix[k]
                    print("Corresponding embedding:")
                    print(emb)
    
    
    def update_x_sequences_no_rep(self):
        """Function to update X_sequences with the new indices of common embedding matrix and the new sequence length
        extracted from the maximum length of the datasets
        NOTE: this approach cannot be used with the common embedding matrix, since it is designed to work with one-hot encoding,
        avoiding action repetion (it uses the same index for the same actions across datasets)
        """
        self.X_seq_no_rep = []
        dataset = 0
        for x in self.X_sequences:
            x_updated = []
            for seq in x:
                # This a 1-d array with "old" action indices
                seq_updated = []
                for action_index in seq:
                    if action_index != 0:
                        action_name = self.int_to_action_dicts[dataset][str(int(action_index))]
                        new_index = self.common_action_to_int[action_name]
                    else:
                        new_index = 0
                    seq_updated.append(new_index)

                
                x_updated.append(seq_updated)

            self.X_seq_no_rep.append(np.array(x_updated))
            dataset += 1
        
        maxlen = 0
        for x in self.X_sequences:
            maxlen = max(maxlen, x.shape[1])
            
        for i in range(len(self.X_seq_no_rep)):
            self.X_seq_no_rep[i] = pad_sequences(self.X_seq_no_rep[i], maxlen=maxlen, dtype='float32')       
        
    def test_view_x_seq_no_rep(self):
        """Function to test and view whether X_seq_no_rep is well formatted
        """
        for i in range(len(self.X_sequences)):
            X = self.X_sequences[i]
            X_up = self.X_seq_no_rep[i]
            sample = random.randint(0, X.shape[0]-1)
            print("Sample of X " + str(i))
            print(X[sample])
       
            for j in X[sample]:
                j = j.astype(int)
                if j != 0:                    
                    print(self.int_to_action_dicts[i][str(j)], end=" | ")                    
       
            print("")
            print("Sample of X updated " + str(i))
            print(X_up[sample])
       
            for k in X_up[sample]:
                k = k.astype(int)
                if k != 0:                
                    print(self.common_int_to_action[k], end=" | ")
            
            print("")


    def update_y_onehot(self):
        """Function to update the y_onehot for all datasets, taking into account the common activities found in them
        NOTE: y_onehot are one-hot vectors
        """
        num_classes = len(self.common_activity_to_int.keys())
        self.y_onehot_updated = []
        for i in range(len(self.y_onehot)):            
            y = self.y_onehot[i]
            int_to_activity = self.int_to_activity_dicts[i]            
            y_up = []
            for one_hot in y:            
                index = np.argmax(one_hot, axis=0)            
                new_index = self.common_activity_to_int[int_to_activity[str(index)]]
                y_up.append(new_index)
        
            # TODO: Test the following list comprehension instead of the second for loop
            # y_up = [common_activity_to_int[int_to_activity[index]] for index in y]
            y_up = np.array(y_up) 
            self.y_onehot_updated.append(y_up)
    
        # At this point, all y_up in y_onehot_updated are numeric indices of activities -> Convert to categorical
        for i in range(len(self.y_onehot_updated)):
            self.y_onehot_updated[i] = np_utils.to_categorical(self.y_onehot_updated[i], num_classes=num_classes)

        if self.stem == True:
            porter = PorterStemmer()
            num_classes = len(self.stemmed_activity_to_int.keys())
            self.y_stemmed_onehot_updated = []
            for i in range(len(self.y_onehot)):
                y = self.y_onehot[i]
                int_to_activity = self.int_to_activity_dicts[i]            
                y_up = []
                for one_hot in y:
                    index = np.argmax(one_hot, axis=0)
                    orig_activity = int_to_activity[str(index)]
                    stemmed_activity = ""
                    words = orig_activity.split("_")
                    for word in words:
                        stemmed_activity += "_" + porter.stem(word)

                    stemmed_activity = stemmed_activity[1:]
                    new_index = self.stemmed_activity_to_int[stemmed_activity]
                    y_up.append(new_index)
        
                # TODO: Test the following list comprehension instead of the second for loop
                # y_up = [common_activity_to_int[int_to_activity[index]] for index in y]
                y_up = np.array(y_up) 
                self.y_stemmed_onehot_updated.append(y_up)
    
            # At this point, all y_up in y_onehot_updated are numeric indices of activities -> Convert to categorical
            for i in range(len(self.y_stemmed_onehot_updated)):
                self.y_stemmed_onehot_updated[i] = np_utils.to_categorical(self.y_stemmed_onehot_updated[i], num_classes=num_classes)

    def test_view_y_onehot(self):
        for i in range(len(self.y_onehot_updated)):
            print("Dataset " + str(i))
            y_up = self.y_onehot_updated[i]
            y = self.y_onehot[i]
            sample = random.randint(0, y.shape[0]-1)
            print("Original activity: " + self.int_to_activity_dicts[i][str(np.argmax(y[sample]))])
            print(y[sample])
            print("Updated activity: " + self.common_int_to_activity[np.argmax(y_up[sample])])
            print(y_up[sample])

    def build_common_activity_to_emb(self):
        """Function to build a common activity index to embedding dictionary
        """
        self.activity_index_to_embedding = {}
        for ae_dict in self.activity_to_emb_dicts:
            for key in ae_dict.keys():
                try:
                    index = self.common_activity_to_int[key]
                    if index not in self.activity_index_to_embedding:
                        self.activity_index_to_embedding[index] = ae_dict[key]
                    else:
                        print("Activity " + key + " is already in the dict")
                except KeyError:
                    if key.lower() == 'none':
                        print(key + " activity ignored")
                    else:
                        print(key + " activity not in the common activity to int dict. Halting process!")
                        sys.exit(-1)

    def test_view_activity_to_emb(self, num_samples):
        for i in range(len(self.activity_to_emb_dicts)):
            ae_dict = self.activity_to_emb_dicts[i]
            ai_dict = self.activity_to_int_dicts[i]
            ia_dict = self.int_to_activity_dicts[i]
            print("Dataset " + str(i))
            for j in range(num_samples):
                sample = random.randint(0, len(ai_dict.keys())-1)
                activity = ia_dict[str(sample)]
                orig_emb = ae_dict[activity]
                new_emb = self.activity_index_to_embedding[self.common_activity_to_int[activity]]
                print("   Sample " + str(j) + " activity " + activity)
                print("   Original embedding:")
                print(orig_emb)
                print("   New embedding:")
                print(new_emb)
        

"""
Main function used to test the functionality of CrossDatasetFormatter class
"""

def main(argv):
    # List of datasets to reformat and fuse
    DATASETS = ['kasterenA', 'tapia_s1']
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

    cross_dataset_formatter = CrossDatasetFormatter(DATASETS, BASE_INPUT_DIR, DAYTIME, NONES, OP, DELTA)
    cross_dataset_formatter.set_stemmer(True)
    keyslist = []
    for d in cross_dataset_formatter.action_to_int_dicts:
        keyslist.append(d.keys())
        print("-------------------------")
        print(d)
    common_actions = set(keyslist[0]).intersection(*keyslist)
    print("common actions:")
    print(common_actions)   
    
    X_seq_up, y_onehot_up, common_embedding_matrix, common_activity_to_int, common_int_to_activity, common_activity_to_emb = cross_dataset_formatter.reformat_datasets()
    print("common action to int:")
    print(cross_dataset_formatter.common_action_to_int)
    print("common int to action:")
    print(cross_dataset_formatter.common_int_to_action)
        
    print("View reformatted datasets' information")
    print("Common activity_to_int (" + str(len(common_activity_to_int.keys())) +"):")
    print(common_activity_to_int)
    print("Common int_to_activity:")
    print(common_int_to_activity)
    print("--------------------------------------")
    print("Stemmed activity_to_int (" + str(len(cross_dataset_formatter.stemmed_activity_to_int.keys())) +"):")
    print(cross_dataset_formatter.stemmed_activity_to_int)
    print("Stemmed int_to_activity:")
    print(cross_dataset_formatter.stemmed_int_to_activity)
    print("--------------------------------------")
    print("common embedding matrix shape: ", common_embedding_matrix.shape)

    # New method update_x_sequences_no_rep
    cross_dataset_formatter.update_x_sequences_no_rep()
    print("--------------------------------------")
    i = 0
    for x_seq in cross_dataset_formatter.X_seq_no_rep:
        print("X seq no rep shape for dataset " + str(i) + ": " + str(cross_dataset_formatter.X_seq_no_rep[i].shape))
        i += 1
    
    cross_dataset_formatter.test_view_x_seq_no_rep()    

    print("--------------------------------------")
    print("Test and view X sequences updated samples:")
    cross_dataset_formatter.test_view_x_seq_updated()
    print("--------------------------------------")
    print("Testing y indices")
    cross_dataset_formatter.test_view_y_onehot()
    print("--------------------------------------")
    print("Testing activity to embedding")    
    cross_dataset_formatter.test_view_activity_to_emb(3)
    

if __name__ == "__main__":
   main(sys.argv)