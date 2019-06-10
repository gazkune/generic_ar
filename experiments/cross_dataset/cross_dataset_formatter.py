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

import numpy as np
import json
import random


class CrossDatasetFormatter:
    def __init__(self, datasets, base_input_dir, daytime, nones, op):
        """Constructor        
        """
        # Load X files
        self.X_sequences = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_60_x.npy'
            print("File name: " + filename)
            self.X_sequences.append(np.load(filename))        
    
        print("X_sequences length: " + str(len(self.X_sequences)))
        for x in self.X_sequences:
            print("   X shape: " + str(x.shape))

        # Load y files (only index)
        self.y_indices = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_60_y_index.npy'
            print("File name: " + filename)
            self.y_indices.append(np.load(filename))
    
        print("y_indices length: " + str(len(self.y_indices)))
        for y in self.y_indices:
            print("   y shape: " + str(y.shape))

        # Load embedding files
        self.embedding_weights = []
        for dataset in datasets:
            filename = base_input_dir + dataset + '/complete/' + daytime + '_' + nones + '/' + dataset + '_' + op  + '_60_embedding_weights.npy'
            print("File name: " + filename)
            self.embedding_weights.append(np.load(filename))

        print("embedding_weights length: " + str(len(self.embedding_weights)))
        for e in self.embedding_weights:
            print("   Embedding matrix shape: " + str(e.shape))

        # Load activity_to_int and int_to_activity files
        self.activity_to_int_dicts = []
        self.int_to_activity_dicts = []
        for dataset in datasets:
            filename_ai = base_input_dir + dataset + '/activity_to_int_' + nones + '.json'
            filename_ia = base_input_dir + dataset + '/int_to_activity_' + nones + '.json'
            print("File name activity to int: " + filename_ai)
            print("File name int to activity: " + filename_ia)
            with open(filename_ai) as f:
                self.activity_to_int_dicts.append(json.load(f))
            with open(filename_ia) as f:
                self.int_to_activity_dicts.append(json.load(f))

    def reformat_datasets(self):
        """ Function to reformat datasets, which means:
        1) build_common_activity_to_int_dict
        2) build_common_embedding_matrix
        3) update_x_sequences
        4) update_y_indices
        """
        print("Reformatting datasets for cross dataset usage")
        self.build_common_activity_to_int_dict()
        self.build_common_embedding_matrix()
        self.update_x_sequences()
        self.update_y_indices()

        return self.X_seq_updated, self.y_indices_updated, self.common_embedding_matrix, self.common_activity_to_int, self.common_int_to_activity


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
    
        print("Original number of activities: " + str(counter))
        print("Fused number of activities: " + str(index))
        # Now build the int_to_activity dict
        self.common_int_to_activity = {}
        for key in self.common_activity_to_int:
            newkey = self.common_activity_to_int[key]
            self.common_int_to_activity[newkey] = key

    def build_common_embedding_matrix(self):
        """Function to fuse all embbeding matrices
        Take into account that all individual embedding matrices have the first element reserved for unknown words (0 vector)
        """
        self.common_embedding_matrix = self.embedding_weights[0]
        #print("common embedding matrix shape: ", common_embedding_matrix.shape)
        for i in range(1, len(self.embedding_weights)):
           #print("Matrix " + str(i) + " shape: " + str(embedding_weights[i].shape))        
            #print(embedding_weights[i][0])
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
            print("Iteration " + str(i))
            # Generate a mask for values greater than 0
            boolean_mask = x > 0
            mask = boolean_mask.astype(int)
            # Calculate the displacement using embedding_weights shape        
            if i > 0:            
                displacement += self.embedding_weights[i-1].shape[0] - 1 # The number of rows - 1
            print("displacement = " + str(displacement))
            x_updated = x + displacement*mask
            i += 1
            self.X_seq_updated.append(x_updated)

        # Second step: use the maximum length of X_sequences (max number of actions per activity), to pad all the other sequences
        maxlen = 0
        for x in self.X_sequences:
            maxlen = max(maxlen, x.shape[1])
    
        print("max length: " + str(maxlen))
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


    def update_y_indices(self):
        """Function to update the y_indices for all datasets, taking into account the common activities found in them
        NOTE: y_indices are one-hot vectors
        """
        num_classes = len(self.common_activity_to_int.keys())
        self.y_indices_updated = []
        for i in range(len(self.y_indices)):
            y = self.y_indices[i]
            int_to_activity = self.int_to_activity_dicts[i]
            y_up = []
            for one_hot in y:            
                index = np.argmax(one_hot, axis=0)            
                new_index = self.common_activity_to_int[int_to_activity[str(index)]]
                y_up.append(new_index)
        
            # TODO: Test the following list comprehension instead of the second for loop
            # y_up = [common_activity_to_int[int_to_activity[index]] for index in y]
            y_up = np.array(y_up) 
            self.y_indices_updated.append(y_up)
    
        # At this point, all y_up in y_indices_updated are numeric indices of activities -> Convert to categorical
        for i in range(len(self.y_indices_updated)):
            self.y_indices_updated[i] = np_utils.to_categorical(self.y_indices_updated[i], num_classes=num_classes)

    def test_view_y_indices(self):
        for i in range(len(self.y_indices_updated)):
            print("Dataset " + str(i))
            y_up = self.y_indices_updated[i]
            y = self.y_indices[i]
            sample = random.randint(0, y.shape[0]-1)
            print("Original activity: " + self.int_to_activity_dicts[i][str(np.argmax(y[sample]))])
            print(y[sample])
            print("Updated activity: " + self.common_int_to_activity[np.argmax(y_up[sample])])
            print(y_up[sample])

"""
Main function used to test the functionality of CrossDatasetFormatter class
"""

def main(argv):
    # List of datasets to reformat and fuse
    DATASETS = ['kasterenC', 'kasterenB']
    # Directories of formatted datasets
    BASE_INPUT_DIR = '../../formatted_datasets/'

    # Select between 'with_time' and 'no_time'
    DAYTIME = 'with_time'
    # Select between 'with_nones' and 'no_nones'
    NONES = 'no_nones'
    # Select between 'avg' and 'sum' for action/activity representation
    OP = 'sum'

    cross_dataset_formatter = CrossDatasetFormatter(DATASETS, BASE_INPUT_DIR, DAYTIME, NONES, OP)
    X_seq_up, y_indices_up, common_embedding_matrix, common_activity_to_int, common_int_to_activity = cross_dataset_formatter.reformat_datasets()
    print("View reformatted datasets' information")
    print("Common activity_to_int:")
    print(common_activity_to_int)
    print("Common int_to_activity:")
    print(common_int_to_activity)
    print("--------------------------------------")
    print("common embedding matrix shape: ", common_embedding_matrix.shape)
    print("--------------------------------------")
    print("Test and view X sequences updated samples:")
    cross_dataset_formatter.test_view_x_seq_updated()
    print("--------------------------------------")
    print("Testing y indices")
    cross_dataset_formatter.test_view_y_indices()

if __name__ == "__main__":
   main(sys.argv)