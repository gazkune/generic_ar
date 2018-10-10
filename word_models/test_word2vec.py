#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:14:31 2018

@author: gazkune
"""
import sys

from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import numpy as np
import matplotlib.pyplot as plt


from gensim.models import KeyedVectors

import pandas as pd

DATASETS_DIR = "../datasets/"
DATASET_NAME = "kasterenC"
sensor_labels = DATASETS_DIR + DATASET_NAME + "/sensor_labels.txt"
activity_labels = DATASETS_DIR + DATASET_NAME + "/activity_labels.txt"

model_path = "GoogleNews-vectors-negative300.bin.gz"


def tsne_plot(embeddings, words, save, filename):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for i in xrange(len(words)):
        tokens.append(embeddings[i])
        labels.append(words[i])    
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    plt.title('Embedding visualization with TSNE')
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    
    if save == False:        
        plt.show()
    else:
        plt.savefig(filename)
        plt.gcf().clear()



# Load sensor names
sensor_df = pd.read_csv(sensor_labels, header=None, sep=' ')
sensors = sensor_df[1].values

# Load activity names
act_df = pd.read_csv(activity_labels, header=None, sep=' ')
activities = act_df[1].values

print "Loading", model_path, "model"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print "Model loaded"

# In this version, we only visualize the atomic words of the composed sensor names
# Some sensors have composed names such as 'hall_bathroom_door'; extract the words
"""
words = []
for sensor in sensors:
    words.extend(sensor.split('_'))
    
# Remove repeated words
words = set(words)
# Build a lit again
words = list(words)

tsne_plot(model[words], words)
"""

# In this version, we build one embedding per sensor name, suming the 
# embeddings for the words
embeddings = []
for sensor in sensors:
    words = sensor.split('_')
    embedding = np.zeros(300)
    for word in words:
        embedding = embedding + model[word]
    
    embeddings.append(embedding)        

#tsne_plot(embeddings, sensors, True, DATASET_NAME+"-sensor-tsne.png")

# We can also plot sensors and activities in the same plot
for activity in activities:
    words = activity.split('_')
    embedding = np.zeros(300)
    for word in words:
        if word != 'to': # word 'to' is not in the model (??)
            embedding = embedding + model[word]

    embeddings.append(embedding)

# Add "morning", "evening", "midday", "afternoon", "night" to test
moments = np.array(["morning", "evening", "midday", "afternoon", "night"])
for moment in moments:
    words = moment.split('_')
    embedding = np.zeros(300)
    for word in words:
        if word != 'to': # word 'to' is not in the model (??)
            embedding = embedding + model[word]

    embeddings.append(embedding)
    
    
words = np.concatenate((sensors, activities, moments), axis=0)
tsne_plot(embeddings, words, True, DATASET_NAME+"-sensor-activity-time-tsne.png")

#print (model.most_similar(model['hall'] + model['toilet'] + model['door']))

















