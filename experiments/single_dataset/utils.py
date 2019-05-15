# -*- coding: utf-8 -*-
"""
Created on Thur Mar 14 15:33:52 2019
@author: gazkune
"""
from __future__ import print_function

import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import json

from sklearn import metrics
from sklearn.metrics import confusion_matrix

class Utils:

    def __init__(self):
        """ Constructor
        
        Usage example:
            utils = Utils()
            
        Parameters
        ----------
        None
            
        Returns
        ----------
        Instance of the class        
        """
        # TODO: Let the user select the metrics and variants?
        self.metric_names = ['precision', 'recall', 'f1']
        self.metric_variants = ['micro', 'weighted', 'macro']
        

    def plot_training_info(self, metrics, save, history, filename):
        """
        Function to draw and save plots
        """
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
        
    def calculate_evaluation_metrics(self, y_gt, y_preds):
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
        # TODO: Update to use self.metric_names                
        metric_results = {
            'precision' : {},
            'recall' : {},
            'f1' : {},
            'acc' : -1.0        
        }
        
        for t in self.metric_variants:
            metric_results['precision'][t] = metrics.precision_score(y_gt, y_preds, average = t)
            metric_results['recall'][t] = metrics.recall_score(y_gt, y_preds, average = t)
            metric_results['f1'][t] = metrics.f1_score(y_gt, y_preds, average = t)
            metric_results['acc'] = metrics.accuracy_score(y_gt, y_preds) 
                
        return metric_results

    def init_metrics_per_fold(self):
        """Function to initialize a dictionary to store all the metrics per fold in a cross-validation process
        """
        metrics_per_fold = {} # This dict is to store the metrics of each fold
        # Initialize the dictionary with empty lists        
        metrics_per_fold['acc'] = []

        for metric in self.metric_names:
            metrics_per_fold[metric] = {}
            for variant in self.metric_variants:
                metrics_per_fold[metric][variant] = []

        return metrics_per_fold

    def update_metrics_per_fold(self, metrics_per_fold, metrics):
        """Function to update the metrics_per_fold dictionary with the metrics of a given fold
        """
        metrics_per_fold['acc'].append(metrics['acc'])
        for metric in self.metric_names:
            for variant in self.metric_variants:
                metrics_per_fold[metric][variant].append(metrics[metric][variant])
        
        return metrics_per_fold

    def calculate_aggregate_metrics_per_fold(self, metrics_per_fold):
        """Function to calculate mean and std for each metric
        """
        metrics_per_fold['mean_acc'] = np.mean(np.array(metrics_per_fold['acc']))
        metrics_per_fold['std_acc'] = np.std(np.array(metrics_per_fold['acc']))    
        for metric in self.metric_names:
            metrics_per_fold['mean_' + metric] = {}
            metrics_per_fold['std_' + metric] = {}
            for variant in self.metric_variants:
                metrics_per_fold['mean_' + metric][variant] = np.mean(np.array(metrics_per_fold[metric][variant]))
                metrics_per_fold['std_' + metric][variant] = np.std(np.array(metrics_per_fold[metric][variant]))
        
        return metrics_per_fold


    def find_file_maxnumber(self, directorypath):
        """Returns the maximum index number of the files of a given directory
        It assumes file names are separated by '-' and the first part contains the index
        Usage example:
            maxnumber = find_file_maxnumber("/results/kasterenA/")            
        
        Parameters
        ----------
            directorypath: str
                The path of the directory where target files are stored.            
           
        Returns
        -------
            maxnumber: int
                The maximum index number of the file names of the directory
        """
        contents = os.listdir(directorypath)
        maxnumber = -1
        for f in contents:
            parts = f.split('-')
            if int(parts[0]) > maxnumber:
                maxnumber = int(parts[0])
        
        return maxnumber

