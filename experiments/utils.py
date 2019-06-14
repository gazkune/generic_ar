# -*- coding: utf-8 -*-
"""
Created on Thur Mar 14 15:33:52 2019
@author: gazkune
"""
from __future__ import print_function

import sys
import os

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn

import numpy as np
import json

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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
        plt.figure(figsize=None)
        # summarize history for accuracy
        if 'accuracy' in metrics:
            leg = ['train']
            plt.plot(history['acc'])
            if 'val_acc' in history:
                plt.plot(history['val_acc'])
                leg.append('val')

            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            lgd = plt.legend(leg, bbox_to_anchor=(1.04,1), loc="upper left")
            if save == True:
                plt.savefig(filename + '-accuracy.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.gcf().clear()
            else:
                plt.show()

        # summarize history for loss
        if 'loss' in metrics:
            leg = ['train']
            plt.plot(history['loss'])
            if 'val_loss' in history:
                plt.plot(history['val_loss'])
                leg.append('val')
                
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            #plt.ylim(1e-3, 1e-2)
            #plt.yscale("log")
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

    def calculate_accuracy_at_k(self, y_gt, y_preds, k):
        """Function to calculate top k accuracy or accuracy at k
        Parameters
        ----------
            y_gt : array, shape = [n_samples]
                Classes that appear in the ground truth.
        
            y_preds: array, shape = [m_samples, n_samples]
                Predicted classes (m per test sample, where m >= k). Take into account that they must follow the same
                order as in y_ground_truth
           
        Returns
        -------
            accuracy : float
                A float between 0-1 with the top k accuracy
        """
        kpreds = y_preds[:, :k] # Use only the first k columns of y_preds
        hits = 0
        for i in range(len(y_gt)):
            if y_gt[i] in kpreds[i]:
                hits += 1
        
        return float(hits)/float(len(y_gt))



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

    def plot_confusion_matrix(self, y_true, y_pred, classes, path,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """        
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data        
        classes = np.array(classes)        
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        #return ax
        #plt.show()
        plt.savefig(path)
        plt.gcf().clear()
    

    def plot_heatmap(self, y_true, y_pred, classes, path,
                     normalize=False,
                     title=None,
                     cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        #np.set_printoptions(precision=2)        
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Utils: ypred unique values: " + str(np.unique(y_pred)))
        # Only use the labels that appear in the data        
        classes = np.array(classes)        
        classes = classes[unique_labels(y_true, y_pred)]
        fmt = ""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            print("Normalized confusion matrix")
        else:
            fmt = "d"
            print('Confusion matrix, without normalization')
       
        fontsize = 10
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        fig = plt.figure(figsize=(25, 25))
        
        heatmap = sn.heatmap(df_cm, annot=True, fmt=fmt, cmap=cmap)        
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.axes().set_title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.savefig(path)
        plt.gcf().clear()