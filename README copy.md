# PigCallClassifier
-----------------------
Ciara Cleo-Rose Sypherd
-----------------------

This repository contains the scripts used to train the classifiers for the Soundwel paper: "Classification of pig calls produced from birth to slaughter according to their emotional valence and context of production."

OverallNNs.m is the main script for this task. The other scripts included here are either function files or preprocessing programs.

OverallNNs takes in a dataset of uniformly formatted spectrograms and a spreadsheet of class labels (in our case, Valence and Context) and trains a ResNet-50 CNN on this dataset.

In order to provide meaningful performance statistics, this process is iterated for a set number of times. The highest validation set accuracies are compiled to calculate the F1, Recall, and Precision.
