### Benjamin Frost Individual Project 2022

#### MSc Artificial Intelligence


This dissertation set out to determine methods of categorising numerical time series data so that it can be used as an input to an LEN model.
The purpose of this is that time series data can then have first order logic explanations generated and used in various high stakes domains such as medicine.

-------------------------------

This repository contains the code to accompany the written section of dissertation.

The Notebooks/ directory contains the notebooks and helper classes created throughout the project. 

The processingCache/ contains data at different stages of processing, as well as outputs of training models on different dates.

The only cached data file that has been ommitted is the Sepsis dataset which is too large to be included in the repository (440MB).
This file can be computed through execution of the Sepsis.ipynb notebook.

If you intend to train a model using the categorised data in this notebook, the final data files can be found in the categorisedData/ directory.
The testing.ipynb file contains k-fold validation code for training and extracting explanations from LEN models.

Thank you,
Ben