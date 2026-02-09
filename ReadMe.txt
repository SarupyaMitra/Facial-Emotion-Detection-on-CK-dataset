# CK+ Emotion Recognition

This projects implements facial emotion recognition on the CK+ dataset using Gabor filter-based features and Support Vector Machines. 

## Dataset:
ckextended Dataset is found from https://www.kaggle.com/datasets/davilsena/ckdataset. 

Fer2013 Dataset is found from https://www.kaggle.com/datasets/msambare/fer2013.

All information about the dataset is present in this link. 

## Feature Extraction
 - Gabor filter bank
 - Block-wise statistical features

## Classification
 - Linear SVM

## Results
See RESULTS.md for detailed evaluation.

## How to Run
1. Download the CK+ dataset
2. Place it inside the src folder
3. Run src/Gabor_Outputs_generation.py (This will generate a file named outputs_ck.npy inside data folder)
4. Run src/Feature_Extraction.py (This will generate a file named imgwise_blockmeans_ck.npy inside data folder)
5. Run src/Classification.py

## General Description

There are 3 main files:
1) Gabor_Outputs_generation.py =: Here I created the Gabor Filter bank and then apply them on each image present in the dataset. We get (no_of_input_images * no_of_filters_in_the_bank) no. of outputs.

2) Feature_Extraction.py =: In this file, I took all the outputs and break them into non-overlapping blocks. For every block, I calculated some statistic and I will use these statistics as features. So for one input image, I will get (no_of_outputs_per_input_image * no_of_blocks_per_output) features.

NOTE: no_of_outputs_per_input_image is equal to no_of_filters_in_the_bank

3) Classification.py =: In this file, I split the data into training(80%) and testing(20%). In the dataset itself there was a column that showed which image is used in Training or in Testing. I followed that column. I used SVM classifier for the classification after normalizing all the features.




