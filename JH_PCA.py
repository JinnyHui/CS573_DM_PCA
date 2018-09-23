#!/usr/bin/env python
# Jingyi hui 09/22/2018
# CSCI573 DATAMINING
# HOMEWORK 2 PCA

import sys
import pandas as pd
import numpy as np

# read data from file into dataframe
# file_name = sys.argv[1]
raw_data = pd.read_csv('iris.txt', header=None)
raw_data_array = np.array(raw_data)
row, column_label = raw_data_array.shape  # shape(19020, 10)
column = column_label - 1
data = raw_data_array[:, 0:column_label]  # slice the array to eliminate label column
data = np.float64(data)


# 1st step: Z-Normalization
mean_vec = np.mean(data, axis=0)
# print(mean_vec)
std_vec = np.std(data, axis=0)
norm_data = (data - mean_vec)/std_vec
#print(centered_data)


# 2nd step: Covariance Matrix
cov_matrix = np.zeros((column, column))
for i in range(row):
    a = norm_data[i]  # each centered point
    b = np.transpose(a)  # transpose of centered point
    cov_matrix = cov_matrix + np.outer(a, b)  # sum of outer product of each centered point
cov_matrix = cov_matrix/row  # normalize by n
# comparison = np.cov(norm_data.T)
# print(cov_matrix - comparison)

# 3rd step: Dominate eigenvalue and eigenvector
