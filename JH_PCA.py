#!/usr/bin/env python
# Jingyi hui 09/22/2018
# CSCI573 DATAMINING
# HOMEWORK 2 PCA

import pandas as pd
import numpy as np

# read data from file into dataframe
# file_name = raw_input('Please input the name of data file:')
raw_data = pd.read_csv('iris.txt', header=None)
raw_data_array = np.array(raw_data)
row, column_label = raw_data_array.shape  # shape(19020, 10)
column = column_label - 1
data = raw_data_array[:, 0: column]  # slice the array to eliminate label column
data = np.float64(data)


# 1st step: Z-Normalization
mean_vec = np.mean(data, axis=0)
# print(mean_vec)
std_vec = np.std(data, axis=0)
norm_data = (data - mean_vec)/std_vec
# print('1. The normalized data is:')
# print(norm_data)


# 2nd step: Covariance Matrix
cov_matrix = np.zeros((column, column))
for i in range(row):
    a = norm_data[i]  # each centered point
    b = np.transpose(a)  # transpose of centered point
    cov_matrix = cov_matrix + np.outer(a, b)  # sum of outer product of each centered point
cov_matrix = cov_matrix/row  # normalize by n
# comparison = np.cov(norm_data.T)
print('\n2. The covariance matrix is:')
print(cov_matrix)

# 3rd step: Dominate eigenvalue and eigenvector
change = float('INF')
vector_pre = np.zeros(column)
vector_x = np.random.rand(column)  # initialize a random vector
while change >= 0.000001:  # keep calculating the new vector until the change < threshold
    vector_pre = vector_x
    vector_unscaled_x = np.dot(cov_matrix, vector_x)
    max_val = np.amax(np.absolute(vector_unscaled_x))  # get the max abs value
    vector_x = vector_unscaled_x/max_val
    change = np.linalg.norm(vector_x - vector_pre)
norm = np.linalg.norm(vector_x)  # normalize the vector
vector_norm_x = vector_x/norm
max_x = np.amax(np.absolute(vector_unscaled_x))
max_pre = np.amax(np.absolute(vector_pre))
eigenvalue = max_x/max_pre  # calculate the max eigenvalue
print('\n3. The maximum eigenvalue is:', str(eigenvalue))
# print(np.linalg.eig(cov_matrix))

# 4th step: Projection on the first two eigen vectors
eigen_vec = np.linalg.eig(cov_matrix)[1]
eigen_vec_1 = eigen_vec[:, 0]
eigen_vec_2 = eigen_vec[:, 1]
projection = np.zeros((row, column))
for i in range(row):  # compute the projected points
    projection_1 = (np.dot(eigen_vec_1.T, norm_data[i]))*eigen_vec_1
    projection_2 = (np.dot(eigen_vec_2.T, norm_data[i]))*eigen_vec_2
    projection[i] = projection_1 + projection_2
variance = np.var(projection)
print('\n4. The variance is:', str(variance))

# 5th step: Covariance matrix in eigen-decomposition form
eigen_val = np.linalg.eig(cov_matrix)[0]
similar_matrix = np.diag(eigen_val)
cov_matrix_dec = np.dot(np.dot(eigen_vec, similar_matrix), eigen_vec.T)
print('\n5. The Eigen-decomposition form of covariance matrix:')
print(cov_matrix_dec)

# 6th step:
