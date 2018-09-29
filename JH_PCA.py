#!/usr/bin/env python
# Jingyi hui 09/22/2018
# CSCI573 DATAMINING
# HOMEWORK 2 PCA

import pandas as pd
import numpy as np


def file_reader(name):
    """
    read data from file into dataframe
    :param name: file name input by user
    :return: clean dataset
    """
    raw_data = pd.read_csv(name, header=None)
    raw_data_array = np.array(raw_data)
    row, column_label = raw_data_array.shape  # shape(19020, 10)
    column = column_label - 1
    data_clean = raw_data_array[:, 0: column]  # slice the array to eliminate label column
    return np.float64(data_clean)


def z_norm(dataset):
    """
    perform z-normalization
    :param dataset: dataset
    :return: normed data
    """
    mean_vec = np.mean(dataset, axis=0)
    std_vec = np.std(dataset, axis=0)
    data_norm = (data - mean_vec) / std_vec
    return data_norm


def cov_matrix(dataset):
    """
    compute the covariance matrix
    :param dataset: cleaned dataset
    :return: covariance matrix
    """
    row, column = dataset.shape
    matrix = np.zeros((column, column))
    for x in range(row):
        a = dataset[x]  # each centered point
        b = np.transpose(a)  # transpose of centered point
        matrix = matrix + np.outer(a, b)  # sum of outer product of each centered point
    matrix = matrix / row  # normalize by n
    return matrix


def eigen(dataset, e):
    """
    Compute dominant eigenvalue and eigenvector with power iteration
    :param dataset: clean dataset
    :param e: the threshold to converge
    :return: tuple of eigenvalue and eigenvector
    """
    row, column = dataset.shape
    change = float('INF')
    vector_pre = np.zeros(column)
    vector_x = np.random.rand(column)  # initialize a random vector
    while change >= e:  # keep calculating the new vector until the change < threshold
        vector_pre = vector_x
        vector_unscaled_x = np.dot(cov_matrix, vector_x)
        max_val = np.amax(np.absolute(vector_unscaled_x))  # get the max abs value
        vector_x = vector_unscaled_x / max_val
        change = np.linalg.norm(vector_x - vector_pre)
    norm = np.linalg.norm(vector_x)
    eigenvector = vector_x / norm  # normalize the vector
    max_x = np.amax(np.absolute(vector_unscaled_x))
    max_pre = np.amax(np.absolute(vector_pre))
    eigenvalue = max_x / max_pre  # calculate the max eigenvalue
    return eigenvalue, eigenvector

###############################################################################################
#                                 START OF THE PROGRAM                                        #
###############################################################################################


# file_name = raw_input('Please input the name of data file:')
data = file_reader('iris.txt')

# step a: Z-Normalization
norm_data = z_norm(data)

# step b: Covariance Matrix
cov_matrix = cov_matrix(norm_data)
npSigma = np.cov(norm_data,  bias=True, rowvar=False)

print('\nb. The covariance matrix is:')
print(cov_matrix)
print('\n   comparison with np.cov:')
print(npSigma)


# step c: Dominant eigenvalue and eigenvector
epsilon = 0.000001
dominant_value, dominant_vector = eigen(norm_data, epsilon)
print('\nc. My maximum eigenvalue is:', str(dominant_value))
print('   My dominant eigenvector is:')
print(dominant_vector)
print('\nCompare with np linalg.eig method:')
print(np.linalg.eig(cov_matrix))


# step d: Projection on the first two eigen vectors
eigen_vec = np.linalg.eig(cov_matrix)[1][:, 0:2]
row_data, column_data = norm_data.shape
projection = np.zeros((row_data, column_data))
matrix_a = np.zeros((row_data, 2))
for i in range(row_data):  # compute the projected points
    matrix_a[i] = (np.dot(eigen_vec.T, norm_data[i]))
for i in range(row_data):
    projection[i] = np.dot(eigen_vec, matrix_a[i].T)
variance = np.var(projection)
print('\n4. The variance is:', str(variance))

# step e: Covariance matrix in eigen-decomposition form
eigen_val = np.linalg.eig(cov_matrix)[0]
similar_matrix = np.diag(eigen_val)
eigen_matrix = eigen_vec = np.linalg.eig(cov_matrix)[1]
cov_matrix_dec = np.dot(np.dot(eigen_vec, similar_matrix), eigen_vec.T)
print('\n5. The Eigen-decomposition form of covariance matrix:')
# print(cov_matrix_dec)

# step f: PCA sub-routine
def PCA(D, alpha):
    return 
