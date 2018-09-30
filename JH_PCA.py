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
        vector_unscaled_x = np.dot(covariance_matrix, vector_x)
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
data = file_reader('magic04.data')

# step a: Z-Normalization
norm_data = z_norm(data)

# step b: Covariance Matrix
covariance_matrix = cov_matrix(norm_data)
npSigma = np.cov(norm_data,  bias=True, rowvar=False)

print('\nb. The covariance matrix is:')
print(covariance_matrix)
print('\n   comparison with np.cov:')
print(npSigma)


# step c: Dominant eigenvalue and eigenvector
epsilon = 0.000001
dominant_value, dominant_vector = eigen(norm_data, epsilon)
print('\nc. My maximum eigenvalue is:', str(dominant_value))
print('   My dominant eigenvector is:')
print(dominant_vector)
print('\nCompare with np linalg.eig method:')
print(np.linalg.eig(covariance_matrix))


# step d: Projection on the first two eigen vectors
eigen_value_unsorted, eigen_vector_unsorted = np.linalg.eig(covariance_matrix)
idx = eigen_value_unsorted.argsort()[::-1]
eigen_value_sorted = eigen_value_unsorted[idx]
eigen_vector_sorted = eigen_vector_unsorted[:, idx]
eigen_vec = eigen_vector_sorted[:, 0:2]
row_data, column_data = norm_data.shape
projection = np.zeros((row_data, column_data))
matrix_a = np.zeros((row_data, 2))
for i in range(row_data):  # compute the projected points
    matrix_a[i] = (np.dot(eigen_vec.T, norm_data[i]))
for i in range(row_data):
    projection[i] = np.dot(eigen_vec, matrix_a[i].T)
variance = np.var(projection)
print('\nd. The variance is:', str(variance))

# step e: Covariance matrix in eigen-decomposition form
eigen_val = np.linalg.eig(covariance_matrix)[0]
similar_matrix = np.diag(eigen_val)
eigen_matrix = eigen_vec = np.linalg.eig(covariance_matrix)[1]
cov_matrix_dec = np.dot(np.dot(eigen_vec, similar_matrix), eigen_vec.T)
print('\ne. The Eigen-decomposition form of covariance matrix:')
print('  U:')
print(eigen_matrix)
print('similar matrix:')
print(similar_matrix)
print('  UT:')
print(eigen_matrix.T)
print('  U LAMBDA UT:')
print(cov_matrix_dec)

# step f: PCA sub-routine
def PCA(D, alpha):
    """
    PCA algorithm sub-routine, page 198
    :param D: the input dataset
    :param alpha: how much variance we want to preserve
    :return: an array of principal vectors
    """
    norm_D = z_norm(D)  # step 1,2
    cov_D = cov_matrix(norm_D)  # step 3
    val_D_unsorted, vector_D_unsorted = np.linalg.eig(cov_D)
    index = val_D_unsorted.argsort()[::-1]
    eigen_val_D = val_D_unsorted[index]  # step 4
    U = vector_D_unsorted[index]  # step 5
    row, column = D.shape
    variance_fraction = 0
    vector_index = -1
    total_eigen_value = np.sum(eigen_val_D)
    sum_eigen_value = 0
    while variance_fraction < alpha:
        vector_index += 1
        sum_eigen_value += eigen_val_D[vector_index]
        variance_fraction = sum_eigen_value/total_eigen_value
    Ur = U[:vector_index+1]
    Ar = np.zeros((row, vector_index+1))
    for i in range(row):
        Ar[i] = np.dot(Ur, norm_D[i].T)
    return Ur, Ar


# use PCA sub-routine to get
principal_vectors, reduced_dime_data = PCA(data, 0.95)
first_10 = reduced_dime_data[:10]
print('\nf. The principal vectors we need are:')
print(principal_vectors)
print('\n   The first 10 reduced dimensionality data co-ordinates:')
print(first_10)


# step g: compute the co-variance of projected data
cov = np.cov(reduced_dime_data,  bias=True, rowvar=False)
projected_points = np.zeros((row_data, column_data))
for i in range(row_data):
    projected_points[i] = np.dot(principal_vectors.T, reduced_dime_data[i].T)
sum_eigen_val = np.sum(eigen_val[:len(principal_vectors)])
