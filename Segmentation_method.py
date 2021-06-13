# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:34:05 2019

@author: NUS
"""
import numpy as np 
#####################################################K-means clustering###########################################
# randomly select the centroids
def randCent(data,k):
    """random gengerate the centroids
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
    """
    centroids = np.zeros([k, data.shape[1]], dtype=object)
    # grab the kth n_features
    for ind1 in range (k):
        centroids[ind1] = data[np.random.randint(0, high=data.shape[0]-1)]
    
    return centroids

def euclidist(point1,point2):
    # get numer of features
    m = point1.shape[0]
    features_sum = 0
    # features_num = 0
    for feature in range(m):
        features_sum += np.square(point1[feature] - point2[feature])
    distance = features_sum ** 0.5
    
    return distance
        

def create_dist_list(data, centroids, k):
    ### output array of distances of the nth point in data from kth centroid
    n = data.shape[0]
    m = data.shape[1]
    dist_list = np.zeros((n,k), dtype=np.float64)
    for point in range(n):
        for centroid in range(k):
            dist_list[point, centroid] = euclidist(data[point], centroids[centroid])
    return dist_list

def create_clusterAssment(data, centroids, k):
    n = data.shape[0]
    m = data.shape[1]
    
    dist_list = create_dist_list(data,centroids, k)
    clusterAssment = np.zeros((n, 1), dtype=np.int16)
    for point in range (n):
        centroids = np.where(dist_list[point] == np.amin(dist_list[point]))
        clusterAssment[point] = centroids[0][0]
    
    return clusterAssment

def create_newCentroids(data, clusterAssment, k):
    ### output array of [k,m] of centroids
    # Update centroid to the mean of the cluster
    n = data.shape[0]
    m = data.shape[1]
    # k is given
    sum_feat = np.zeros((k,m,2))
    centroids = np.zeros((k,m))
    
    for point in range(n):
        cluster = int(clusterAssment[point]) # in range k
        for feature in range (m):
            sum_feat[cluster][feature][0] += data[point][feature]
            sum_feat[cluster][feature][1] +=1
            
    for centroid in range(k):  
        for feature in range (m):
            centroids[centroid][feature] = sum_feat[centroid][feature][0]/sum_feat[centroid][feature][1]
            
    return centroids
                    
                    
                        
def KMeans(data,k):
    """ KMeans algorithm 
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
        clusterAssignment:  <class 'numpy.matrix'>, shape=[n_samples, 1]
    """
    #Initialize arrays and randomize centroids
    n_samples = data.shape[0]
    oldClusterAssment = np.zeros((n_samples, 1))
    centroids = randCent(data, k)
    
    #The distance to each centroid is calculated and the point is assigned to the nearest centroid, from 0 to k-1
    clusterAssment = create_clusterAssment(data, centroids, k)
    
    #Loop till convergence of cluster assignments
    while (not (np.array_equal(oldClusterAssment, clusterAssment))):                             
        
        #Update centroid to the mean of the cluster
        centroids = create_newCentroids(data, clusterAssment, k)
                                 
        #Reassign clusters
        oldClusterAssment = clusterAssment
        clusterAssment = create_clusterAssment(data,centroids, k)
    
    return centroids, clusterAssment

    """# number of cluster = k
    # data points = data
    # data points number = n = data.shape[0]
    # data feature number = m = data.shape[1]
    # centroids = centroids = randCent(data, k), will be proccessed
    n = data.shape[0]
    m = data.shape[1]
    centroids = randCent(data,k)
    # euclidean distance calculator = np.linalg.norm(data1, data2, m)
    # data 1 is the starting point, data 2 is the end point
    # make dist_array = array[k, sample_no_eucli_dist]
    # each keeping the euclidean distance between the centroid and other data points
    dist_array = np.zeros([k, n], dtype=object)
    for ind1 in range (k):
        for ind2 in range (n):
            dist_array[ind1, ind2] = np.linalg.norm(centroids[ind1] - data[ind2])
    # find which centroid has smallest eucli_dist to data point, assign
    clusterAssment = np.zeros([n,1], dtype=object)
    # clusterAssment_holder: current_smallest_euclidistance
    clusterAssment_holder = np.full([n], np.inf, dtype=object)
    for ind3 in range (k):
        for ind4 in range (n):
            if dist_array[ind3, ind4] < clusterAssment_holder[ind4]:
                clusterAssment_holder[ind4] = dist_array[ind3, ind4]
                clusterAssment[ind4,0] = ind3
    # first rounds success
    # now to find new center
    centroids_old = np.zeros([k], dtype=object)
    while centroids_old.all() != centroids.all():
        # while resultant center not previous center
        centroids_old = centroids
        # find new center
        # get mean of all k-th centroid cluster data
        for ind5 in range (k):
            # sum_data = sum of the features of the data of the current loop
            sum_data = np.zeros([m], dtype=object)
            sum_num = 0
            # run through the n data points and add ones belonging to the cluster together
            for ind6 in range (n):
                if clusterAssment[ind6] == ind5:
                    sum_data += data[ind6]
                    sum_num += 1
        # update the ind5-th centroid
        if sum_num > 1:
            centroids[ind5] = sum_data/sum_num
        # now we have new centroids, reassign data cluster
        for ind7 in range (k):
            for ind8 in range (n):
                if dist_array[ind7, ind8] < clusterAssment_holder[ind8]:
                    clusterAssment_holder[ind4] = dist_array[ind7, ind8]
                    clusterAssment[ind8,0] = ind7
                    # now it is ready for reassign of centroid
    # until centroids converge
    # that is the new center
    
    return centroids, clusterAssment"""

##############################################color #############################################################
import random
def colors(k):
    """ generate the color for the plt.scatter
    parameters
    ------------
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        ret: <class 'list'>, len = k
    """    
    ret = []
    for i in range(k):
        ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret


############################################mean shift clustering##############################################
from collections import defaultdict
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
 
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    """mean shift cluster for single seed.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Samples to cluster.
    nbrs: NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
    max_iter: max interations 
    return:
        mean(center) and the total number of pixels which is in the sphere
    """
    n,m = X.shape[0], X.shape[1]
    # For each seed, climb gradient until convergence or max_iter
    # get first data point from my_mean(seed) and X
    # define as mean_old
    # initialize mean_old
    #print(my_mean)
    #mean_old = np.zeros([1,m], dtype=object)
    #mean = np.zeros([1,m], dtype=object)
    num_data = 0
    mean = np.zeros([1,m], dtype=object)
    mean_old = np.zeros([1,m], dtype=object)
    for feature in range(m):
        mean[0][feature] = my_mean[feature]
    #print("this is mean before")
    #print(mean)
    iter_num=0
    placeholder_mean = np.zeros([1,m], dtype=object)
    while (not (np.array_equal(mean, mean_old))) and iter_num < max_iter:
        mean_old = mean
        mean_total = np.zeros([1,m], dtype=object)
        distances, indices = nbrs.radius_neighbors(mean)
        #print ("indices")
        #print (indices)
        neighbors = indices[0]
        #print ("neighbors")
        #print (neighbors)
        num_data = len(neighbors)
        mean_total = np.zeros([1,m], dtype=object)
        for neighbor in neighbors:
            #print(X[index])
            for feature in range (m):
                mean_total[0][feature]+=X[neighbor][feature]
        if num_data > 0:
            mean = np.divide(mean_total, num_data)
                
    # calculate new mean
    #print("this is mean after")
    #print(mean)   
    #print(mean, num_data)
    return mean, num_data


def mean_shift(X, bandwidth=5, seeds=None, bin_seeding=False,min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=None):
    """pipline of mean shift clustering
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bandwidth: the radius of the sphere
    seeds: whether use the bin seed algorithm to generate the initial seeds
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        cluster_centers <class 'numpy.ndarray'> shape=[n_cluster, n_features] ,labels <class 'list'>, len = n_samples
    """
    # find the points within the sphere
    #print (X) 
    #print("this is X[0]")
    #print(X[0])
    #print("this is X[0][0]")
    #print(X[0][0])
    n,m = X.shape[0], X.shape[1]
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
    # get seeds
    seeds = get_bin_seeds(X, bandwidth) 
    # get new means from seed
    ##########################################parallel computing############################
    center_intensity_dict = {}
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(_mean_shift_single_seed)
        (seed, X, nbrs, max_iter) for seed in seeds)#
    ##########################################parallel computing############################
    #print("all_res")
    #print(all_res)
    # rank the seeds by intensity
    for seed in range(len(all_res)):
        center_intensity_dict[tuple(all_res[seed][0][0])] = int(all_res[seed][1])
    #print("center_intensity_dict")
    #print(center_intensity_dict)
    
    # clean zero entries
    center_intensity_dict_copy = center_intensity_dict.copy()
    for cluster in center_intensity_dict_copy:
        if center_intensity_dict[cluster] == 0:
            del center_intensity_dict[cluster]
    #print("center_intensity_dict")
    #print(center_intensity_dict)
    
    # sort by descending intensity
    center_intensity_dict_sorted = sorted(center_intensity_dict, reverse = True)
    centroids = center_intensity_dict_sorted
    #print("center_intensity_dict_sorted")
    #print(center_intensity_dict_sorted)
    
    # remove clusters too close to bigger cluster
    bandwidth2 = bandwidth/2
    nbrs2 = NearestNeighbors(radius=bandwidth2, n_jobs=1).fit(center_intensity_dict_sorted)
    keep_centroid = []
    remove_centroid = []
    num_centroids = len(centroids)
    for centroid in range (num_centroids):
        if centroid not in remove_centroid:
            keep_centroid.append(centroid)
            #print("centroids[centroid]")
            #print(centroids[centroid])
            dist, indices = nbrs2.radius_neighbors(np.array(centroids[centroid]).reshape(1,m))
            removableIndices = (indices[0])
            for index in removableIndices:
                if (index not in keep_centroid 
                    and index not in remove_centroid):
                    remove_centroid.append(index)
                
    #assign all data points
    cluster_centers = np.array([centroids[index] for index in keep_centroid])
    clusterAssment = create_clusterAssment(X, cluster_centers, len(keep_centroid))
    #print("clusterAssment")
    #print(clusterAssment)
    labels = clusterAssment.flatten()
    
    return cluster_centers, labels

def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """generate the initial seeds, in order to use the parallel computing 
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        bin_seeds: dict-like bin_seeds = {key=seed, key_value=he total number of pixels which is in the sphere }
    """

    n = X.shape[0]
    m = X.shape[1]
    
    # process all data by bin_size
    # flatten with round
    seeds = bin_size * np.round(np.divide(X, bin_size))
    bin_seeds = {}
    seedKeys = tuple(map(tuple, seeds))
    
    # bin seeds by freq
    for seed in seedKeys:
        if seed in bin_seeds:
            bin_seeds[seed] += 1
        else:
            bin_seeds[seed] = 1

    bin_seeds_old = bin_seeds.copy()
    # get list of seeds to delete
    for key in bin_seeds_old:
        if bin_seeds_old[key] < min_bin_freq:
            del bin_seeds[key]
    # print (bin_seeds)
    return bin_seeds