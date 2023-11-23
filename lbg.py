# import
import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import defaultdict


# main function to generate code book
def generate_codebook(data, size_codebook, epsilon=0.00001, verbose=False):
    """
    Cluster data into centers by Lined-Buzo-Gray algorithm
    
    :param data: ndarray of shape (n_data, dim_data). The input data to be clustered.
    :param size_codebook: int. The number of cluster center(code vector)
    :param epsilon: float. The value to increase or decrease while spliting code vector.
    :param verbose: bool. Whether print results after each iteration or not.
    
    :return
    codebook: list of 1-dim ndarray. Each element is a code vector.
    codebook_abs_weights: list of int. The i-th element is the number of input vector which is in cluster i.
    codebook_rel_weights: list of float. The i-th element is the percentage of input vector which is in cluster i.
    """
    
    # initialization
    global _n_data, _dim_data
    _n_data, _dim_data = data.shape
    codebook = []
    codebook_abs_weights = [_n_data]
    codebook_rel_weights = [1]
    
    # initialize codevector as mean of all input data vectors
    c0 = np.mean(data, axis=0)
    
    # add code vector into the codebook
    codebook.append(c0)
    
    # calculate distortion
    avg_dist_c0 = distortion_c0(c0, data)
    
    # split code vectors until we have enough(size_codebook)
    while len(codebook) < size_codebook:
        
        # step: split code vector
        codebook = split_codebook(codebook, epsilon)
        
        # update
        len_codebook = len(codebook)  # length of current code book
        codebook_abs_weights = np.zeros(len_codebook)
        codebook_rel_weights = np.zeros(len_codebook)
        
        # Step: minimize distortion
        
        # initialization
        avg_dist = 0
        err = epsilon + 1
        num_iter = 0
        
        # for fixed length of code book, iterate untill converge
        while err > epsilon:
            
            # initialization - list recording the nearest code vector for each input data(vector)
            closest_c_list = [None] * _n_data
            
            # initialization - dict mapping code vector index -> input data(vector),
            # the key is index, the value is ndarray of data(vector)
            vecs_near_c = defaultdict(list)
            
            # step: cluster feature vectors, where we find the closest code vector for all input data(vector)
            
            # calculate euclidean distance between each input vector and code vector
            # element (i,j) represent distance between input vector i and code vector j
            dist = distance.cdist(data, np.array(codebook))
            closest_c = np.argmin(dist, axis=1)
            
            # get the minimum index of code vector for each input vector, and record
            # closest_c_idx_list = closest_c.tolist()
            
            # get the series of closest input vector idx for each code vector
            closest_c_seris = pd.Series(closest_c)
            
            # for each code vector index
            for codevec_idx in closest_c_seris.unique():
                # get list of input vector idx closest to a code vector
                list_input_idx = closest_c_seris[closest_c_seris == codevec_idx].index.tolist()
                
                # get list of input vector closest to a code vector
                list_input = data[list_input_idx]
                
                # add the list of input vector to vecs_near_c
                vecs_near_c[codevec_idx] = list_input
            
            # step: find centroid and update code book
            codebook, codebook_abs_weights, codebook_rel_weights = update_codebook(len_codebook, vecs_near_c, codebook,
                                                                                   codebook_abs_weights,
                                                                                   codebook_rel_weights)
            
            # step: compute Distortion and new err value
            avg_dist_prev = avg_dist if avg_dist > 0 else avg_dist_c0
            avg_dist = calculate_distance(closest_c, codebook, data)
            err = (avg_dist_prev - avg_dist) / avg_dist_prev
            
            # print results
            if verbose == True:
                print(f'iteration: {num_iter}, size codebook: {len_codebook}, average distance: {avg_dist.item():.3f}, previous average distance: {avg_dist_prev.item():.3f}, err: {err.item():.3f}')
            
            num_iter += 1
        lbg.ipynb
    return codebook, codebook_abs_weights, codebook_rel_weights


def calculate_distance(closest_c, codebook, data):
    """
    calculate euclidean distance between codebook and input data
    
    :param closest_c: list of int. Each element of index i is the index of code vector closest to the i-th input vector i.
    :param codebook: list of 1-dim ndarray. Each element is a code vector.
    :param data: ndarray of shape (n_data, dim_data). The input data to be clustered.
    
    :return:
    total_distance: float.The total euclidean distance between each input vector and its closest code vector.
    """
    total_distance = 0
    
    for data_idx, codevec_idx in enumerate(closest_c):
        
        # reshape to 2-dim ndarray
        input_vector = np.reshape(data[data_idx], (1, -1))
        code_vector = np.reshape(codebook[codevec_idx], (1, -1))
        
        # distance between one input data vector and its code vector
        dist = distance.cdist(input_vector, code_vector, metric='euclidean')
        
        # add to total distance
        total_distance += dist
    
    # take average
    total_distance /= data.shape[0]
    
    return total_distance


def update_codebook(len_codebook, vecs_near_c, codebook, codebook_abs_weights, codebook_rel_weights):
    """
    After finding the closest code vector for each input vector, re-calculate the code vector as the mean of all input vector in a cluster.
    
    :param len_codebook: int. Number of code vector in code book.
    :param vecs_near_c: dict with int as key and list as value.
        The key is the index of code vector, and the value is a list of input vector index which is in the cluster.
    :param codebook: list of 1-dim ndarray. Each element is a code vector.
    :param codebook_abs_weights: list of int. The i-th element is the number of input vector which is in cluster i.
    :param codebook_rel_weights: list of float. The i-th element is the percentage of input vector which is in cluster i.
    
    :return:
    codebook: list of ndarray. Updated code book.
    codebook_abs_weights: list of int. Updated codebook_abs_weights.
    codebook_rel_weights: list of float. Updated codebook_rel_weights.
    """
    
    # for all code vector index of current codebook
    for codevec_idx, codevec in enumerate(codebook):
        
        # get the proximity input vectors
        vecs = vecs_near_c.get(codevec_idx, [])
        
        # number of proximity input vectors
        num_vecs_near_c = len(vecs)
        
        # update code vector, if it has more than 1 vector of its proximity input vectors
        if num_vecs_near_c > 0:
            # calculate new code vector
            new_codevec = np.mean(vecs, axis=0)
            
            # update code vector
            codebook[codevec_idx] = new_codevec
            
            # update weights
            codebook_abs_weights[codevec_idx] = num_vecs_near_c
            codebook_rel_weights[codevec_idx] = num_vecs_near_c / _n_data
    
    return codebook, codebook_abs_weights, codebook_rel_weights


def split_codebook(codebook, epsilon):
    """
    split code book by adding and subtracting epsilon from each original code vector. The total number of code vector will be doubled.
    
    :param codebook: list of 1-dim ndarray. Each element is a code vector.
    :param epsilon: int. Value to be added or subtracted.
    
    :return:
    new_codebook: list of 1-dim ndarray. The new code book after adding and subtracting epsilon from original code vectors.
    """
    # initialize new codebook
    new_codebook = []
    
    # add and subtract from all original code vector
    for code_vector in codebook:
        new_codebook.append(code_vector + epsilon)
        new_codebook.append(code_vector - epsilon)
    
    return new_codebook


def distortion_c0(c0, data):
    """
    calculate the average distortion(euclidean distance) between initialized code vector(c0) and all input vectors
    
    :param c0: 1-dim ndarray. The initialized code vector.
    :param data: ndarray of shape (n_data, dim_data). The input data to be clustered.
    
    :return: float. The average distortion.
    """
    n_data = data.shape[0]
    c0 = np.reshape(c0, (1, -1))
    distortion = (1 / n_data) * np.sum(np.power(np.linalg.norm(data - c0), 2))
    
    return distortion
