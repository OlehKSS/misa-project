"""Helper functions for the EM algorithm."""
from math import sqrt, pi
import numpy as np


def gaussian_mixture(features, mean, cov):
    """
    Return Gaussian mixture function for a class.
    
    Inputs:
        features (numpy.ndarray): n by d dimentional array of features of points from feature space, where d is 
                                  is the dimentionality of feature space, n - number of points in the data.
        mean (numpy.ndarray): d-dimentional mean value.
        con (numpy.ndarray): d by d dimentional covariance matrix.
    
    Returns:
        (numpy.ndarray): Gaussian mixture for every point in feature space.
    """
    return np.exp(-0.5*(features - mean) * (1/cov) * np.transpose(features - mean)) / (2 * pi * sqrt(cov))


def membership_weight(p0, p1, p2, a0, a1, a2):
    """
    Return membership weights for three classes.
    
    Inputs:
        p0 (numpy.ndarray): mixture component for the 0th class, which is a probability distribution.
        p1 (numpy.ndarray): mixture component for the 1st class, which is a probability distribution.
        p2 (numpy.ndarray): mixture component for the 2nd class, which is a probability distribution.
        a0 (float): the probability of the 0th class.
        a1 (float): the probability for the 1st class.
        a2 (float): the probability for the 2nd class.
        
    Returns:
        (numpy.ndarray): membership weights for each point from the feature space, 
    """
    denominator = (p0 * a0) + (p1 * a1) + (p2 * a2)
    w0 = (p0 * a0) / denominator
    w1 = (p1 * a1) / denominator
    w2 = (p2 * a2) / denominator
    
    return np.stack((w0, w1, w2), axis=1)


def get_log_likelihood(class_dist, gauss_density):
    """
    Return loglikelihood.
    
    Parameters:
        class_dist: iterable with class distributions.
        gauss_density: iterable with Gaussian densities for each class.
    
    Returns:
        float: log likelihood value. 
    """
    for index, alpha in enumerate(class_dist):
        if index == 0:
            total_sum = alpha * gauss_density[index]
        else:
            total_sum += alpha * gauss_density[index]
    
    return np.sum(np.log(total_sum))
