"""Script containing some additonal functions used thorughout the other main scripts."""
from math import isfinite

import numpy as np
from scipy.stats import norm


def gaussian_kernel(x_var, x_i, bandwidth):
    """
    Gaussian Kernel Function for calculating weights.

    Parameters:
        x_var (float): The variable for which the kernel weight is calculated.
        x_i (float): The data point around which the kernel is centered.
        bandwidth (float): The bandwidth (standard deviation) of the Gaussian.

    Returns:
        float: The Gaussian kernel weight for x_var.
    """
    return norm.pdf((x_var - x_i) / bandwidth)


def objective_function(coeff, x_var, y_var):
    """Function for constraint fitting of polynomial"""
    return np.sum((y_var - np.polyval(coeff, x_var)) ** 2)


def weighted_median(data, weights):
    """
    Compute the weighted median of data with weights.
    Assumes data is sorted.
    """
    if not all(isfinite(w) for w in weights):
        return None
    total_weight = sum(weights)
    midpoint = total_weight / 2
    cumulative_weight = 0
    for _, (datum, weight) in enumerate(zip(data, weights)):
        cumulative_weight += weight
        if cumulative_weight >= midpoint:
            return datum
