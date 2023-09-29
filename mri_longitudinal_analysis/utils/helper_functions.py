"""Script containing some additonal functions used thorughout the other main scripts."""
from math import isfinite

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors


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


def pearson_correlation(x_var, y_var):
    """
    Calculate Pearson correlation coefficient between two variables.

    Parameters:
        x, y (array-like): Variables to correlate.

    Returns:
        float: Pearson correlation coefficient.
    """
    return pearsonr(x_var, y_var)[0]


def spearman_correlation(x_var, y_var):
    """
    Calculate Spearman correlation coefficient between two variables.

    Parameters:
        x, y (array-like): Variables to correlate.

    Returns:
        float: Spearman correlation coefficient.
    """
    return spearmanr(x_var, y_var)[0]


def chi_squared_test(x_var, y_var):
    """
    Perform a Chi-Squared test between two categorical variables.

    Parameters:
        var1, var2 (str): Column names for the variables to test.

    Returns:
        float: The Chi-Squared test statistic.
        float: The p-value of the test.
    """
    contingency_table = pd.crosstab(x_var, y_var)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p


def bonferroni_correction(p_values):
    """
    Apply Bonferroni correction to a list of p-values.

    Parameters:
        p_values (list): List of p-values to correct.

    Returns:
        list: Bonferroni corrected p-values.
    """
    return multipletests(p_values, method="bonferroni")[1]


def sensitivity_analysis(merged_data, column, z_threshold=2):
    """
    Perform sensitivity analysis by excluding outliers based on Z-scores.

    Parameters:
        column (str): Column name to analyze.
        z_threshold (float): Z-score threshold for identifying outliers.

    Returns:
        DataFrame: Filtered data.
    """
    z_scores = np.abs(
        (merged_data[column] - merged_data[column].mean()) / merged_data[column].std()
    )
    return merged_data[z_scores < z_threshold]


def propensity_score_matching(merged_data, treatment_col, match_cols):
    """
    Perform propensity score matching based on nearest neighbors.

    Parameters:
        treatment_col (str): Column indicating treatment type.
        match_cols (list): List of columns for matching.

    Returns:
        DataFrame: Matched data.
    """
    treated = merged_data[merged_data[treatment_col] == 1]
    untreated = merged_data[merged_data[treatment_col] == 0]

    nbrs = NearestNeighbors(n_neighbors=1).fit(untreated[match_cols])
    _, indices = nbrs.kneighbors(treated[match_cols])

    treated["matched_index"] = indices
    untreated["index_copy"] = untreated.index
    return pd.merge(
        treated,
        untreated,
        left_on="matched_index",
        right_on="index_copy",
        suffixes=("", "_matched"),
    )
