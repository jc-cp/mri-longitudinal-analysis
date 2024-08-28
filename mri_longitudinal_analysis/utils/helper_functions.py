"""Script containing some additonal functions used thorughout the other main scripts."""
import os
import warnings
from math import isfinite

import matplotlib.lines as lines
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from cfg.utils import helper_functions_cfg
from scikit_posthocs import posthoc_dunn
from scipy.optimize import curve_fit
from scipy.stats import (chi2_contingency, f_oneway, fisher_exact, kruskal,
                         mannwhitneyu, norm, pearsonr, pointbiserialr,
                         spearmanr, ttest_ind, zscore, shapiro, levene)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.interpolate import interp1d

######################################
# SMOOTHING and FILTERING OPERATIONS #
######################################

def fit_linear(x, y):
    """
    Fit a linear model to the data.
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    predictions = model.predict(x.reshape(-1, 1))
    return r2_score(y, predictions)


def exponential_model(x, a, b, c):
    """Description of an exponential model."""
    return a * np.exp(b * x) + c


def fit_exponential(x, y):
    """
    Fit the data to an exponential model and return the goodness of fit (R-squared value).

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data (growth rates).

    Returns:
        float: R-squared value indicating the goodness of fit.
    """
    warnings.filterwarnings("ignore")
    # Scale x to reduce the range and potentially avoid overflow
    x_scaled = x / np.max(x)

    # Use data-driven initial guesses if possible
    a_initial = max(y)
    b_initial = 0  # Start with a flat exponential curve
    c_initial = min(y)
    initial_guesses = [a_initial, b_initial, c_initial]

    # Set more conservative bounds to avoid extreme values
    a_bounds = (0, max(y) * 10)  # Allow a to vary within an order of magnitude
    b_bounds = (-1, 1)  # Restrict the rate of exponential growth/decay
    c_bounds = (min(y) * 0.5, max(y) * 2)  # Allow some variation for baseline shift
    bounds = ([a_bounds[0], b_bounds[0], c_bounds[0]], [a_bounds[1], b_bounds[1], c_bounds[1]])

    try:
        #pylint: disable=unbalanced-tuple-unpacking
        params, _ = curve_fit(exponential_model, x_scaled, y, p0=initial_guesses, bounds=bounds, maxfev=10000)
        predictions = exponential_model(x, *params)
        r2 = r2_score(y, predictions)
        return r2
    except (RuntimeError, OverflowError, ValueError):
        return -1


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


def normalize_data(data):
    """
    Normalize a list of values using min-max scaling.

    Args:
        data (list): List of values to normalize.

    Returns:
        list: Normalized values.
    """
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:  # prevent division by zero
        return [1 for _ in data]  # return a list of ones

    return [(x - min_val) / (max_val - min_val) for x in data]


######################################
# STATISTICAL TESTS and CORRELATIONS #
######################################

def pearson_correlation(x_var, y_var):
    """
    Calculate Pearson correlation coefficient between two variables.

    Parameters:
        x, y (array-like): Variables to correlate.

    Returns:
        float: Pearson correlation coefficient.
    """
    coef, p_val = pearsonr(x_var, y_var)
    return coef, p_val


def spearman_correlation(x_var, y_var):
    """
    Calculate Spearman correlation coefficient between two variables.

    Parameters:
        x, y (array-like): Variables to correlate.

    Returns:
        float: Spearman correlation coefficient.
    """
    coef, p_val = spearmanr(x_var, y_var)
    return coef, p_val


def chi_squared_test(data, x_val, y_val):
    """
    Perform a Chi-Squared test between two categorical variables.

    Parameters:
        var1, var2 (str): Column names for the variables to test.

    Returns:
        float: The Chi-Squared test statistic.
        float: The p-value of the test.
    """
    contingency_table = pd.crosstab(data[x_val], data[y_val])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_val, dof, expected


def bonferroni_correction(p_values, alpha):
    """
    Apply Bonferroni correction to a list of p-values.

    Parameters:
        p_values (list): List of p-values to correct.

    Returns:
        list: Bonferroni corrected p-values.
    """
    return multipletests(p_values, alpha=alpha, method="bonferroni")[1]


def visualize_p_value_bonferroni_corrections(original_p_values, corrected_p_values, alpha, path):
    """
    Scatter and bar plots for the p-value corrections performed through the Bonferroni method.
    """
    plt.scatter(original_p_values, corrected_p_values, alpha=0.7, cmap="viridis")
    plt.axhline(y=alpha, color="r", linestyle="--", label=f"Alpha={alpha}")
    plt.xlabel("Original P-Values")
    plt.ylabel("Bonferroni Corrected P-Values")
    plt.title("Effect of Bonferroni Correction on P-Values")
    plt.legend()
    filename_scatter = os.path.join(path, "p_value_bonferroni_corrections.png")
    plt.savefig(filename_scatter)
    plt.close()

    indices = range(len(original_p_values))
    plt.figure(figsize=(10, 5))
    plt.bar(indices, original_p_values, alpha=0.7, label="Original P-Values", color="blue")
    plt.bar(
        indices, corrected_p_values, alpha=0.7, label="Bonferroni Corrected P-Values", color="green"
    )
    plt.axhline(
        y=alpha, color="red", linestyle="--", label=f"Significance Threshold (Alpha={alpha})"
    )
    plt.xlabel("Test Indices")
    plt.ylabel("P-Values")
    plt.yscale("log")  # Log scale to better visualize small p-values
    plt.title("Original vs Bonferroni Corrected P-Values")
    plt.legend()
    filename_bar = os.path.join(path, "p_value_bonferroni_comparison.png")
    plt.savefig(filename_bar)
    plt.close()


def fdr_correction(p_values, alpha=0.05):
    """
    Apply False Discovery Rate (FDR) correction to a list of p-values.

    Parameters:
        p_values (list): List of p-values to correct.
        alpha (float): Significance level, default is 0.05.

    Returns:
        tuple: FDR corrected p-values and the boolean array of which hypotheses are rejected.
    """
    is_rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return corrected_p_values, is_rejected


def visualize_fdr_correction(original_p_values, corrected_p_values, is_rejected, alpha, path):
    """
    Scatter and bar plots for the p-value corrections performed through the FDR method.
    """
    indices = range(len(original_p_values))
    plt.figure(figsize=(10, 5))
    plt.bar(indices, original_p_values, alpha=0.7, label="Original P-Values")
    plt.bar(indices, corrected_p_values, alpha=0.7, label="FDR Corrected P-Values", color="red")
    plt.scatter(indices, is_rejected, color="green", label="Rejected Hypotheses", zorder=3)
    plt.axhline(
        y=alpha, color="blue", linestyle="--", label=f"Significance Threshold (Alpha={alpha})"
    )
    plt.xlabel("Test Indices")
    plt.ylabel("P-Values")
    plt.yscale("log")  # Log scale for visibility of small p-values
    plt.title("Original vs FDR Corrected P-Values")
    plt.legend()
    filename = os.path.join(path, "p_value_fdr_correction.png")
    plt.savefig(filename)
    plt.close()


def sensitivity_analysis(data, variable, z_threshold=2):
    """
    Perform a sensitivity analysis by removing outliers based on a Z-score threshold.

    Parameters:
        data (DataFrame): The data to analyze.
        variable (str): The name of the variable to check for outliers.
        z_threshold (float): The Z-score threshold to identify outliers.

    Returns:
        DataFrame: The data with outliers removed.
    """
    # Calculate Z-scores for the specified variable
    data["Z_score"] = np.abs(zscore(data[variable], nan_policy="omit"))

    # Filter out outliers
    no_outliers = data[data["Z_score"] < z_threshold]

    # Drop the Z-score column as it's no longer needed
    no_outliers = no_outliers.drop(columns=["Z_score"])

    # Return the data with outliers removed
    return no_outliers


def calculate_propensity_scores(data, treatment_column, covariate_columns):
    """
    Calculate propensity scores using logistic regression to find cofounding variables.
    """
    if data[treatment_column].nunique() != 2:
        raise ValueError(
            f"Not enough classes present in {treatment_column}. "
            "Make sure the column contains both treated and untreated indicators."
        )

    # Scale covariates to improve logistic regression performance
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(data[covariate_columns])

    # Fit logistic regression to calculate propensity scores
    lr = LogisticRegression()
    lr.fit(covariates_scaled, data[treatment_column])
    propensity_scores = lr.predict_proba(covariates_scaled)[:, 1]

    return propensity_scores


def perform_propensity_score_matching(
    data, propensity_scores, treatment_column, match_ratio=1, caliper=None
):
    """
    Perform the actual matching using propensity scores.
    """
    # Add propensity scores to the dataframe
    data = data.copy()
    data["propensity_score"] = propensity_scores

    # Separate treatment and control groups
    treatment = data[data[treatment_column] == 1]
    control = data[data[treatment_column] == 0].copy()

    # Reset index to ensure alignment
    control.reset_index(drop=True, inplace=True)

    # Nearest neighbor matching with caliper
    nn = NearestNeighbors(n_neighbors=match_ratio, radius=caliper)
    nn.fit(control[["propensity_score"]].values)

    matched_indices = []
    for _, row in treatment.iterrows():
        query_features = row[["propensity_score"]].values.reshape(1, -1)
        distances, indices = nn.radius_neighbors(query_features)
        for distance, index_array in zip(distances[0], indices[0]):
            if distance <= caliper:
                matched_indices.append(control.iloc[index_array].name)

    # Create the matched dataset
    matched_data = pd.concat([treatment, control.loc[matched_indices]]).drop_duplicates()

    return matched_data


def calculate_smd(groups, covariate, treatment_column):
    """
    Calculate the standardized mean difference (SMD) for a single covariate.

    Parameters:
        groups (DataFrame): The dataset containing treatment and control groups.
        covariate (str): The name of the covariate to calculate SMD for.

    Returns:
        float: The SMD for the covariate.
    """
    mean_treatment = groups[covariate][groups[treatment_column] == 1].mean()
    mean_control = groups[covariate][groups[treatment_column] == 0].mean()
    std_pooled = np.sqrt(
        (
            groups[covariate][groups[treatment_column] == 1].var()
            + groups[covariate][groups[treatment_column] == 0].var()
        )
        / 2
    )
    smd = (mean_treatment - mean_control) / std_pooled
    return smd


def visualize_smds(balance_df, path):
    """
    Bar plot for the squared mean differences (SMDs) of covariates after matching.
    """
    balance_df.plot(kind="bar")
    plt.axhline(y=0.1, color="r", linestyle="--")
    plt.title("Standardized Mean Differences for Covariates After Matching")
    plt.ylabel("Standardized Mean Difference")
    plt.xlabel("Covariates")
    plt.tight_layout()
    filename = os.path.join(path, "smds.png")
    plt.savefig(filename)
    plt.close()


def check_balance(matched_data, covariate_columns, treatment_column):
    """
    Check the balance of covariates in the matched dataset using
    Standardized Mean Differences (SMD).

    Parameters:
        matched_data (DataFrame): The matched dataset after performing PSM.
        covariate_columns (list of str): A list of covariate column names.

    Returns:
        DataFrame: A dataframe with SMD for all covariates.
    """
    balance = {
        covariate: calculate_smd(matched_data, covariate, treatment_column)
        for covariate in covariate_columns
    }
    balance_df = pd.DataFrame.from_dict(balance, orient="index", columns=["SMD"])
    balanced = balance_df["SMD"].abs() < 0.1
    print("\tCovariate balance after matching:")
    print(balance_df)
    print("\n\tBalanced covariates:")
    print(balanced)
    return balance_df


def ttest(data, x_val, y_val):
    """
    T-test between two groups.
    """
    group1 = data[data[x_val] == data[x_val].unique()[0]][y_val]
    group2 = data[data[x_val] == data[x_val].unique()[1]][y_val]
    t_stat, p_val = ttest_ind(group1.dropna(), group2.dropna())
    return t_stat, p_val


def f_one(data, x_val, y_val):
    """
    F-1 test between groups.
    """
    groups = [group[y_val].dropna() for name, group in data.groupby(x_val)]
    f_stat, p_val = f_oneway(*groups)
    return f_stat, p_val


def point_bi_serial(data, binary_var, continuous_var):
    """
    Point-biserial correlation between a binary and a continuous variable.
    """
    binary_data = data[binary_var].cat.codes
    continuous_data = data[continuous_var]
    coef, p_val = pointbiserialr(binary_data, continuous_data)
    return coef, p_val


def compute_95_ci(data):
    """
    Compute the 95% confidence interval for a dataset.

    Args:
        data (list): List of data points.

    Returns:
        tuple: Lower and upper bounds of the 95% CI.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    se = std_dev / np.sqrt(len(data))

    lower_bound = mean - 1.96 * se
    upper_bound = mean + 1.96 * se

    return lower_bound, upper_bound


def kruskal_wallis_test(data, x_val, y_val):
    """
    Perform Kruskal-Wallis test for a given numerical variable across different groups.
    """
    groups = [group[y_val].dropna() for name, group in data.groupby(x_val)]
    if len(groups) > 1:
        test_stat, p_val = kruskal(*groups)
        print(f"\t\tKruskal-Wallis Test on {y_val} across {x_val}: Statistic={test_stat}, P-value={p_val}")
        return test_stat, p_val
    else:
        print("\t\tNot enough groups for Kruskal-Wallis Test.")
        return None, None


def fisher_exact_test(data, x_val, y_val):
    """
    Perform Fisher's Exact test for 2x2 contingency tables.
    """
    contingency_table = pd.crosstab(data[x_val], data[y_val])
    if contingency_table.shape == (2, 2):  # Fisher's Exact test is applicable only for 2x2 tables
        odds_ratio, p_val = fisher_exact(contingency_table)
        print(f"\t\tFisher's Exact Test between {x_val} and {y_val}: P-value={p_val}")
        return odds_ratio, p_val
    else:
        print("\t\tFisher's Exact Test requires a 2x2 contingency table.")
        return None, None


def mann_whitney_u_test(data, var, cohort_var):
    """
    Perform the Mann-Whitney U test to compare two independent samples.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - var (str): The variable to be compared between the two cohorts.
    - cohort_var (str): The variable defining the cohorts.

    Returns:
    - u_stat (float): The Mann-Whitney U statistic.
    - p_val (float): The p-value associated with the test.
    """
    cohort1_data = data[data[cohort_var] == data[cohort_var].unique()[0]]
    cohort2_data = data[data[cohort_var] == data[cohort_var].unique()[1]]

    u_stat, p_val = mannwhitneyu(cohort1_data[var], cohort2_data[var])

    return u_stat, p_val


def logistic_regression_analysis(y, x, regularization=None, C=1.0):
    """
    Perform logistic regression to analyze the impact of various factors on tumor progression.
    """
    if regularization == 'l1':
        model_result = sm.Logit(y,x).fit_regularized(method='l1', alpha=1/C, disp=0, maxiter=100)
    else:
        model_result = sm.Logit(y, x).fit(disp=0, maxiter=100, method='lbfgs')
    return model_result


def stepwise_selection(y, x):
    """
    Perform stepwise variable selection for logistic regression.
    """    
    # Perform forward stepwise selection
    selected_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0, maxiter=100, method='lbfgs')
    remaining_vars = set(x.columns)
    
    while len(remaining_vars) > 0:
        best_score = np.inf
        best_var = None
        
        for var in remaining_vars:
            candidate_model = sm.Logit(y, sm.add_constant(x[[var]])).fit(disp=0, maxiter=100, method='lbfgs')
            score = candidate_model.aic
            if score < best_score:
                best_score = score
                best_var = var
        
        if best_var is not None:
            selected_model = sm.Logit(y, sm.add_constant(x[selected_model.params.index.tolist() + [best_var]])).fit(disp=0, maxiter=100, method='lbfgs')
            remaining_vars.remove(best_var)
        else:
            break
    
    return selected_model


def calculate_vif(X, categorical_columns):
    """
    Calculate Variance Inflation Factor (VIF) for each variable in the DataFrame X.
    X should already have dummy variables for categorical features and should not contain the outcome variable.
    """
    #print("Data types before VIF calculation:")
    #print(X.dtypes)
    #print("\nData description:")
    #print(X.describe())
    X = X.copy()
    # Convert boolean columns to integer
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    
    # Replace inf values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    
    #print("\nData types after conversion and cleaning:")
    #print(X.dtypes)
    #print("\nData shape after cleaning:", X.shape)
    
    # Identify dummy variables
    dummy_columns = [col for col in X.columns if any(col.startswith(cat + '_') for cat in categorical_columns)]
    
    # Identify continuous variables
    continuous_columns = [col for col in X.columns if col not in dummy_columns]
    
    vif_data = []
    
    # Calculate VIF for continuous variables
    for i, col in enumerate(continuous_columns):
        try:
            vif = variance_inflation_factor(X[continuous_columns].values, i)
            vif_data.append({'Variable': col, 'VIF': vif})
        except ExceptionGroup as e:
            print(f"Error calculating VIF for {col}: {e}")
    
    # Calculate VIF for each set of dummy variables
    for cat in categorical_columns:
        cat_dummies = [col for col in dummy_columns if col.startswith(cat + '_')]
        try:
            X_with_cat = X[continuous_columns + cat_dummies]
            last_column_index = X_with_cat.shape[1] - 1
            cat_vif = variance_inflation_factor(X_with_cat.values, last_column_index)
            vif_data.append({'Variable': cat, 'VIF': cat_vif})
        except ExceptionGroup as e:
            print(f"Error calculating VIF for {cat}: {e}")
    
    vif_df = pd.DataFrame(vif_data)
    print("\nVIF Calculation Results:")
    print(vif_df)
    return vif_df


def cumulative_stats(group, variable):
    """
    Calculates cumulative statistics for a given variable.
    """
    group[f"{variable} CumMean"] = group[variable].expanding().mean()
    group[f"{variable} CumMedian"] = group[variable].expanding().median()
    group[f"{variable} CumStd"] = group[variable].expanding().std().fillna(0)
    return group


def rolling_stats(group, variable, window_size=3, min_periods=1):
    """
    Calculates rolling statistics for a given variable.
    """
    group[f"{variable} RollMean"] = (
        group[variable]
        .rolling(window=window_size, min_periods=min_periods)
        .mean()
    )
    group[f"{variable} RollMedian"] = (
        group[variable]
        .rolling(window=window_size, min_periods=min_periods)
        .median()
    )
    group[f"{variable} RollStd"] = (
        group[variable]
        .rolling(window=window_size, min_periods=min_periods)
        .std()
        .fillna(0)
    )
    return group

#######################################
# DATA HANDLING and SIMPLE OPERATIONS #
#######################################


def prefix_zeros_to_six_digit_ids(patient_id):
    """
    Adds 0 to the beginning of 6-digit patient IDs.
    """
    str_id = str(patient_id)
    if len(str_id) == 6:
        # print(f"Found a 6-digit ID: {str_id}. Prefixing a '0'.")
        patient_id = "0" + str_id

    else:
        patient_id = str_id
    return patient_id


def calculate_stats(row, col_name):
    """
    Given a row of data, calculate the mean, median, and standard deviation of a column.
    """
    values = row[col_name]
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    return pd.Series(
        [mean_val, median_val, std_val],
        index=[f"{col_name}_mean", f"{col_name}_median", f"{col_name}_std"],
    )


def zero_fill(series, width):
    """
    Given a series of strings, zero-fill the values to a specified width in front.
    """
    return series.astype(str).str.zfill(width)


def save_dataframe(df: pd.DataFrame, output_dir: str, cohort: str):
    """
    Save data for deep learning in csv format.
    """
    if df is not None:
        filename = f"{cohort}_cohort_data_features.csv"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"\tData saved for further analyses in {file_path}.")


def categorize_age_group(data, column):
    """
    Categorize patients according to an age group for more thorough analysis.
    """
    age_days = data[column]
    age_years = age_days / 365.25  # Convert age to years
    #print(f"Age in days: {age_days}, Age in years: {age_years}")  # Debug print
    if age_years <= 2:
        category =  "Infant"
    elif age_years <= 6:
        category =  "Preschool"
    elif age_years <= 13:
        category =  "School Age"
    # elif age_years <= 18:
    #     category =  "Adolescent"
    else:
        #category = "Young Adult"
        category =  "Adolescent"
    
    return category


def categorize_time_since_first_diagnosis(data, column="Age"):
    """
    Function that calculates the time since first diagnosis and categorizes in years.
    """
    age_at_diagnosis = data["Age at First Diagnosis"] / 365
    age = data[column] /365
    time_since_diagnosis = age - age_at_diagnosis
    
    if time_since_diagnosis <= 1:
        return "0-1 years"
    elif time_since_diagnosis <= 3:
        return "1-3 years"
    elif time_since_diagnosis <= 5:
        return "3-5 years"
    # elif time_since_diagnosis <= 10:
    #     return "5-10 years"
    else:
        return "5+ years"
    
    
def calculate_group_norms_and_stability(data, volume_column, volume_change_column):
    """
    Calculate group-wise statistics for each age group based on the
    volume and volume change columns.
    """
    # Use mean to reflect average variability within each group
    group_norms = (
        data.groupby("Age Group")
        .agg(
            {
                f"{volume_column} RollStd": "mean",
                f"{volume_change_column} RollStd": "mean",
            }
        )
        .reset_index()
    )

    data = data.merge(group_norms, on="Age Group", suffixes=("", " GroupNorm"))
    data["Volume Stability Score"] = (
        data[f"{volume_column} RollStd"] / data[f"{volume_column} RollStd GroupNorm"]
    )
    data["Growth Stability Score"] = (
        data[f"{volume_change_column} RollStd"] / data[f"{volume_change_column} RollStd GroupNorm"]
    )
    data["Period"] = data["Age"]

    return data


def calculate_slope_and_angle(data, patient_id, column_name):
    """
    Calculate the slope of the trend line for a patient's data.
    """
    patient_data = data[data["Patient_ID"] == patient_id]
    if len(patient_data) < 2:
        return None  # Not enough data to calculate a slope

    # Reshape for sklearn
    x = patient_data["Time since First Scan"].values.reshape(-1, 1)
    y = patient_data[column_name].values

    # Fit linear model
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    angle = np.arctan(slope) * 180 / np.pi

    return slope, angle


def classify_patient_volumetric(
    data,
    patient_id,
):
    """
    Classify a patient based on the 'Age at First Progression' and 'Age at First Regression' data.
    """
    patient_data = data[data["Patient_ID"] == patient_id]
    
    if len(patient_data) == 0:
        return None

    age_at_first_progression = patient_data["Age at First Progression"].iloc[0]
    age_at_first_regression = patient_data["Age at First Regression"].iloc[0]
    
    if pd.notna(age_at_first_progression):
        return "Progressor"
    elif pd.notna(age_at_first_regression):
        return "Regressor"
    else:
        return "Stable"


def classify_patient_composite(data,patient_id):
    """
    Classify a patient based on volumetric progression and treatment received.
    """
    patient_data = data[data["Patient_ID"] == patient_id]

    volumetric_class = classify_patient_volumetric(data, patient_id)
    received_treatment = (patient_data["Received Treatment"] == "Yes").all()
   
    if volumetric_class == "Progressor" or received_treatment:
        return "Progressor"
    else:
        return "Non-progressor"


def calculate_percentage_change(data, patient_id, column_name):
    """
    Calculate the percentage change in a volume for a patient
    considering the first and last value.
    """
    patient_data = data[data["Patient_ID"] == patient_id].sort_values(by="Time since First Scan")

    start_volume = patient_data[column_name].iloc[0]
    end_volume = patient_data[column_name].iloc[-1]

    if start_volume == 0:
        return None  # Avoid division by zero

    percent_change = ((end_volume - start_volume) / start_volume) * 100
    return percent_change


def read_exclusion_list(file_path):
    """
    Reads a file containing patient IDs and their scan IDs to be excluded.

    :param file_path: Path to the .txt file.
    :return: A set of patient IDs to exclude.
    """
    exclude_patients = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("----"):
                patient_id = line.strip()
                exclude_patients.add(patient_id)
    return exclude_patients


def consistency_check(data):
    """
    Check for consistency amongst classification in the data.
    """
    mismatched_patients = []

    for patient_id, group in data.groupby("Patient_ID"):
        first_row = group.iloc[0]
        last_row = group.iloc[-1]

        patient_classification = first_row["Patient Classification Volumetric"]
        initial_tumor_classification = first_row["Tumor Classification"]
        final_tumor_classification = last_row["Tumor Classification"]

        # Define your matching logic here
        if patient_classification == "Regressor":
            if not final_tumor_classification == "Unstable":
                mismatched_patients.append(patient_id)
        elif patient_classification == "Progressor":
            if not final_tumor_classification == "Unstable":
                mismatched_patients.append(patient_id)
        elif patient_classification == "Stable":
            if not initial_tumor_classification == "Stable":
                mismatched_patients.append(patient_id)

    # Output or handle mismatched_patients
    if len(mismatched_patients) != 0:
        print(f"\tMismatched Patients: {mismatched_patients}")
    else:
        print("\tAll patients were rightly classified.")


def check_assumptions(x_val, y_val, data, test_type):
    """
    Function to check the assumptions of the statistical tests.

    Parameters:
    - x_val (str): The name of the first variable.
    - y_val (str): The name of the second variable.
    - data (DataFrame): The data containing the variables.
    - test_type (str): The statistical method to be used.

    Returns:
    - bool: True if assumptions are met, False otherwise.
    """
    if test_type in ["t-test", "ANOVA"]:
        categories = data[x_val].unique()
        if len(categories) < 2:
            print(f"Not enough categories in {x_val} for {test_type}")
            return False
        
        groups = [data[data[x_val] == category][y_val].dropna() for category in categories]
        if any(len(group) < 3 for group in groups):
            print(f"One or more groups have less than 3 samples for {x_val} and {y_val}")
            return False
        
        normality_tests = [shapiro(group)[1] > 0.05 for group in groups]
        equal_variance_test = levene(*groups)[1] > 0.05
        
        assumptions_met = all(normality_tests) and equal_variance_test
        if not assumptions_met:
            print(f"Assumptions not met for {test_type} on {x_val} and {y_val}")
        return assumptions_met

    elif test_type in ["Spearman", "Pearson"]:
        x_data = data[x_val].dropna()
        y_data = data[y_val].dropna()
        if len(x_data) < 3 or len(y_data) < 3:
            print(f"Not enough samples for {test_type} on {x_val} and {y_val}")
            return False
        
        _, p_norm_x = shapiro(x_data)
        _, p_norm_y = shapiro(y_data)
        
        assumptions_met = p_norm_x > 0.05 and p_norm_y > 0.05
        if not assumptions_met:
            print(f"Normality assumption not met for {test_type} on {x_val} and {y_val}")
        return assumptions_met

    else:
        print(f"No assumption check defined for {test_type}")
        return True

######################################
# VISUALIZATION and PLOTTING METHODS #
######################################


def visualize_tumor_stability(data, output_dir, stability_threshold, change_threshold):
    """
    Create a series of plots to visualize the stability index and tumor classification.
    """
    classification_distribution = data["Tumor Classification"].value_counts(normalize=True)
    sns.set_palette(helper_functions_cfg.NORD_PALETTE)
    ############
    # BAR PLOT #
    #############
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Age Group", hue="Tumor Classification", data=data)
    plt.title("Count of Stable vs. Unstable Tumors Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    annotate_plot(plt.gca())
    plt.legend(title="Tumor Classification")
    plt.tight_layout()
    filename = os.path.join(output_dir, "tumor_classification.png")
    plt.savefig(filename)
    plt.close()

    ############
    # PIE PLOT #
    ############
    plt.figure(figsize=(10, 6))
    plt.pie(
        classification_distribution,
        labels=classification_distribution.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette(helper_functions_cfg.PIE_PALETTE),
    )
    plt.title("Proportion of Stable vs. Unstable Tumors")
    plt.axis("equal")
    filename_dist = os.path.join(output_dir, "tumor_classification_distribution.png")
    plt.savefig(filename_dist)
    plt.close()

    # Enhanced Scatter Plot with Labels for Extremes
    plt.figure(figsize=(18, 6))
    ax = sns.scatterplot(
        x="Overall Volume Change",
        y="Stability Index",
        hue="Tumor Classification",
        data=data,
        alpha=0.6,
    )
    extremes = data.nlargest(4, "Stability Index")  # Adjust the number of points as needed
    for _, point in extremes.iterrows():
        ax.text(
            point["Overall Volume Change"],
            point["Stability Index"],
            str(point["Patient_ID"]),
        )
    plt.title("Scatter Plot of Stability Index Over Time by Classification")
    plt.xlabel("Overall Volume Change [%]")
    plt.ylabel("Stability Index")
    plt.axhline(y=stability_threshold, color="g", linestyle="--", label="Stability Threshold")
    plt.axvline(x=change_threshold, color="b", linestyle="--", label="Volume Change Threshold")
    plt.axvline(x=-change_threshold, color="b", linestyle="--")
    plt.legend(title="Tumor Classification")
    plt.tight_layout()
    filename_scatter = os.path.join(output_dir, "stability_index_scatter.png")
    plt.savefig(filename_scatter)
    plt.close()

    ###############
    # VIOLIN PLOT #
    ###############
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Age Group", y="Stability Index", data=data, bw=0.2)
    sns.stripplot(x="Age Group", y="Stability Index", data=data, color="k")
    plt.title("Distribution of Stability Index Across Age Groups")
    plt.ylabel("Stability Index")
    plt.xlabel("Age Group")
    plt.tight_layout()
    filename_distribution = os.path.join(output_dir, "stability_index_distribution.png")
    plt.savefig(filename_distribution)
    plt.close()

    ################
    # SCATTER PLOT #
    ################
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x="Volume Change",
        y="Stability Index",
        hue="Tumor Classification",
        style="Tumor Classification",
        data=data,
    )
    plt.title("Stability Index Over Volume Change")
    plt.ylabel("Stability Index")
    plt.xlabel("Volume Change [%]")
    plt.legend(title="Tumor Classification")
    plt.tight_layout()
    filename_scatter = os.path.join(output_dir, "stability_index_over_volume_change.png")
    plt.savefig(filename_scatter)
    plt.close()


def annotate_plot(a_x):
    """Annotate the bar plot with the respective heights.

    Args:
        a_x (matplotlib.axis): The axis object to be annotated.
    """
    for patch in a_x.patches:
        height = patch.get_height()
        a_x.text(
            x=patch.get_x() + (patch.get_width() / 2),
            y=height,
            s=f"{height:.0f}",
            ha="center",
        )


def plot_histo_distributions(data, output_dir, time_bins, age_groups, endpoint="volumetric"):
    """
    Several distributions and histograms.
    """
    data = data.copy()
    dir_name = os.path.join(output_dir, "progression_status")
    os.makedirs(dir_name, exist_ok=True)
    if endpoint == "volumetric":
        data['Progressed'] = data.apply(lambda row: 1 if classify_patient_volumetric(data, row['Patient_ID']) == "Progressor" else 0, axis=1)
    else:   # clinical, composite progression
        data['Progressed'] = data.apply(lambda row: 1 if classify_patient_composite(data, row['Patient_ID']) == 'Progressor' else 0, axis=1)
    
    data['Previously Progressed'] = 0
    cv_distribution(data, dir_name, age_groups=age_groups)
    plot_histo_progression(data, dir_name, time_bins, endpoint)    
    plot_histo_age_group(data, dir_name, age_groups, endpoint)
    plot_modified_progression(data, dir_name, age_groups, endpoint)
    plot_modified_progression(data, dir_name, time_bins, endpoint)

def cv_distribution(data, output_dir, age_groups):
    """
    Plot the distribution of the coefficient of variation.
    """
    data["CV"] = data.groupby("Patient_ID")["Volume"].transform(lambda x: x.std() / x.mean())
    
    median_cv_per_group = data.groupby("Age Group at Diagnosis")["CV"].median().sort_index()
    # Kruskal-Wallis H-test
    age_group_data = [group for _, group in data.groupby("Age Group at Diagnosis")["CV"]]
    h_statistic, p_value = kruskal(*age_group_data)
    print("\nKruskal-Wallis H-test Results:")
    print(f"H-statistic: {h_statistic}")
    print(f"p-value: {p_value}")
    print(f"Median CV per group: {median_cv_per_group}")
    if p_value < 0.05:
        posthoc_results = posthoc_dunn(data, val_col='CV', group_col='Age Group at Diagnosis', p_adjust='bonferroni')
        print("\nPost-hoc Dunn's test results (p-values):")
        print(posthoc_results)
    
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, n_colors=len(age_groups)) 
    # Create boxplots for the coefficient of variation for each age group
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    sns.boxplot(x="Age Group at Diagnosis", y="CV", data=data, order=age_groups, hue="Age Group at Diagnosis", palette=palette,)
    plt.xlabel("Age Group at Diagnosis", fontdict={"size": 15}, labelpad=25)
    plt.ylabel("Coefficient of Variation", fontdict={"size": 15})
    plt.title("Distribution of Coefficient of Variation by Age Group", fontdict={"size": 20})
    
    plt.tight_layout()
    file_name_cv_ = os.path.join(output_dir, "coefficient_of_variation_by_age_group_boxplots.png")
    plt.savefig(file_name_cv_, dpi=300)
    plt.close()
    
    cv_data = data.groupby("Patient_ID")["Coefficient of Variation"].first()
    cv_mean = cv_data.mean()
    cv_median = cv_data.median()
    cv_std = cv_data.std()
    plt.figure(figsize=(10, 7))
    sns.histplot(cv_data, bins=20, kde=True)
    plt.xlabel("Coefficient of Variation", fontdict={"size": 15})
    plt.ylabel("Count", fontdict={"size": 15})
    plt.title("Distribution of Coefficient of Variation", fontdict={"size": 20})
    plt.axvline(cv_mean, color='red', linestyle='--', label=f'Mean: {cv_mean:.3f}')
    plt.axvspan(cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, color='red', label=f'Std Dev: ±{cv_std:.3f}')
    plt.axvline(cv_median, color='green', linestyle='-.', label=f'Median: {cv_median:.3f}')
    plt.legend()
    file_name_cv = os.path.join(output_dir, "coefficient_of_variation.png")
    plt.savefig(file_name_cv, dpi=300)
    plt.close()

def plot_histo_age_group(data, output_dir, age_groups, endpoint):
    """
    Histogram of progression status by age group with properly aligned trend lines.
    """
    classified_data = classify_patients_age_group(data, age_groups)
    
    # Prepare data for plotting
    plot_data = pd.melt(classified_data, id_vars=['Age Group'], 
                        value_vars=['Not Progressed', 'Progressed', 'Cumulative Progression'],
                        var_name='Status', value_name='Count')
    plot_data['Total'] = plot_data.groupby('Age Group')['Count'].transform('sum')

    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Bar plot
    bar_plot = sns.barplot(x="Age Group", y="Count", hue="Status", data=plot_data, order=age_groups, ax=ax)
    
    # Calculate bar centers for each group
    n_groups = len(age_groups)
    bar_width = bar_plot.containers[0][0].get_width()
    
    x_positions = np.arange(n_groups)

    # Trend lines
    for i, status in enumerate(['Not Progressed', 'Progressed', 'Cumulative Progression']):
        status_data = plot_data[plot_data['Status'] == status]
        y = status_data['Count'].values
        x = x_positions + (i-1) * bar_width
        
        plt.plot(x, y, marker='o', label=f"{status} Trend")

    # Customize the plot
    plt.xlabel("Age Group", fontdict={"size": 15})
    plt.ylabel("Count", fontdict={"size": 15})
    plt.title("Patient Progression Status by Age Group", fontdict={"size": 20})
    plt.legend(title='Status')

    # Add total count labels above bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.5, f'{int(height)}',
                ha="center", va="bottom")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    file_name = os.path.join(output_dir, f"{endpoint}_patient_progression_age_group.png")
    plt.savefig(file_name, dpi=300)

def plot_histo_progression(data, output_dir, time_bins, endpoint):
    """
    Histogram of progression status.
    """

    classified_data = classify_patients_time_since_diagnosis(data, time_bins)
    
    # Prepare data for plotting
    plot_data = pd.melt(classified_data, id_vars=['Time Since Diagnosis'], value_vars=['Not Progressed', 'Progressed', 'Cumulative Progression'],
                        var_name='Status', value_name='Count')
    plot_data['Total'] = plot_data.groupby('Time Since Diagnosis')['Count'].transform('sum')

    # Create the triple bar plot with total count annotations
    plt.figure(figsize=(10, 8))
    barplot = sns.barplot(x="Time Since Diagnosis", y="Count", hue="Status", data=plot_data, order=time_bins)
    # Calculate bar centers for each group
    n_groups = len(time_bins)
    bar_width = barplot.containers[0][0].get_width()
    x_positions = np.arange(n_groups)

    # Trend lines
    for i, status in enumerate(['Not Progressed', 'Progressed', 'Cumulative Progression']):
        status_data = plot_data[plot_data['Status'] == status]
        y = status_data['Count'].values
        x = x_positions + (i-1) * bar_width
        
        plt.plot(x, y, marker='o', label=f"{status} Trend")
    
    # Add total count labels above bars
    for p in barplot.patches:
        height = p.get_height()
        barplot.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    plt.xlabel("Time Since Diagnosis", fontdict={"size": 15})
    plt.ylabel("Count", fontdict={"size": 15})
    plt.title("Patient Progression Status Over Time", fontdict={"size": 20})
    plt.legend(title='Status')
    file_name = os.path.join(output_dir, f"{endpoint}_patient_progression_status.png")
    plt.savefig(file_name, dpi=300)

def classify_patients_time_since_diagnosis(data, time_bins):
    """
    Classifies patients based on their progression status at different time points since diagnosis.
    """
    classifications = []
    patient_ids_previously_progressed = set()
    all_patient_ids = set(data['Patient_ID'].unique())

    for idx, time_bin in enumerate(time_bins):        
        # Filter patients for the current time bin
        bin_data = data[data['Time Since Diagnosis'] == time_bin].copy()

        # Update the status of patients who have progressed in the current time bin
        progressed_patient_ids = set(bin_data[bin_data['Progressed'] == 1]['Patient_ID'])
        progressed_count = len(progressed_patient_ids - patient_ids_previously_progressed)
        
        not_progressed_patient_ids = all_patient_ids - patient_ids_previously_progressed - progressed_patient_ids
        not_progressed_count = len(not_progressed_patient_ids)
        
        # Update the Cumulative Progression patients
        cumulative_progression_count = len(patient_ids_previously_progressed) + progressed_count

        classifications.append({
            'Time Since Diagnosis': time_bin,
            'Not Progressed': not_progressed_count,
            'Progressed': progressed_count,
            'Cumulative Progression': cumulative_progression_count
        })

        # Update the set of Cumulative Progression patients
        patient_ids_previously_progressed.update(progressed_patient_ids)

    return pd.DataFrame(classifications)

def classify_patients_age_group(data, age_groups):
    """
    Classifies patients based on their progression status in different age groups.
    """
    classifications = []
    patient_ids_previously_progressed = set()
    all_patient_ids = set(data['Patient_ID'].unique())

    for idx, age_group in enumerate(age_groups):
        # Filter patients for the current age group
        group_data = data[data['Age Group'] == age_group].copy()

        # Update the status of patients who have progressed in the current age group
        progressed_patient_ids = set(group_data[group_data['Progressed'] == 1]['Patient_ID'])
        progressed_count = len(progressed_patient_ids - patient_ids_previously_progressed)
        
        not_progressed_patient_ids = all_patient_ids - patient_ids_previously_progressed - progressed_patient_ids
        not_progressed_count = len(not_progressed_patient_ids)

        # Update the Cumulative Progression patients
        cumulative_progression_count = len(patient_ids_previously_progressed) + progressed_count

        classifications.append({
            'Age Group': age_group,
            'Not Progressed': not_progressed_count,
            'Progressed': progressed_count,
            'Cumulative Progression': cumulative_progression_count
        })

        # Update the set of Cumulative Progression patients
        patient_ids_previously_progressed.update(progressed_patient_ids)

    return pd.DataFrame(classifications)

def plot_modified_progression(data, output_dir, variables, endpoint):
    """
    Refined modified histogram of progression status by age group with adjusted shaded areas and trend lines.
    """
    if "Infant" in variables: 
        classified_data = classify_patients_age_group(data, variables)
        id_vars = 'Age Group'
    else:
        classified_data = classify_patients_time_since_diagnosis(data, variables)
        id_vars = 'Time Since Diagnosis'
    
    # Prepare data for plotting
    plot_data = pd.melt(classified_data, id_vars=[id_vars], 
                        value_vars=['Not Progressed', 'Progressed', 'Cumulative Progression'],
                        var_name='Status', value_name='Count')
    plot_data['Total'] = plot_data.groupby(id_vars)['Count'].transform('sum')

    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Calculate positions
    n_groups = len(variables)
    bar_width = 0.5
    x_positions = np.arange(n_groups)
    
    # Calculate the start and end points for shading
    x_start = x_positions[0] - bar_width/2
    x_end = x_positions[-1] + bar_width/2
    x_shade = np.linspace(x_start, x_end, 100)  # More points for smoother shading
    
    # Prepare data for each status
    status_data = {status: plot_data[plot_data['Status'] == status].set_index(id_vars)['Count'] 
                   for status in ['Not Progressed', 'Progressed', 'Cumulative Progression']}

    # Ensure all status data have values for all age groups
    for status in status_data:
        status_data[status] = status_data[status].reindex(variables, fill_value=0)

    # Define colors from NORD_PALETTE
    colors = {
        'Not Progressed': helper_functions_cfg.NORD_PALETTE[0],
        'Progressed': helper_functions_cfg.NORD_PALETTE[1],
        'Cumulative Progression': helper_functions_cfg.NORD_PALETTE[2]
    }

    # Interpolate data for smooth shading
    def interpolate_data(data):
        return np.interp(x_shade, x_positions, data)

    # Shaded areas
    plt.fill_between(x_shade, 0, interpolate_data(status_data['Not Progressed']), 
                     alpha=0.3, color=colors['Not Progressed'], label='Not Progressed')
    plt.fill_between(x_shade, 0, interpolate_data(status_data['Cumulative Progression']), 
                     alpha=0.3, color=colors['Cumulative Progression'], label='Cumulative Progression')

    # Bar plot for 'Progressed'
    plt.bar(x_positions, status_data['Progressed'], color=colors['Progressed'], 
            alpha=0.7, label='Progressed', width=bar_width)

    # Trend lines
    for status in ['Not Progressed', 'Progressed', 'Cumulative Progression']:
        plt.plot(x_positions, status_data[status], marker='o', color=colors[status], label=f"{status} Trend")

    # Add numbers with increased size and higher position
    for status in ['Not Progressed', 'Progressed', 'Cumulative Progression']:
        for x, y in zip(x_positions, status_data[status]):
            plt.text(x, y + 1, f'{int(y)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize the plot
    plt.xlabel(id_vars, fontdict={"size": 15})
    plt.ylabel("Count", fontdict={"size": 15})
    plt.title(f"Patient Progression Status by {id_vars}", fontdict={"size": 20})
    plt.legend(title='Status')

    plt.xticks(x_positions, variables, fontdict={"size": 15})
    plt.xlim(x_start, x_end)  # Set x-axis limits to match shading

    plt.tight_layout()
    file_name = os.path.join(output_dir, f"{endpoint}_modified_patient_progression_{id_vars.lower()}.png")
    plt.savefig(file_name, dpi=300)
    plt.close()

def calculate_progression(group, progression_threshold, regression_threshold, volume_change_threshold, time_period_mapping):
        """
        Calculate the age at first progression and time to progression for each patient.
        """
        baseline_volume = group.iloc[0]["Baseline Volume"]
        progression_threshold = baseline_volume * float(progression_threshold)
        regression_threshold = baseline_volume * float(regression_threshold)
        volume_change = baseline_volume * float(volume_change_threshold)

        progression_mask = group["Volume"] >= progression_threshold
        regression_mask = group["Volume"] <= regression_threshold
        volume_change_mask = group["Volume"] >= volume_change

        inverse_mapping = {v: k for k, v in time_period_mapping.items()}

        if progression_mask.any():
            first_progression_index = progression_mask.idxmax()
            age_at_first_progression = group.loc[first_progression_index, "Age"]
            age_at_first_diagnosis = group.iloc[0]["Age at First Diagnosis"]
            time_to_progression = age_at_first_progression - age_at_first_diagnosis
            
            # calculate time period since diagnosis at progression
            time_period_since_diagnosis_value  = group.loc[first_progression_index, "Time Period Since Diagnosis"]
            time_period_since_diagnosis_numeric = get_time_period_numeric(time_period_since_diagnosis_value, time_period_mapping)
            time_period_since_diagnosis_at_progression = inverse_mapping[time_period_since_diagnosis_numeric]
        
            # volume change calculations    
            if volume_change_mask.any():
                first_volume_change_index = volume_change_mask.idxmax()
                age_at_volume_change = group.loc[first_volume_change_index, "Age"]
                if (
                    pd.notnull(age_at_first_progression)
                    and age_at_first_progression > age_at_volume_change
                ):
                    time_gap = age_at_first_progression - age_at_volume_change
                else:
                    time_gap = 0
            else:
                time_gap = 0
        
        else:
            age_at_first_progression = np.nan
            age_at_volume_change = np.nan
            time_period_since_diagnosis_numeric = group["Time Period Since Diagnosis"].map(time_period_mapping)
            max_time_period_numeric = time_period_since_diagnosis_numeric.max()
            time_period_since_diagnosis_at_progression = inverse_mapping[max_time_period_numeric]
            time_to_progression = np.nan
            time_gap = np.nan

        if regression_mask.any():
            first_regression_index = regression_mask.idxmax()
            age_at_first_regression = group.loc[first_regression_index, "Age"]
            time_to_regression = age_at_first_progression - age_at_first_regression
        else:
            age_at_first_regression = np.nan
            time_to_regression = np.nan

        time_to_progression_years = time_to_progression / 365.25 if pd.notnull(time_to_progression) else np.nan
        time_to_regression_years = time_to_regression / 365.25 if pd.notnull(time_to_regression) else np.nan
        time_gap_years = time_gap / 365.25 if pd.notnull(time_gap) else np.nan
        
        return pd.Series(
            {
                "Age at First Progression": age_at_first_progression,
                "Age at First Regression": age_at_first_regression,
                "Age at Volume Change": age_at_volume_change,
                "Time to Progression": time_to_progression,
                "Time to Regression": time_to_regression,
                "Time to Progression Years": time_to_progression_years,
                "Time to Regression Years": time_to_regression_years,
                "Time Gap": time_gap,
                "Time Gap Years": time_gap_years,
                "Time Since Diagnosis": time_period_since_diagnosis_at_progression, 
            }
        )

def get_time_period_numeric(time_period_value, category_mapping):
       if isinstance(time_period_value, str):
           return category_mapping.get(time_period_value, np.nan)
       elif isinstance(time_period_value, (int, float)):
           return time_period_value
       elif isinstance(time_period_value, pd.Series):
           return time_period_value.map(category_mapping).astype(float)
       else:
           return np.nan
       
##################################
# TUMOR STABILITY CLASSIFICATION #
##################################
def normalize_index(index):
    return (index - index.min()) / (index.max() - index.min())

def calculate_percentage_volume_change(data):
    first_volume = data.groupby('Patient_ID')['Volume'].first()
    last_volume = data.groupby('Patient_ID')['Volume'].last()
    percentage_change = ((last_volume - first_volume) / first_volume) * 100
    return percentage_change.to_dict()

def calculate_intra_tumor_variability_index(data):
    cv = data.groupby('Patient_ID')['Volume'].std() / data.groupby('Patient_ID')['Volume'].mean()
    return cv
    #return np.log1p(cv) # log-scale for visualization
    
def calculate_intra_tumor_growth_index(data):
    data['Volume Change Rate'] = data.groupby('Patient_ID')['Volume'].pct_change()
    avg_growth_rate = data.groupby('Patient_ID')['Volume Change Rate'].mean()
    return avg_growth_rate
    #return np.log1p(avg_growth_rate)

def calculate_inter_tumor_variability_index(data):
    cv_per_time = data.groupby('Time since First Scan')['Volume'].std() / data.groupby('Time since First Scan')['Volume'].mean()
    return cv_per_time # cv = coefficient of variation, having different values for each time point

def calculate_inter_tumor_growth_index(data):
    avg_growth_rate_per_time = data.groupby('Time since First Scan')['Volume Change Rate'].mean()
    return avg_growth_rate_per_time

def calculate_stability_index(data, intra_var_weight=0.8, intra_growth_weight=0.2, inter_var_weight=0, inter_growth_weight=0):
    print(data.columns)
    percentage_volume_change = calculate_percentage_volume_change(data)
    intra_var_index = calculate_intra_tumor_variability_index(data)
    intra_growth_index = calculate_intra_tumor_growth_index(data)
    inter_var_index = calculate_inter_tumor_variability_index(data)
    inter_growth_index = calculate_inter_tumor_growth_index(data)
    
    # Volume percentage change
    data['Volume Percentage Change'] = data['Patient_ID'].map(percentage_volume_change)
    data['Volume Percentage Change'] = data['Volume Percentage Change'].clip(lower=-150, upper=300)
    data['Volume Percentage Change'] = normalize_index(data['Volume Percentage Change'])

    # Intra-tumor indices
    data['Intra-Tumor Variability Index'] = data['Patient_ID'].map(intra_var_index)
    data['Intra-Tumor Variability Index'] = normalize_index(data['Intra-Tumor Variability Index'])
    data['Intra-Tumor Growth Index'] = data['Patient_ID'].map(intra_growth_index)
    data['Intra-Tumor Growth Index'] = normalize_index(data['Intra-Tumor Growth Index'])
    
    # Inter-tumor indices
    data['Inter-Tumor Variability Index'] = inter_var_index
    data['Inter-Tumor Growth Index'] = inter_growth_index
    print(data['Inter-Tumor Variability Index'].describe())
    print(data['Inter-Tumor Growth Index'].describe())
    data['Inter-Tumor Variability Index'] = normalize_index(data['Inter-Tumor Variability Index'])
    data['Inter-Tumor Growth Index'] = normalize_index(data['Inter-Tumor Growth Index'])
    print(data['Inter-Tumor Variability Index'].describe())
    print(data['Inter-Tumor Growth Index'].describe())
    
    data['Stability Index'] = (
        intra_var_weight * data['Intra-Tumor Variability Index'] +
        intra_growth_weight * data['Intra-Tumor Growth Index'] +
        inter_var_weight * data['Inter-Tumor Variability Index'] + 
        inter_growth_weight * data['Inter-Tumor Growth Index']
    )
    
    data['Predicted Classification'] = pd.cut(data['Stability Index'], bins=[-float('inf'), 0.1, 0.4, float('inf')], labels=['Stable', 'Progressor', 'Regressor'])
    
    return data

def visualize_stability_index(data, output_dir):
    classifications = data['Patient Classification Volumetric'].unique()
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))
    color_mapping = {"Regressor": palette[0], "Stable": palette[2], "Progressor": palette[1]}
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(x='Stability Index', hue='Patient Classification Volumetric', data=data, kde=True, alpha=0.5, palette=color_mapping)
    plt.title('Distribution of Stability Index')
    
    plt.subplot(2, 2, 2)
    for classification in classifications:
        subset = data[data['Patient Classification Volumetric'] == classification]
        first_patient_plotted = False
        for patient_id in subset['Patient_ID'].unique():
            patient_data = subset[subset['Patient_ID'] == patient_id]
            plt.plot(patient_data['Time since First Scan'], patient_data['Normalized Volume'], alpha=0.7, color=color_mapping[classification], label=classification if not first_patient_plotted else "" ) 
            first_patient_plotted = True
    plt.title('Tumor Volume over Time')
    plt.xlabel('Time since First Scan')
    plt.ylabel('Volume')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Patient Classification Volumetric', y='Stability Index', data=data, palette=color_mapping)
    plt.title('Stability Index Distribution by Patient Classification Volumetric')
    
    plt.subplot(2, 2, 4)
    sns.heatmap(pd.crosstab(data['Patient Classification Volumetric'], data['Predicted Classification'], normalize='index'), annot=True, cmap='YlGnBu')
    plt.title('Patient Classification Volumetric vs. Predicted Classification')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'stability_index_visualization.png')
    plt.savefig(output_file, dpi=300)

def visualize_individual_indexes(data, output_dir):
    classifications = data['Patient Classification Volumetric'].unique()
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))
    color_mapping = {"Regressor": palette[0], "Stable": palette[2], "Progressor": palette[1]}
    plt.figure(figsize=(18, 14))
    
    plt.subplot(6, 2, 1)
    sns.histplot(x='Intra-Tumor Variability Index', hue='Patient Classification Volumetric', data=data, kde=True, alpha=0.5, palette=color_mapping)
    plt.title('Distribution of Intra-Tumor Variability Index')
    
    plt.subplot(6, 2, 2)
    sns.histplot(x='Intra-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, kde=True, alpha=0.5, palette=color_mapping)
    plt.title('Distribution of Intra-Tumor Growth Index')
    
    plt.subplot(6, 2, 3)
    sns.histplot(x='Inter-Tumor Variability Index', hue='Patient Classification Volumetric', data=data, kde=True, alpha=0.5, palette=color_mapping)
    plt.title('Distribution of Inter-Tumor Variability Index')
    
    plt.subplot(6, 2, 4)
    sns.histplot(x='Inter-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, kde=True, alpha=0.5, palette=color_mapping)
    plt.title('Distribution of Inter-Tumor Growth Index')
    
    plt.subplot(6, 2, 5)
    sns.scatterplot(x='Intra-Tumor Variability Index', y='Intra-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Intra-Tumor Variability vs. Growth')
    
    plt.subplot(6, 2, 6)
    sns.scatterplot(x='Intra-Tumor Variability Index', y='Inter-Tumor Variability Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Intra-Tumor Variability vs. Inter-Tumor Variability')
    
    plt.subplot(6, 2, 7)
    sns.scatterplot(x='Intra-Tumor Growth Index', y='Inter-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Intra-Tumor Growth vs. Inter-Tumor Growth')
    
    plt.subplot(6, 2, 8)
    sns.scatterplot(x='Inter-Tumor Variability Index', y='Inter-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Inter-Tumor Variability vs. Growth')
    
    plt.subplot(6, 2, 9)
    sns.scatterplot(x='Stability Index', y='Intra-Tumor Variability Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Intra-Tumor Variability vs. Stability Index')
    
    plt.subplot(6, 2, 10)
    sns.scatterplot(x='Stability Index', y='Intra-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Intra-Tumor Growth vs. Stability Index')
    
    plt.subplot(6, 2, 11)
    sns.scatterplot(x='Stability Index', y='Inter-Tumor Variability Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Inter-Tumor Variability CV vs. Stability Index')
    
    plt.subplot(6, 2, 12)
    sns.scatterplot(x='Stability Index', y='Inter-Tumor Growth Index', hue='Patient Classification Volumetric', data=data, palette=color_mapping)
    plt.title('Inter-Tumor Growth CV vs. Stability Index')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'stability_individual_indexes.png')
    plt.savefig(output_file, dpi=300)
    
def visualize_ind_indexes_distrib(data, output_dir):
    classifications = data['Patient Classification Volumetric'].unique()
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))
    color_mapping = {"Regressor": palette[0], "Stable": palette[2], "Progressor": palette[1]}
    
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(x='Patient Classification Volumetric', y='Intra-Tumor Variability Index', data=data, ax=axs[0, 0], palette=color_mapping)
    sns.boxplot(x='Patient Classification Volumetric', y='Intra-Tumor Growth Index', data=data, ax=axs[0, 1], palette=color_mapping)
    sns.boxplot(x='Patient Classification Volumetric', y='Inter-Tumor Variability Index', data=data, ax=axs[1, 0], palette=color_mapping)
    sns.boxplot(x='Patient Classification Volumetric', y='Inter-Tumor Growth Index', data=data, ax=axs[1, 1], palette=color_mapping)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ind_stability_indexes_distrib.png')
    plt.savefig(output_file, dpi=300)
    
def visualize_volume_change(data, output_dir):
    classifications = data['Patient Classification Volumetric'].unique()
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))
    color_mapping = {"Regressor": palette[0], "Stable": palette[2], "Progressor": palette[1]}

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for classification in classifications:
        subset = data[data['Patient Classification Volumetric'] == classification]
        sns.kdeplot(subset['Volume Percentage Change'], label=classification, color=color_mapping[classification], shade=True, alpha=0.7)
    
    plt.title('Distribution of Volume Percentage Change by Patient Classification Volumetric')
    plt.xlabel('Volume Percentage Change')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Patient Classification Volumetric', y='Volume Percentage Change', data=data, palette=color_mapping)
    plt.title('Volume Percentage Change Distribution by Patient Classification Volumetric')
    plt.xlabel('Patient Classification Volumetric')
    plt.ylabel('Volume Percentage Change')
    plt.legend()

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'volume_change_distribution.png')
    plt.savefig(output_file, dpi=300)
    
def roc_curve_and_auc(data, output_dir):
    classifications = data['Patient Classification Volumetric'].unique()
    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))
    color_mapping = {"Regressor": palette[0], "Stable": palette[2], "Progressor": palette[1]}
    data = data.dropna(subset=['Patient Classification Volumetric', 'Stability Index'])
    
    true_labels = data['Patient Classification Volumetric']
    classes = data['Patient Classification Volumetric'].unique()
    predicted_probabilities = data['Stability Index']
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    
    for cls in classes:
        binary_labels = (true_labels == cls).astype(int)
        fpr[cls], tpr[cls], _ = roc_curve(binary_labels, predicted_probabilities)
        roc_auc[cls] = auc(fpr[cls], tpr[cls])
    
    plt.figure(figsize=(10, 6))
    for cls in classes:
        plt.plot(fpr[cls], tpr[cls], color=color_mapping[cls], label=f'{cls} (AUC={roc_auc[cls]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_file, dpi=300)
    
def grid_search_weights(data):

    print(data[['Intra-Tumor Variability Index', 'Intra-Tumor Growth Index', 'Inter-Tumor Variability Index', 'Inter-Tumor Growth Index']].describe())
    param_grid = [
        {'intra_var_weight': 0.3, 'intra_growth_weight': 0.3, 'inter_var_weight': 0.2, 'inter_growth_weight': 0.2},
        {'intra_var_weight': 0.4, 'intra_growth_weight': 0.4, 'inter_var_weight': 0.1, 'inter_growth_weight': 0.1},
        {'intra_var_weight': 0.45, 'intra_growth_weight': 0.45, 'inter_var_weight': 0.05, 'inter_growth_weight': 0.05},
        {'intra_var_weight': 0.59, 'intra_growth_weight': 0.29, 'inter_var_weight': 0.01, 'inter_growth_weight': 0.01},
        {'intra_var_weight': 0.7, 'intra_growth_weight': 0.3, 'inter_var_weight': 0, 'inter_growth_weight': 0},
        {'intra_var_weight': 0.3, 'intra_growth_weight': 0.7, 'inter_var_weight': 0, 'inter_growth_weight': 0},
        {'intra_var_weight': 0.59, 'intra_growth_weight': 0.29, 'inter_var_weight': 0.01, 'inter_growth_weight': 0.01},
        {'intra_var_weight': 0.45, 'intra_growth_weight': 0.45, 'inter_var_weight': 0.05, 'inter_growth_weight': 0.05},
        # Add more parameter combinations to the grid
        ]

    
    X_train, X_val, y_train, y_val = train_test_split(data, data['Patient Classification Volumetric'], test_size=0.2, random_state=42)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    print(f"y_train unique values and counts: {np.unique(y_train, return_counts=True)}")
    print(f"y_val unique values and counts: {np.unique(y_val, return_counts=True)}")
    best_params = None
    best_auc = 0
    for params in param_grid:
        X_train_stability = calculate_stability_index(X_train, **params)
        X_val_stability = calculate_stability_index(X_val, **params)
        print(f"X_train_stability shape: {X_train_stability.shape}")
        print(f"X_val_stability shape: {X_val_stability.shape}")
        X_val_stability = X_val_stability.dropna(subset=['Stability Index'])  # Remove rows with NaN values
        
        if X_val_stability.empty:
            print(f"Skipping parameter set: {params} due to empty validation data after removing NaN values.")
            continue
        print(f"X_val_stability shape after removing NaN values: {X_val_stability.shape}")
        print(f"y_val unique values and counts after removing NaN values: {np.unique(y_val[X_val_stability.index], return_counts=True)}")
        
        y_val = y_val[X_val_stability.index]  # Update the corresponding labels
        print(f"y_val values: {y_val.values}")
        print(f"X_val_stability['Stability Index'] values: {X_val_stability['Stability Index'].values}")
        
        auc_value = roc_auc_score(y_val, X_val_stability['Stability Index'], average='macro', multi_class='ovo')
        
        if auc_value > best_auc:
            best_auc = auc_value
            best_params = params
    
    print(f"\t\tBest parameters: {best_params}")
    print(f"\t\tBest ROC AUC: {best_auc}")
