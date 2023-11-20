"""Script containing some additonal functions used thorughout the other main scripts."""
from math import isfinite
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, zscore
from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, f_oneway, pointbiserialr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


#####################################
# SMOOTHIN and FILTERING OPERATIONS #
#####################################


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
    plt.scatter(original_p_values, corrected_p_values, alpha=0.7)
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
    Check the balance of covariates in the matched dataset using Standardized Mean Differences (SMD).

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
    group1 = data[data[x_val] == data[x_val].unique()[0]][y_val]
    group2 = data[data[x_val] == data[x_val].unique()[1]][y_val]
    t_stat, p_val = ttest_ind(group1.dropna(), group2.dropna())
    return t_stat, p_val


def f_one(data, x_val, y_val):
    groups = [group[y_val].dropna() for name, group in data.groupby(x_val)]
    f_stat, p_val = f_oneway(*groups)
    return f_stat, p_val


def point_bi_serial(data, binary_var, continuous_var):
    binary_data = data[binary_var].cat.codes
    continuous_data = data[continuous_var]
    coef, p_val = pointbiserialr(binary_data, continuous_data)
    return coef, p_val


def visualize_time_to_treatment_effect(filtered_data, prefix, path):
    # Plot the observed vs predicted values
    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        filtered_data["Time_to_Treatment"], filtered_data["Growth[%]"], label="Observed Growth"
    )
    ax.plot(
        filtered_data["Time_to_Treatment"],
        filtered_data["Predicted_Growth"],
        color="red",
        label="Predicted Growth",
    )
    ax.set_title("Effect of Time to Treatment on Tumor Growth")
    ax.set_xlabel("Time to Treatment (days)")
    ax.set_ylabel("Tumor Growth (%)")
    ax.legend()
    plt.tight_layout()
    filename = os.path.join(path, f"{prefix}_time_to_treatment_effect.png")
    plt.savefig(filename)


#######################################
# DATA HANDLING and SIMPLE OPERATIONS #
#######################################


def prefix_zeros_to_six_digit_ids(patient_id):
    str_id = str(patient_id)
    if len(str_id) == 6:
        # print(f"Found a 6-digit ID: {str_id}. Prefixing a '0'.")
        patient_id = "0" + str_id

    else:
        patient_id = str_id
    return patient_id


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


def calculate_stats(row, col_name):
    values = row[col_name]
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    return pd.Series(
        [mean_val, median_val, std_val],
        index=[f"{col_name}_mean", f"{col_name}_median", f"{col_name}_std"],
    )


def zero_fill(series, width):
    return series.astype(str).str.zfill(width)


def save_for_deep_learning(df: pd.DataFrame, output_dir, prefix):
    if df is not None:
        filename = f"{prefix}_dl_features.csv"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Data saved for deep learning in {filename}.csv")
    else:
        print("No data to save.")


def calculate_brain_growth(data, younger_age, older_age):
    # Assuming data contains columns 'Age' and 'Volume'
    younger_volume = data[data["Age"] == younger_age]["Volume"].mean()
    older_volume = data[data["Age"] == older_age]["Volume"].mean()

    percentage_growth = ((older_volume - younger_volume) / younger_volume) * 100
    return percentage_growth


def process_race_ethnicity(race):
    # Removing 'Non-Hispanic' from the string
    if "Non-Hispanic" in race:
        race = race.replace("Non-Hispanic ", "")
    return race


def categorize_age_group(data, debug=False):
    if debug:
        print(data["Age"].max())
        print(data["Age"].min())
        print(data["Age"].mean())
        print(data["Age"].median())
        print(data["Age"].std())

    age = data["Age"]

    if age <= 2:
        return "Infant"
    elif age <= 5:
        return "Preschool"
    elif age <= 11:
        return "School Age"
    elif age <= 18:
        return "Adolescent"
    else:
        return "Young Adult"


def calculate_group_norms_and_stability(data, output_dir):
    # Calculate group-wise statistics for each age group
    group_norms = (
        data.groupby("Age_Group")
        .agg(
            {
                "Volume_RollStd": "median",  # Or mean, based on preference
                "Growth[%]_RollStd": "median",  # Or mean
            }
        )
        .reset_index()
    )

    data = data.merge(group_norms, on="Age_Group", suffixes=("", "_GroupNorm"))
    data["Volume_Stability_Score"] = data["Volume_RollStd"] / data["Volume_RollStd_GroupNorm"]
    data["Growth_Stability_Score"] = data["Growth[%]_RollStd"] / data["Growth[%]_RollStd_GroupNorm"]
    data["Period"] = data["Date"].dt.to_period("M")

    # Plotting Stability Scores
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        3,
        2,
    )

    # Heatmap
    # heatmap_data = (
    #     data.groupby(["Age_Group", "Period"])["Volume_Stability_Score"].median().unstack()
    # )
    ax0 = plt.subplot(gs[0, 0])
    heatmap_data = data.pivot_table(
        index="Age_Group", columns="Period", values="Volume_Stability_Score", aggfunc="median"
    )
    sns.heatmap(heatmap_data, cmap="coolwarm", ax=ax0)
    ax0.set_title("Median Volume Stability Score Across Age Groups Over Time")

    # Violin Plot
    ax1 = plt.subplot(gs[0, 1])
    sns.violinplot(x="Age_Group", y="Volume_Stability_Score", data=data, ax=ax1, bw=0.2)
    sns.swarmplot(
        x="Age_Group", y="Volume_Stability_Score", data=data, ax=ax1, color="k", alpha=0.6
    )
    ax1.set_title("Distribution of Volume Stability Scores Across Age Groups")

    # Scatter Plot with Trend Lines
    ax2 = plt.subplot(gs[1:, :])
    for age_group in data["Age_Group"].unique():
        group_data = data[data["Age_Group"] == age_group]
        group_data = group_data.sort_values("Date")
        moving_average = (
            group_data["Volume_Stability_Score"].rolling(window=3, min_periods=1).mean()
        )
        ax2.plot(group_data["Date"], moving_average, label=age_group)
        # sns.scatterplot(
        #     x="Date", y="Volume_Stability_Score", data=group_data, label=age_group, ax=ax2
        # )
        # sns.lineplot(x="Date", y="Volume_Stability_Score", data=group_data, ax=ax2)
        ax2.scatter(data["Date"], data["Volume_Stability_Score"], alpha=0.3)
    ax2.set_title("Scatter Plot of Volume Stability Scores Over Time by Age Group")
    ax2.legend(title="Age Group")

    plt.tight_layout()
    filename = os.path.join(output_dir, "stability_scores.png")
    plt.savefig(filename)
    plt.close()

    return data


def calculate_slope(data, patient_id, column_name):
    """
    Calculate the slope of the trend line for a patient's data.
    """
    patient_data = data[data["Patient_ID"] == patient_id]
    if len(patient_data) < 2:
        return None  # Not enough data to calculate a slope

    # Reshape for sklearn
    x = patient_data["Time_since_First_Scan"].values.reshape(-1, 1)
    y = patient_data[column_name].values

    # Fit linear model
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    return slope


def classify_patient(
    data,
    patient_id,
    column_name,
    early_progression_threshold,
    stability_threshold,
    high_risk_threshold,
):
    """
    Classify a patient based on the slope of actual and predicted growth rates.
    """
    patient_data = data[data["Patient_ID"] == patient_id]
    if len(patient_data) < 2:
        return f"Insufficient Data for patient {patient_id}"

    actual_slope = calculate_slope(data, patient_id, column_name)
    predicted_slope = calculate_slope(data, patient_id, "Predicted_" + column_name)

    if actual_slope is None or predicted_slope is None:
        return f"Failed to calculate slope for patient {patient_id}"

    if actual_slope > 0 and predicted_slope > 0:
        # Determine early vs late progression
        progression_time = patient_data["Time_since_First_Scan"].iloc[
            np.argmax(patient_data[column_name] > 0)
        ]
        if progression_time < early_progression_threshold:
            progression_type = "Early Progressor"
        else:
            progression_type = "Late Progressor"
        # Determine risk level based on the magnitude of the slope
        risk_level = (
            "High-risk" if max(actual_slope, predicted_slope) > high_risk_threshold else "Low-risk"
        )
        return f"{progression_type}, {risk_level}"

    if abs(actual_slope) < stability_threshold and abs(predicted_slope) < stability_threshold:
        return "Stable"
    if actual_slope < 0 and predicted_slope < 0:
        return "Regressor"
    return "Erratic"


def plot_trend_trajectories(data, output_filename, column_name):
    """
    Plot the growth trajectories of patients with classifications.

    Parameters:
    - data: DataFrame containing patient growth data and classifications.
    - output_filename: Name of the file to save the plot.
    """
    plt.figure(figsize=(15, 8))

    # Unique classifications
    classifications = data["Classification"].unique()

    # Define a color palette
    palette = sns.color_palette("hsv", len(classifications))

    for classification, color in zip(classifications, palette):
        # Filter data for each classification
        class_data = data[data["Classification"] == classification]

        # Plot actual growth
        sns.lineplot(
            x="Time_since_First_Scan",
            y=column_name,
            data=class_data,
            label=f"{classification} - Actual",
            color=color,
            linestyle="-",
            alpha=0.7,
        )

    plt.xlabel("Days Since First Scan")
    plt.ylabel(f"Tumor {column_name}")
    plt.title("Patient Trend Trajectories")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()


def plot_growth_predictions(data, filename, column_name):
    """
    Plot the actual versus predicted values over time.

    Parameters:
    - filename (str): The filename to save the plot image.

    This method plots the actual and predicted growth percentages from the
    pre-treatment data and saves the plot as an image.
    """
    sns.lineplot(
        x="Time_since_First_Scan",
        y=f"{column_name}",
        data=data,
        alpha=0.5,
        color="blue",
        label=f"Actual {column_name}",
    )
    sns.lineplot(
        x="Time_since_First_Scan",
        y=f"Predicted_{column_name}",
        data=data,
        color="red",
        label=f"Predicted {column_name}",
    )
    plt.xlabel("Days Since First Scan")
    plt.ylabel(f"Tumor {column_name}")
    plt.title(f"Actual vs Predicted {column_name} Over Time")
    plt.legend()
    plt.savefig(filename)
    plt.close()
