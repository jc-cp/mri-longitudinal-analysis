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
import matplotlib.lines as lines
import seaborn as sns
from cfg.utils import helper_functions_cfg

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


def save_for_deep_learning(df: pd.DataFrame, output_dir, prefix):
    """
    Save data for deep learning in csv format.
    """
    if df is not None:
        filename = f"{prefix}_dl_features.csv"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"\tData saved for deep learning in {filename}.csv")
    else:
        print("No data to save.")


def process_race_ethnicity(race):
    """
    Removing 'Non-Hispanic' from the string for consistency in data.
    """
    if "Non-Hispanic" in race:
        race = race.replace("Non-Hispanic ", "")
    return race


def categorize_age_group(data, debug=False):
    """
    Categorize patients according to an age group for more thorough analysis.
    """
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
    data["Period"] = data["Date"].dt.to_period("M")

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


def classify_patient(
    data,
    patient_id,
    column_name,
    progression_threshold,
    stability_threshold,
    high_risk_threshold,
    angle=False,
):
    """
    Classify a patient based on the slope of actual and predicted growth rates.
    """
    data = data.sort_values(by="Time since First Scan")
    patient_data = data[data["Patient_ID"] == patient_id]

    if len(patient_data) < 2:
        return None

    if angle:
        _, actual_angle = calculate_slope_and_angle(data, patient_id, column_name)

        if actual_angle is None:
            return f"Failed to calculate slope for patient {patient_id}"

        if actual_angle > progression_threshold:
            progression_type = "Progressor"
            risk_level = "High-risk" if actual_angle > high_risk_threshold else "Low-risk"
            return f"{progression_type}, {risk_level}"
        elif abs(actual_angle) < stability_threshold:
            return "Stable"
        elif actual_angle < -progression_threshold:
            return "Regressor"
        else:
            return "Erratic"
    else:
        first_value = patient_data[column_name].iloc[0]
        last_value = patient_data[column_name].iloc[-1]
        progression_threshold = 25
        # Calculate percentage change
        if first_value == 0:
            return "Erratic"  # Avoid division by zero

        percent_change = ((last_value - first_value) / first_value) * 100

        if percent_change >= progression_threshold:
            return "Progressor"
        elif percent_change <= -progression_threshold:
            return "Regressor"
        else:
            return "Stable"


def calculate_percentage_change(data, patient_id, column_name):
    """
    Calculate the percentage change in a volume for a patient
    considering the first and last value.
    """
    patient_data = data[data["Patient_ID"] == patient_id].sort_values(by="Time since First Scan")
    if len(patient_data) < 2:
        return None  # Not enough data

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

        patient_classification = first_row["Patient Classification"]
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


def plot_trend_trajectories(data, output_filename, column_name, unit=None):
    """
    Plot the growth trajectories of patients with classifications.

    Parameters:
    - data: DataFrame containing patient growth data and classifications.
    - output_filename: Name of the file to save the plot.
    """
    plt.figure(figsize=(15, 8))

    # Unique classifications & palette
    data = data[data["Time since First Scan"] <= 4000]
    classifications = data["Classification"].unique()

    palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(classifications))

    for classification, color in zip(classifications, palette):
        class_data = data[data["Classification"] == classification]
        first_patient_plotted = False

        # Plot individual trajectories
        for patient_id in class_data["Patient_ID"].unique():
            patient_data = class_data[class_data["Patient_ID"] == patient_id]

            if classification is not None:
                plt.plot(
                    patient_data["Time since First Scan"],
                    patient_data[column_name],
                    color=color,
                    alpha=0.5,
                    linewidth=1,
                    label=classification if not first_patient_plotted else "",
                )
            first_patient_plotted = True

        if classification is not None:
            # Plot median trajectory for each classification
            median_data = (
                class_data.groupby(
                    pd.cut(
                        class_data["Time since First Scan"],
                        pd.interval_range(
                            start=0, end=class_data["Time since First Scan"].max(), freq=273
                        ),
                    )
                )[column_name]
                .median()
                .reset_index()
            )
            sns.lineplot(
                x=median_data["Time since First Scan"].apply(lambda x: x.mid),
                y=column_name,
                data=median_data,
                color=color,
                linestyle="--",
                label=f"{classification} Median",
                linewidth=2.5,
            )

    num_patients = data["Patient_ID"].nunique()

    plt.xlabel("Days Since First Scan")
    plt.ylabel(f"Tumor {column_name} [{unit}]")
    plt.title(f"Patient Trend Trajectories (N={num_patients})")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()


def plot_individual_trajectories(name, plot_data, column, category_column=None, unit=None):
    """
    Plot the individual trajectories for a sample of patients.

    Parameters:
    - name (str): The filename to save the plot image.

    This method samples a n number of unique patient IDs from the pre-treatment data,
    plots their variable trajectories, and saves the plot to the specified filename.
    """
    plt.figure(figsize=(10, 6))

    # Cutoff the data at 4000 days
    plot_data = plot_data[plot_data["Time since First Scan"] <= 4000]
    num_patients = plot_data["Patient_ID"].nunique()
    # Get the median every 3 months
    median_data = (
        plot_data.groupby(
            pd.cut(
                plot_data["Time since First Scan"],
                pd.interval_range(start=0, end=plot_data["Time since First Scan"].max(), freq=91),
            )
        )[column]
        .median()
        .reset_index()
    )

    if category_column:
        categories = plot_data[category_column].unique()
        patient_palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(categories))
        median_palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(categories))
        legend_handles = []

        for (category, patient_color), median_color in zip(
            zip(categories, patient_palette), median_palette
        ):
            category_data = plot_data[plot_data[category_column] == category]
            median_data_category = (
                category_data.groupby(
                    pd.cut(
                        category_data["Time since First Scan"],
                        pd.interval_range(
                            start=0, end=category_data["Time since First Scan"].max(), freq=91
                        ),
                    )
                )[column]
                .median()
                .reset_index()
            )
            legend_handles.append(
                lines.Line2D([], [], color=patient_color, label=f"{category_column} {category}")
            )
            for patient_id in category_data["Patient_ID"].unique():
                patient_data = category_data[category_data["Patient_ID"] == patient_id]
                plt.plot(
                    patient_data["Time since First Scan"],
                    patient_data[column],
                    color=patient_color,
                    alpha=0.5,
                    linewidth=1,
                )
            sns.lineplot(
                x=median_data_category["Time since First Scan"].apply(lambda x: x.mid),
                y=column,
                data=median_data_category,
                color=median_color,
                linestyle="--",
                label=f"{category_column} {category} Median Trajectory",
            )

        sns.lineplot(
            x=median_data["Time since First Scan"].apply(lambda x: x.mid),
            y=column,
            data=median_data,
            color="red",
            linestyle="--",
            label="Cohort Median Trajectory",
        )
        # Retrieve the handles and labels from the current plot
        handles, _ = plt.gca().get_legend_handles_labels()
        # Combine custom category handles with the median trajectory handles
        combined_handles = legend_handles + handles[-(len(categories) + 1) :]

        plt.title(f"Individual Tumor {column} Trajectories by {category_column} (N={num_patients})")
        plt.legend(handles=combined_handles)

    else:
        # Plot each patient's data
        for patient_id in plot_data["Patient_ID"].unique():
            patient_data = plot_data[plot_data["Patient_ID"] == patient_id]
            sns.set_palette(helper_functions_cfg.NORD_PALETTE)
            plt.plot(
                patient_data["Time since First Scan"],
                patient_data[column],
                alpha=0.5,
                linewidth=1,
            )

        sns.lineplot(
            x=median_data["Time since First Scan"].apply(lambda x: x.mid),
            y=column,
            data=median_data,
            color="blue",
            linestyle="--",
            label="Median Trajectory",
        )
        plt.title(f"Individual Tumor {column} Trajectories (N={num_patients})")
        plt.legend()

    plt.xlabel("Days Since First Scan")
    plt.ylabel(f"Tumor {column} [{unit}]")
    plt.savefig(name)
    plt.close()
    if category_column:
        print(f"\t\tSaved tumor {column} trajectories plot by category: {category_column}.")
    else:
        print(f"\t\tSaved tumor {column} trajectories plot for all patients.")
