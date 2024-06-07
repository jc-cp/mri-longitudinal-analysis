"""
Evaluation script for ARIMA model forecasts.
"""
import ast
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
csv_path_bch = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_plots_bch/BCH_forecasts.csv"
csv_path_cbtn = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_plots_cbtn/CBTN_forecasts.csv"
output_dir = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_evaluation"
os.makedirs(output_dir, exist_ok=True)
cohort = "JOINT" # "BCH" or "CBTN or "JOINT"

if cohort == "JOINT":
    bch_df = pd.read_csv(csv_path_bch)
    cbtn_df = pd.read_csv(csv_path_cbtn)

    # Add a cohort column to each dataframe
    bch_df['Cohort'] = 'BCH'
    cbtn_df['Cohort'] = 'CBTN'

    # Concatenate the dataframes vertically
    cohort_df = pd.concat([bch_df, cbtn_df], ignore_index=True)
else:
    csv_path = csv_path_cbtn
    cohort = "BCH" if csv_path == csv_path_bch else "CBTN"
    cohort_df = pd.read_csv(csv_path)
    print(cohort_df.head())

# Get the lists
cohort_df["Forecast"] = cohort_df["Forecast"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
cohort_df["Rolling Predictions"] = cohort_df["Rolling Predictions"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
cohort_df["Validation Data"] = cohort_df["Validation Data"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
cohort_df["Validation Error"] = cohort_df.apply(
    lambda row: [
        f - v for f, v in zip(row["Rolling Predictions"], row["Validation Data"])
    ],
    axis=1,
)

# Flatten the lists
forecast_flat = [item for sublist in cohort_df["Forecast"] for item in sublist]
rolling_flat = [
    item for sublist in cohort_df["Rolling Predictions"] for item in sublist
]
validation_flat = [item for sublist in cohort_df["Validation Data"] for item in sublist]
error_flat = [item for sublist in cohort_df["Validation Error"] for item in sublist]
error_mean = np.mean(error_flat)
error_std = np.std(error_flat)
upper_bound = error_mean + 2.5 * error_std
lower_bound = error_mean - 2.5 * error_std
error_flat_filtered = [x for x in error_flat if lower_bound <= x <= upper_bound]


@staticmethod
def roll_vs_val(df, directory):
    """
    Rollling predictions vs validation data plot.
    """
    plt.figure(figsize=(10, 8))
    for i, (_, rw) in enumerate(df.iterrows()):
        min_length = min(len(rw["Validation Data"]), len(rw["Rolling Predictions"]))
        validation_data = rw["Validation Data"][:min_length]
        rolling_predictions = rw["Rolling Predictions"][:min_length]

        plt.scatter(
            validation_data, rolling_predictions, alpha=0.5, label=f"Patient {i+1}"
        )

    plt.xlabel("Validation Data", fontsize=12)
    plt.ylabel("Rolling Predictions", fontsize=12)
    plt.title("Rolling Predictions vs Validation Data", fontsize=14)
    plt.tight_layout(pad=3.0)
    file_path_fore_val = os.path.join(
        directory, f"rolling_vs_validation_{cohort.lower()}.png"
    )
    plt.savefig(file_path_fore_val, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, (_, rw) in enumerate(cohort_df.iterrows()):
        min_length = min(len(rw["Validation Data"]), len(rw["Rolling Predictions"]))
        validation_data = rw["Validation Data"][:min_length]
        forecast_data = rw["Rolling Predictions"][:min_length]
        residuals = np.array(forecast_data) - np.array(validation_data)
        plt.scatter(validation_data, residuals, alpha=0.5, label=f"Patient {i+1}")
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

    plt.xlabel("Validation Data", fontsize=12)
    plt.ylabel("Residuals (Rolling Predictions - Validation Data)", fontsize=12)
    plt.title("Residual Plot (Rolling Predictions - Validation Data)", fontsize=14)
    plt.tight_layout()
    file_path_resid = os.path.join(directory, f"residual_plot_{cohort.lower()}.png")
    plt.savefig(file_path_resid, dpi=300)
    plt.close()


@staticmethod
def error_distrib(errors, directory):
    """
    Error distribution plot.
    """
    # Plot distribution of forecast errors
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=20, edgecolor="black", alpha=0.5, stat="density")
    sns.kdeplot(errors, color="orange")
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)

    # Add mean, median, and standard deviation to the plot
    plt.axvline(
        mean_error,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_error:.4f}",
    )
    plt.axvline(
        median_error,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_error:.4f}",
    )
    plt.axvline(
        mean_error - std_error,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Std: {std_error:.4f}",
    )
    plt.axvline(mean_error + std_error, color="gray", linestyle="--", linewidth=1)

    plt.xlabel("Validation Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Validation Errors")
    plt.legend()
    plt.tight_layout()
    file_path_error = os.path.join(
        directory, f"validation_error_distribution_{cohort.lower()}.png"
    )
    plt.savefig(file_path_error, dpi=300)
    plt.close()


@staticmethod
def metrics_plot(df, directory):
    """
    Box plot of performance metrics across patients.
    """
    # Box plot of performance metrics across patients
    metrics1 = ["AIC", "BIC", "HQIC"]
    metrics2 = ["MAE", "MSE", "RMSE"]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for i, metric in enumerate(metrics1):
        data = df[metric]
        filtered_data = data[
            (
                data
                > data.quantile(0.25)
                - 1.5 * (data.quantile(0.75) - data.quantile(0.25))
            )
            & (
                data
                < data.quantile(0.75)
                + 1.5 * (data.quantile(0.75) - data.quantile(0.25))
            )
        ]
        ax1.boxplot(
            [filtered_data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=f"C{i}", alpha=0.7),
            whiskerprops=dict(color=f"C{i}"),
            capprops=dict(color=f"C{i}"),
            medianprops=dict(color="black"),
        )
    ax1.set_xticks(range(len(metrics1)))
    ax1.set_xticklabels(metrics1)
    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Value")
    ax1.set_title("Distribution of AIC, BIC, HQIC")

    for i, metric in enumerate(metrics2):
        data = cohort_df[metric]
        filtered_data = data[
            (
                data
                > data.quantile(0.25)
                - 1.5 * (data.quantile(0.75) - data.quantile(0.25))
            )
            & (
                data
                < data.quantile(0.75)
                + 1.5 * (data.quantile(0.75) - data.quantile(0.25))
            )
        ]
        ax2.boxplot(
            [filtered_data],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=f"C{i}", alpha=0.7),
            whiskerprops=dict(color=f"C{i}"),
            capprops=dict(color=f"C{i}"),
            medianprops=dict(color="black"),
        )
    ax2.set_xticks(range(len(metrics2)))
    ax2.set_xticklabels(metrics2)
    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Value")
    ax2.set_title("Distribution of MAE, MSE, RMSE")

    plt.tight_layout()
    file_path_metrics = os.path.join(
        directory, f"performance_metrics_boxplot_{cohort.lower()}.png"
    )
    plt.savefig(file_path_metrics, dpi=300)
    plt.close()


@staticmethod
def trend_plot(df, directory):
    """
    Trend analysis of forecast values and validation data.
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for i, (_, row) in enumerate(df.iterrows()):
        forecast_trend = pd.Series(row["Forecast"]).pct_change().tolist()
        validation_trend = pd.Series(row["Validation Data"]).pct_change().tolist()

        ax1.plot(forecast_trend, label=f"Patient {i+1}")
        ax2.plot(validation_trend, label=f"Patient {i+1}")
        # Add summary statistics
        rolling_trend_mean = np.mean(
            [
                pd.Series(row["Rolling Predictions"]).pct_change().mean()
                for _, row in cohort_df.iterrows()
            ]
        )
        rolling_trend_std = np.mean(
            [
                pd.Series(row["Rolling Predictions"]).pct_change().std()
                for _, row in cohort_df.iterrows()
            ]
        )
        validation_trend_mean = np.mean(
            [
                pd.Series(row["Validation Data"]).pct_change().mean()
                for _, row in cohort_df.iterrows()
            ]
        )
        validation_trend_std = np.mean(
            [
                pd.Series(row["Validation Data"]).pct_change().std()
                for _, row in cohort_df.iterrows()
            ]
        )
        ax1.text(
            0.95,
            0.95,
            f"Mean = {rolling_trend_mean:.4f}\nStd = {rolling_trend_std:.4f}",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax2.text(
            0.95,
            0.95,
            f"Mean = {validation_trend_mean:.4f}\nStd = {validation_trend_std:.4f}",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Forecast Trend")
    ax1.set_title("Trend Analysis")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Validation Trend")
    ax2.set_title("Validation Trend Analysis")

    plt.tight_layout()
    file_path_trend = os.path.join(
        directory, f"forecast_validation_trend_{cohort.lower()}.png"
    )
    plt.savefig(file_path_trend, dpi=300)
    plt.close()


# Plotting
roll_vs_val(cohort_df, output_dir)
error_distrib(error_flat_filtered, output_dir)
metrics_plot(cohort_df, output_dir)
trend_plot(cohort_df, output_dir)
print("Evaluation plots saved successfully!")
