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

# Function to convert string representations of lists to actual lists
def convert_to_list(x):
    """Simple comversion."""
    if isinstance(x, str):
        try:
            # First, try to evaluate as a Python list
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If that fails, try to convert from numpy array string representation
            try:
                return np.fromstring(x.strip('[]'), sep=' ').tolist()
            except ExceptionGroup as e:
                print(e)
                return x
    return x


def remove_outliers(data, column):
    if data[column].dtype == 'object':
        # For list columns, calculate statistics on the mean of each list
        series = data[column].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
    else:
        series = data[column]
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    
    if data[column].dtype == 'object':
        return data[series.between(lower_bound, upper_bound)]
    else:
        return data[data[column].between(lower_bound, upper_bound)]
    
# Apply the conversion function to relevant columns
list_columns = ['ARIMA_Forecast', 'ARIMA+GARCH_Forecast', 'ARIMA_Rolling_Predictions', 
                'ARIMA+GARCH_Rolling_Predictions', 'Validation_Data']
for col in list_columns:
    cohort_df[col] = cohort_df[col].apply(convert_to_list)

# Calculate validation errors for both models
cohort_df['ARIMA_Validation_Error'] = cohort_df.apply(
    lambda row: [f - v for f, v in zip(row['ARIMA_Rolling_Predictions'], row['Validation_Data'])], axis=1
)
cohort_df['ARIMA+GARCH_Validation_Error'] = cohort_df.apply(
    lambda row: [f - v for f, v in zip(row['ARIMA+GARCH_Rolling_Predictions'], row['Validation_Data'])], axis=1
)

# List of columns to apply outlier removal
columns_to_filter = ['ARIMA_MAE', 'ARIMA_MSE', 'ARIMA_RMSE', 
                     'ARIMA+GARCH_MAE', 'ARIMA+GARCH_MSE', 'ARIMA+GARCH_RMSE',
                     'ARIMA_Rolling_Predictions', 'ARIMA+GARCH_Rolling_Predictions',
                     'ARIMA_Validation_Error', 'ARIMA+GARCH_Validation_Error']

# Apply outlier removal to each column
cohort_df_filtered = cohort_df.copy()
for col in columns_to_filter:
    if col in cohort_df_filtered.columns:
        cohort_df_filtered = remove_outliers(cohort_df_filtered, col)
        print(f"Outliers removed from {col}")
    else:
        print(f"Warning: Column {col} not found in the dataframe")

@staticmethod
def roll_vs_val(df, directory):
    """
    Rolling predictions vs validation data plot for both models.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for i, (_, rw) in enumerate(df.iterrows()):
        min_length = min(len(rw['Validation_Data']), len(rw['ARIMA_Rolling_Predictions']), 
                         len(rw['ARIMA+GARCH_Rolling_Predictions']))
        validation_data = rw['Validation_Data'][:min_length]
        arima_predictions = rw['ARIMA_Rolling_Predictions'][:min_length]
        arimagarch_predictions = rw['ARIMA+GARCH_Rolling_Predictions'][:min_length]

        ax1.scatter(validation_data, arima_predictions, alpha=0.5, label=f"Patient {i+1}")
        ax2.scatter(validation_data, arimagarch_predictions, alpha=0.5, label=f"Patient {i+1}")

    ax1.set_xlabel("Validation Data", fontsize=12)
    ax1.set_ylabel("ARIMA Rolling Predictions", fontsize=12)
    ax1.set_title("ARIMA: Rolling Predictions vs Validation Data", fontsize=14)
    
    ax2.set_xlabel("Validation Data", fontsize=12)
    ax2.set_ylabel("ARIMA+GARCH Rolling Predictions", fontsize=12)
    ax2.set_title("ARIMA+GARCH: Rolling Predictions vs Validation Data", fontsize=14)

    plt.tight_layout(pad=3.0)
    file_path = os.path.join(directory, f"rolling_vs_validation_comparison_{cohort.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    
@staticmethod
def error_distrib(df, directory):
    """
    Error distribution plot for both models.
    """
    arima_errors = [item for sublist in df['ARIMA_Validation_Error'] for item in sublist]
    arimagarch_errors = [item for sublist in df['ARIMA+GARCH_Validation_Error'] for item in sublist]
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(arima_errors, bins=25, kde=True, ax=ax1, color="#8FBCBB")
    ax1.axvline(np.mean(arima_errors), color='r', linestyle='--', label=f'Mean: {np.mean(arima_errors):.2f}')
    ax1.axvline(np.median(arima_errors), color='g', linestyle='--', label=f'Median: {np.median(arima_errors):.2f}')
    ax1.set_title("ARIMA: Distribution of Validation Errors")
    ax1.legend()

    sns.histplot(arimagarch_errors, bins=20, kde=True, ax=ax2, color='#D08770')
    ax2.axvline(np.mean(arimagarch_errors), color='r', linestyle='--', label=f'Mean: {np.mean(arimagarch_errors):.2f}')
    ax2.axvline(np.median(arimagarch_errors), color='g', linestyle='--', label=f'Median: {np.median(arimagarch_errors):.2f}')
    ax2.set_title("ARIMA+GARCH: Distribution of Validation Errors")
    ax2.legend()

    plt.tight_layout()
    file_path = os.path.join(directory, f"validation_error_distribution_comparison_{cohort.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()


@staticmethod
def metrics_plot(df, directory):
    """
    Box plot of performance metrics across patients for both models.
    """
    metrics = ['AIC', 'BIC', 'HQIC', 'MAE', 'MSE', 'RMSE']
    
    _, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    colors = ['#8FBCBB', '#D08770']  # Teal for ARIMA, Orange for ARIMA+GARCH

    for i, metric in enumerate(metrics):
        arima_data = df[f'ARIMA_{metric}']
        arimagarch_data = df[f'ARIMA+GARCH_{metric}']
        
        bplot = axes[i].boxplot([arima_data, arimagarch_data], 
                                labels=['ARIMA', 'ARIMA+GARCH'],
                                patch_artist=True,
                                medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[i].set_title(f'Distribution of {metric}')
        axes[i].set_ylabel('Value')
        
    plt.tight_layout()
    file_path = os.path.join(directory, f"performance_metrics_boxplot_comparison_{cohort.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()


@staticmethod
def trend_plot(df, directory):
    """
    Trend analysis of forecast values for both models.
    """
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    for i, (_, row) in enumerate(df.iterrows()):
        arima_trend = pd.Series(row['ARIMA_Forecast']).pct_change().tolist()
        arimagarch_trend = pd.Series(row['ARIMA+GARCH_Forecast']).pct_change().tolist()
        validation_trend = pd.Series(row['Validation_Data']).pct_change().tolist()

        ax1.plot(arima_trend, label=f"Patient {i+1}")
        ax2.plot(arimagarch_trend, label=f"Patient {i+1}")
        ax3.plot(validation_trend, label=f"Patient {i+1}")

    ax1.set_ylabel("ARIMA Forecast Trend")
    ax1.set_title("ARIMA Trend Analysis")
    
    ax2.set_ylabel("ARIMA+GARCH Forecast Trend")
    ax2.set_title("ARIMA+GARCH Trend Analysis")
    
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Validation Trend")
    ax3.set_title("Validation Trend Analysis")

    plt.tight_layout()
    file_path = os.path.join(directory, f"forecast_trend_comparison_{cohort.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()


@staticmethod
def win_loss(cohort_df, directory):
    
    metrics = ['AIC', 'BIC', 'HQIC', 'MAE', 'MSE', 'RMSE']
    arima_wins = []
    garch_wins = []

    for metric in metrics:
        arima_better = sum(cohort_df[f'ARIMA_{metric}'] < cohort_df[f'ARIMA+GARCH_{metric}'])
        garch_better = sum(cohort_df[f'ARIMA+GARCH_{metric}'] < cohort_df[f'ARIMA_{metric}'])
        total = len(cohort_df)
        arima_wins.append(arima_better / total * 100)
        garch_wins.append(garch_better / total * 100)

    _, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(metrics, arima_wins, label='ARIMA Better', color='#8FBCBB', alpha=0.7)
    bars2 = ax.bar(metrics, garch_wins, bottom=arima_wins, label='ARIMA+GARCH Better', color='#D08770', alpha=0.7)
    ax.set_ylabel('Percentage')
    ax.set_title('Comparison of ARIMA vs ARIMA+GARCH Performance')
    ax.legend()
    
    # Add percentage annotations
    def add_percentages(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2,
                        f'{height:.1f}%', ha='center', va='center', rotation=90)

    add_percentages(bars1)
    add_percentages(bars2)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"win_loss_comparison_{cohort.lower()}.png"), dpi=300)
    plt.close()

# Plotting
roll_vs_val(cohort_df_filtered, output_dir)
error_distrib(cohort_df_filtered, output_dir)
metrics_plot(cohort_df_filtered, output_dir)
trend_plot(cohort_df_filtered, output_dir)
win_loss(cohort_df_filtered, output_dir)
print("Evaluation plots saved successfully!")
