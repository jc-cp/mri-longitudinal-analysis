import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from typing import Tuple, List, Dict
import os

class ForecastEvaluator:
    def __init__(self, results_path: str):
        """Initialize evaluator with path to results file."""
        self.results = pd.read_csv(results_path)

    #### VALIDATION ERRORS ####
    def evaluate_rolling_predictions(self) -> dict:
        """Calculate all rolling prediction metrics using original data."""
        results = {
            'ARIMA': {'errors': [], 'percentage_errors': [], 'predictions': [], 'actuals': []},
            'ARIMA+GARCH': {'errors': [], 'percentage_errors': [], 'predictions': [], 'actuals': []}
        }
        
        for idx, row in self.results.iterrows():
            try:
                val_data = np.array(eval(row['Validation_Data']))
                arima_pred = np.array(eval(row['ARIMA_Rolling_Predictions']))
                garch_pred = np.array(eval(row['ARIMA+GARCH_Rolling_Predictions']))
                
                min_len = min(len(val_data), len(arima_pred), len(garch_pred))
                val_data = val_data[:min_len]
                arima_pred = arima_pred[:min_len]
                garch_pred = garch_pred[:min_len]
                
                # Store absolute errors (in cm³)
                results['ARIMA']['errors'].extend(np.abs(val_data - arima_pred) / 1000)
                results['ARIMA+GARCH']['errors'].extend(np.abs(val_data - garch_pred) / 1000)
                
                # Store predictions and actuals for Bland-Altman
                results['ARIMA']['predictions'].extend(arima_pred / 1000)
                results['ARIMA+GARCH']['predictions'].extend(garch_pred / 1000)
                results['ARIMA']['actuals'].extend(val_data / 1000)
                results['ARIMA+GARCH']['actuals'].extend(val_data / 1000)
                
                # Store percentage errors
                non_zero_mask = val_data != 0
                if np.any(non_zero_mask):
                    arima_pe = np.abs((val_data[non_zero_mask] - arima_pred[non_zero_mask]) / 
                                     val_data[non_zero_mask]) * 100
                    garch_pe = np.abs((val_data[non_zero_mask] - garch_pred[non_zero_mask]) / 
                                     val_data[non_zero_mask]) * 100
                    results['ARIMA']['percentage_errors'].extend(arima_pe)
                    results['ARIMA+GARCH']['percentage_errors'].extend(garch_pe)
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate statistics from complete data
        metrics = {}
        for model in ['ARIMA', 'ARIMA+GARCH']:
            metrics[model] = {
                'mae': np.mean(results[model]['errors']),
                'mae_std': np.std(results[model]['errors']),
                'mae_median': np.median(results[model]['errors']),
                'mae_q1': np.percentile(results[model]['errors'], 25),
                'mae_q3': np.percentile(results[model]['errors'], 75),
                'mape': np.mean(results[model]['percentage_errors']),
                'mape_std': np.std(results[model]['percentage_errors']),
                'predictions': np.array(results[model]['predictions']),
                'actuals': np.array(results[model]['actuals'])
            }
        
        return metrics, results
    
    def plot_validation_accuracy(self, metrics: dict, results: dict, output_path: str):
        """Create validation error plots with outlier filtering for visualization."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(true_vals, pred_vals, threshold=2.5):
            """Filter outliers while keeping true-prediction pairs together."""
            differences = np.abs(true_vals - pred_vals)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            mask = np.abs(differences - mean_diff) <= threshold * std_diff
            return true_vals[mask], pred_vals[mask]
        
        # Get arrays for each model
        true_vals = np.array(metrics['ARIMA']['actuals'])
        arima_pred = np.array(metrics['ARIMA']['predictions'])
        garch_pred = np.array(metrics['ARIMA+GARCH']['predictions'])
        
        # Filter outliers while maintaining pairs
        filtered_true_arima, filtered_arima = filter_outliers(true_vals, arima_pred)
        filtered_true_garch, filtered_garch = filter_outliers(true_vals, garch_pred)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set minimum x and y to 0
        min_val = 0
        max_val = max(max(filtered_true_arima), max(filtered_arima), 
                     max(filtered_true_garch), max(filtered_garch))
        
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity')
        plt.scatter(filtered_true_arima, filtered_arima, alpha=0.5, label='ARIMA')
        plt.scatter(filtered_true_garch, filtered_garch, alpha=0.5, label='ARIMA+GARCH')
        
        # Add both MAE and MAPE to legend
        arima_mae = metrics['ARIMA']['mae']
        garch_mae = metrics['ARIMA+GARCH']['mae']
        arima_mape = metrics['ARIMA']['mape']
        garch_mape = metrics['ARIMA+GARCH']['mape']
        
        plt.title('Prediction Accuracy: True vs Predicted Values\n(Outliers removed for visualization)')
        plt.xlabel('True Values (cm³)')
        plt.ylabel('Predicted Values (cm³)')
        plt.legend(title=f'Model Performance:\nARIMA: MAE={arima_mae:.2f} cm³, MAPE={arima_mape:.1f}%\n' +
                        f'ARIMA+GARCH: MAE={garch_mae:.2f} cm³, MAPE={garch_mape:.1f}%')
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/validation_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_bland_altman(self, metrics: dict, results: dict, output_path: str):
        """Create Bland-Altman plots with outlier filtering for visualization."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(true_vals, pred_vals, threshold=2.5):
            differences = true_vals - pred_vals
            mean = np.mean(differences)
            std = np.std(differences)
            mask = np.abs(differences - mean) <= threshold * std
            return true_vals[mask], pred_vals[mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ARIMA
        true_vals, pred_vals = filter_outliers(
            metrics['ARIMA']['actuals'],
            metrics['ARIMA']['predictions']
        )
        
        # Calculate original statistics for legend
        orig_mean_diff_arima = np.mean(metrics['ARIMA']['actuals'] - metrics['ARIMA']['predictions'])
        orig_std_diff_arima = np.std(metrics['ARIMA']['actuals'] - metrics['ARIMA']['predictions'])
        
        # Plot filtered data
        mean_diff = np.mean(true_vals - pred_vals)
        std_diff = np.std(true_vals - pred_vals)
        
        ax1.scatter((true_vals + pred_vals)/2, true_vals - pred_vals,
                    alpha=0.5, label='ARIMA')
        
        ax1.axhline(y=mean_diff, color='k', linestyle='-')
        ax1.axhline(y=mean_diff + 1.96*std_diff, color='k', linestyle='--')
        ax1.axhline(y=mean_diff - 1.96*std_diff, color='k', linestyle='--')
        ax1.set_title('ARIMA\n(Outliers removed for visualization)')
        
        # Add original statistics
        ax1.text(0.05, 0.95, f'Statistics:\nMean diff: {orig_mean_diff_arima:.2f}\nStd diff: {orig_std_diff_arima:.2f}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ARIMA+GARCH
        true_vals, pred_vals = filter_outliers(
            metrics['ARIMA+GARCH']['actuals'],
            metrics['ARIMA+GARCH']['predictions']
        )
        
        # Calculate original statistics for legend
        orig_mean_diff_garch = np.mean(metrics['ARIMA+GARCH']['actuals'] - metrics['ARIMA+GARCH']['predictions'])
        orig_std_diff_garch = np.std(metrics['ARIMA+GARCH']['actuals'] - metrics['ARIMA+GARCH']['predictions'])
        
        # Plot filtered data
        mean_diff = np.mean(true_vals - pred_vals)
        std_diff = np.std(true_vals - pred_vals)
        
        ax2.scatter((true_vals + pred_vals)/2, true_vals - pred_vals,
                    alpha=0.5, label='ARIMA+GARCH')
        
        ax2.axhline(y=mean_diff, color='k', linestyle='-')
        ax2.axhline(y=mean_diff + 1.96*std_diff, color='k', linestyle='--')
        ax2.axhline(y=mean_diff - 1.96*std_diff, color='k', linestyle='--')
        ax2.set_title('ARIMA+GARCH\n(Outliers removed for visualization)')
        
        # Add original statistics
        ax2.text(0.05, 0.95, f'Statistics:\nMean diff: {orig_mean_diff_garch:.2f}\nStd diff: {orig_std_diff_garch:.2f}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        for ax in [ax1, ax2]:
            ax.set_xlabel('Mean of True and Predicted Values (cm³)')
            ax.set_ylabel('Difference (True - Predicted) (cm³)')
            ax.legend()
        
        fig.suptitle('Bland-Altman Plot: Model Agreement')
        plt.tight_layout()
        plt.savefig(f"{output_path}/validation_bland_altman.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_error_distributions(self, metrics: dict, results: dict, output_path: str):
        """Plot error distributions with outlier filtering for visualization."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(data, threshold=2.5):
            mean = np.mean(data)
            std = np.std(data)
            mask = np.abs(data - mean) <= threshold * std
            return data[mask]
        
        # Filter data for plotting
        filtered_data = {
            'ARIMA': {
                'errors': filter_outliers(np.array(results['ARIMA']['errors'])),
                'percentage_errors': filter_outliers(np.array(results['ARIMA']['percentage_errors']))
            },
            'ARIMA+GARCH': {
                'errors': filter_outliers(np.array(results['ARIMA+GARCH']['errors'])),
                'percentage_errors': filter_outliers(np.array(results['ARIMA+GARCH']['percentage_errors']))
            }
        }
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Absolute Errors
        positions = [1, 2]
        parts = ax1.violinplot(
            [filtered_data['ARIMA']['errors'], filtered_data['ARIMA+GARCH']['errors']], 
            positions, 
            showmeans=True,
            points=100
        )
        
        # Customize violin plots
        parts['bodies'][0].set_facecolor('lightblue')
        parts['bodies'][1].set_facecolor('lightgreen')
        parts['cmeans'].set_color('black')
        
        # Add scatter points with jitter
        for i, (model, color) in enumerate([('ARIMA', 'blue'), ('ARIMA+GARCH', 'green')], 1):
            data = filtered_data[model]['errors']
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax1.scatter(np.repeat(i, len(data)) + jitter, data, alpha=0.3, color=color, s=20)
        # Adjust y-axis limits based on data
        q1 = np.percentile(np.concatenate([filtered_data['ARIMA']['errors'], 
                                     filtered_data['ARIMA+GARCH']['errors']]), 25)
        q3 = np.percentile(np.concatenate([filtered_data['ARIMA']['errors'], 
                                     filtered_data['ARIMA+GARCH']['errors']]), 75)
        iqr = q3 - q1
        y_min = max(0, q1 - 1.5 * iqr)  # Don't go below 0
        y_max = q3 + 1.5 * iqr
        ax1.set_ylim(y_min, y_max)

        # Plot 2: Percentage Errors
        parts = ax2.violinplot(
            [filtered_data['ARIMA']['percentage_errors'], filtered_data['ARIMA+GARCH']['percentage_errors']], 
            positions, 
            showmeans=True,
            points=100
        )
        
        parts['bodies'][0].set_facecolor('lightblue')
        parts['bodies'][1].set_facecolor('lightgreen')
        parts['cmeans'].set_color('black')
        
        # Add scatter points with jitter
        for i, (model, color) in enumerate([('ARIMA', 'blue'), ('ARIMA+GARCH', 'green')], 1):
            data = filtered_data[model]['percentage_errors']
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax2.scatter(np.repeat(i, len(data)) + jitter, data, alpha=0.3, color=color, s=20)
        
        # Adjust y-axis limits based on data
        q1 = np.percentile(np.concatenate([filtered_data['ARIMA']['percentage_errors'], 
                                     filtered_data['ARIMA+GARCH']['percentage_errors']]), 25)
        q3 = np.percentile(np.concatenate([filtered_data['ARIMA']['percentage_errors'], 
                                     filtered_data['ARIMA+GARCH']['percentage_errors']]), 75)
        iqr = q3 - q1
        y_min = max(0, q1 - 1.5 * iqr)  # Don't go below 0
        y_max = q3 + 1.5 * iqr
        ax2.set_ylim(y_min, y_max)

        # Add statistics text boxes using original (unfiltered) data
        for ax, data_type in [(ax1, 'errors'), (ax2, 'percentage_errors')]:
            for model in ['ARIMA', 'ARIMA+GARCH']:
                orig_data = np.array(results[model][data_type])
                stats = (
                    f"{model}:\n"
                    f"Mean: {np.mean(orig_data):.2f}\n"
                    f"Median: {np.median(orig_data):.2f}\n"
                    f"Std: {np.std(orig_data):.2f}\n"
                    f"IQR: [{np.percentile(orig_data, 25):.2f}, "
                    f"{np.percentile(orig_data, 75):.2f}]"
                )
                
                # Position text box based on model
                x_pos = 0.2 if model == 'ARIMA' else 0.8
                
                ax.text(x_pos, 0.95, stats, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Customize plot
            ax.set_xticks(positions)
            ax.set_xticklabels(['ARIMA', 'ARIMA+GARCH'])
            ax.set_ylabel('Absolute Error (cm³)' if data_type == 'errors' else 'Percentage Error (%)')
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
        
        # Add titles
        ax1.set_title('Distribution of Absolute Validation Errors')
        ax2.set_title('Distribution of Percentage Validation Errors')
        
        plt.suptitle('Validation Error Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_path}/validation_error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def perform_statistical_comparison(self) -> Dict:
        """Perform statistical comparison between models."""
        stats_results = {}
        
        # Separate metrics by type
        information_criteria = ['AIC', 'BIC', 'HQIC']
        error_metrics = ['MAE', 'MSE', 'RMSE']
        
        for metric in information_criteria + error_metrics:
            arima_vals = self.results[f'ARIMA_{metric}']
            garch_vals = self.results[f'ARIMA+GARCH_{metric}']
            
            # Test for normality of differences
            differences = arima_vals - garch_vals
            _, normality_pvalue = stats.shapiro(differences)
            
            # Choose appropriate test based on normality
            if normality_pvalue > 0.05:  # Normal distribution
                test_result = stats.ttest_rel(arima_vals, garch_vals)
                test_name = 'paired t-test'
            else:  # Non-normal distribution
                test_result = stats.wilcoxon(arima_vals, garch_vals)
                test_name = 'Wilcoxon signed-rank test'
            
            # For win/loss comparison (binary outcomes)
            wins = (garch_vals < arima_vals).astype(int)
            losses = (garch_vals >= arima_vals).astype(int)
            
            # Create contingency table for McNemar's test
            contingency_table = np.array([[wins.sum(), losses.sum()], 
                                        [losses.sum(), wins.sum()]])
            
            try:
                mcnemar_result = mcnemar(contingency_table, exact=False)
                mcnemar_pvalue = mcnemar_result.pvalue
            except ValueError as e:
                print(f"Warning: Could not compute McNemar's test for {metric}: {e}")
                mcnemar_pvalue = np.nan
            
            stats_results[metric] = {
                'test_used': test_name,
                'test_result': test_result,
                'mcnemar_pvalue': mcnemar_pvalue,
                'wins_arima': losses.sum(),
                'wins_garch': wins.sum(),
                'total_cases': len(wins)
            }
            
            stats_results[metric].update({
                'metric_type': 'information_criteria' if metric in information_criteria else 'error_metrics'
            })
        
        return stats_results

    def plot_win_loss_diagram(self, stats_results: Dict, output_path: str):
        """Create win-loss diagram comparing ARIMA and ARIMA+GARCH models."""
        plt.style.use('seaborn-v0_8')
        
        # Prepare data - combine all metrics
        metrics = ['MAE', 'MSE', 'RMSE', 'AIC', 'BIC', 'HQIC']
        
        # Create single figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        wins_garch = []
        wins_arima = []
        total_cases = []
        
        for metric in metrics:
            data = stats_results[metric]
            wins_garch.append(data['wins_garch'])
            wins_arima.append(data['total_cases'] - data['wins_garch'])
            total_cases.append(data['total_cases'])
        
        # Convert to percentages
        wins_garch_pct = [w/t * 100 for w, t in zip(wins_garch, total_cases)]
        wins_arima_pct = [w/t * 100 for w, t in zip(wins_arima, total_cases)]
        
        # Create stacked bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot bars
        ax.barh(x, wins_garch_pct, width, label='ARIMA+GARCH wins', color='lightgreen')
        ax.barh(x, wins_arima_pct, width, left=wins_garch_pct, label='ARIMA wins', color='lightblue')
        
        # Add vertical line at 50%
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # Customize plot
        ax.set_yticks(x)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Percentage of Cases')
        ax.set_title('Model Performance Comparison\nWin-Loss Distribution')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Add percentage labels
        for i, (garch, arima) in enumerate(zip(wins_garch_pct, wins_arima_pct)):
            # GARCH percentage
            ax.text(garch/2, i, f'{garch:.1f}%', 
                   ha='center', va='center')
            # ARIMA percentage
            ax.text(garch + arima/2, i, f'{arima:.1f}%', 
                   ha='center', va='center')
            
            # Add significance markers
            pvalue = stats_results[metrics[i]]['test_result'].pvalue
            if pvalue < 0.05:
                ax.text(101, i, '*', ha='left', va='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"{output_path}/win_loss_diagram.png", dpi=300, bbox_inches='tight')
        plt.close()

    #### FORECASTS & TREND ANALYSIS ####
    def evaluate_forecasts(self) -> dict:
        """Calculate all forecast metrics using original data."""
        forecast_metrics = {
            'ARIMA': {
                'forecasts': [],
                'upper_ci': [],
                'lower_ci': [],
                'initial_ci_width': [],
                'final_ci_width': [],
                'growth_rates': [],
                'changes': [],
                'pct_changes': []
            },
            'ARIMA+GARCH': {
                'forecasts': [],
                'upper_ci': [],
                'lower_ci': [],
                'initial_ci_width': [],
                'final_ci_width': [],
                'growth_rates': [],
                'changes': [],
                'pct_changes': []
            }
        }
        
        # First pass to get minimum forecast length
        min_length = float('inf')
        for idx, row in self.results.iterrows():
            try:
                for model in ['ARIMA', 'ARIMA+GARCH']:
                    prefix = 'ARIMA+GARCH_' if model == 'ARIMA+GARCH' else 'ARIMA_'
                    upper = np.array(eval(row[f'{prefix}Upper_CI']))
                    min_length = min(min_length, len(upper))
            except Exception:
                continue
        
        # Collect all forecast data
        for idx, row in self.results.iterrows():
            try:
                for model in ['ARIMA', 'ARIMA+GARCH']:
                    prefix = 'ARIMA+GARCH_' if model == 'ARIMA+GARCH' else 'ARIMA_'
                    
                    # Get forecast and CI values and truncate to minimum length
                    forecast = np.array(eval(row[f'{prefix}Forecast']))[:min_length] / 1000
                    upper_ci = np.array(eval(row[f'{prefix}Upper_CI']))[:min_length] / 1000
                    lower_ci = np.array(eval(row[f'{prefix}Lower_CI']))[:min_length] / 1000
                    
                    # Store raw data
                    forecast_metrics[model]['forecasts'].append(forecast)
                    forecast_metrics[model]['upper_ci'].append(upper_ci)
                    forecast_metrics[model]['lower_ci'].append(lower_ci)
                    
                    # Calculate CI widths
                    ci_widths = upper_ci - lower_ci
                    forecast_metrics[model]['initial_ci_width'].append(ci_widths[0])
                    forecast_metrics[model]['final_ci_width'].append(ci_widths[-1])
                    
                    # Calculate growth rate
                    if ci_widths[0] > 0:  # Avoid division by zero
                        growth = ((ci_widths[-1] - ci_widths[0]) / ci_widths[0]) * 100
                        if not np.isinf(growth) and not np.isnan(growth):
                            forecast_metrics[model]['growth_rates'].append(growth)
                    
                    # Calculate absolute and percentage changes in forecast
                    abs_change = forecast[-1] - forecast[0]
                    pct_change = ((forecast[-1] - forecast[0]) / forecast[0]) * 100
                    forecast_metrics[model]['changes'].append(abs_change)
                    forecast_metrics[model]['pct_changes'].append(pct_change)
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {}
        for model in ['ARIMA', 'ARIMA+GARCH']:
            summary[model] = {
                'mean_initial_width': np.mean(forecast_metrics[model]['initial_ci_width']),
                'median_initial_width': np.median(forecast_metrics[model]['initial_ci_width']),
                'mean_final_width': np.mean(forecast_metrics[model]['final_ci_width']),
                'median_final_width': np.median(forecast_metrics[model]['final_ci_width']),
                'mean_growth_rate': np.mean(forecast_metrics[model]['growth_rates']),
                'median_growth_rate': np.median(forecast_metrics[model]['growth_rates']),
                'mean_change': np.mean(forecast_metrics[model]['changes']),
                'median_change': np.median(forecast_metrics[model]['changes']),
                'mean_pct_change': np.mean(forecast_metrics[model]['pct_changes']),
                'median_pct_change': np.median(forecast_metrics[model]['pct_changes']),
                'std_change': np.std(forecast_metrics[model]['changes']),
                'std_pct_change': np.std(forecast_metrics[model]['pct_changes'])
            }
        
        return {
            'raw_data': forecast_metrics,
            'summary': summary,
            'min_length': min_length
        }

    def analyze_forecast_trends(self) -> Dict:
        """
        Analyze forecast trends and agreement between models.
        Returns statistics about progression/stability/regression predictions.
        """
        THRESHOLD = 25  # Percentage change threshold
        
        trend_analysis = {
            'ARIMA': {'progression': 0, 'stability': 0, 'regression': 0},
            'ARIMA+GARCH': {'progression': 0, 'stability': 0, 'regression': 0},
            'agreement': {'count': 0, 'total': 0},
            'agreement_details': {
                'progression': 0,
                'stability': 0,
                'regression': 0
            }
        }
        
        for idx, row in self.results.iterrows():
            try:
                # Get forecasts
                arima_forecast = np.array(eval(row['ARIMA_Forecast']))
                garch_forecast = np.array(eval(row['ARIMA+GARCH_Forecast']))
                
                # Calculate percentage changes
                arima_change = ((arima_forecast[-1] - arima_forecast[0]) / arima_forecast[0]) * 100
                garch_change = ((garch_forecast[-1] - garch_forecast[0]) / garch_forecast[0]) * 100
                
                # Classify ARIMA trend
                if arima_change > THRESHOLD:
                    arima_trend = 'progression'
                elif arima_change < -THRESHOLD:
                    arima_trend = 'regression'
                else:
                    arima_trend = 'stability'
                
                # Classify ARIMA+GARCH trend
                if garch_change > THRESHOLD:
                    garch_trend = 'progression'
                elif garch_change < -THRESHOLD:
                    garch_trend = 'regression'
                else:
                    garch_trend = 'stability'
                
                # Update counts
                trend_analysis['ARIMA'][arima_trend] += 1
                trend_analysis['ARIMA+GARCH'][garch_trend] += 1
                
                # Check agreement
                trend_analysis['agreement']['total'] += 1
                if arima_trend == garch_trend:
                    trend_analysis['agreement']['count'] += 1
                    trend_analysis['agreement_details'][arima_trend] += 1
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate percentages
        total_cases = trend_analysis['agreement']['total']
        trend_analysis['percentages'] = {
            'ARIMA': {k: (v/total_cases)*100 for k, v in trend_analysis['ARIMA'].items()},
            'ARIMA+GARCH': {k: (v/total_cases)*100 for k, v in trend_analysis['ARIMA+GARCH'].items()},
            'agreement': (trend_analysis['agreement']['count'] / total_cases) * 100
        }
        
        return trend_analysis

    def plot_trend_analysis(self, trend_analysis: Dict, output_path: str):
        """Create visualization of trend predictions."""
        plt.style.use('seaborn-v0_8')
        
        # Prepare data for plotting
        trends = ['progression', 'stability', 'regression']
        arima_vals = [trend_analysis['percentages']['ARIMA'][t] for t in trends]
        garch_vals = [trend_analysis['percentages']['ARIMA+GARCH'][t] for t in trends]
        
        x = np.arange(len(trends))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, arima_vals, width, label='ARIMA')
        rects2 = ax.bar(x + width/2, garch_vals, width, label='ARIMA+GARCH')
        
        ax.set_ylabel('Percentage of Cases')
        ax.set_title('Forecast Trend Predictions by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(trends)
        ax.legend()
        
        # Add percentage labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/trend_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_forecast_uncertainty_growth(self, forecast_metrics: dict, output_path: str):
        """Plot uncertainty growth with outlier filtering for visualization."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(data, threshold=2.5):
            mean = np.mean(data)
            std = np.std(data)
            mask = np.abs(data - mean) <= threshold * std
            return data[mask]
        
        # Get raw data
        raw_data = forecast_metrics['raw_data']
        summary = forecast_metrics['summary']
        
        # Filter data for plotting
        filtered_data = {
            'ARIMA': {
                'upper_ci': [],
                'lower_ci': []
            },
            'ARIMA+GARCH': {
                'upper_ci': [],
                'lower_ci': []
            }
        }
        
        # Filter CI values while maintaining pairs
        for model in ['ARIMA', 'ARIMA+GARCH']:
            upper_ci = np.array(raw_data[model]['upper_ci'])
            lower_ci = np.array(raw_data[model]['lower_ci'])
            ci_widths = upper_ci - lower_ci
            
            # Filter based on CI widths
            for i in range(len(ci_widths)):
                if not np.any(np.abs(ci_widths[i] - np.mean(ci_widths)) > 2.5 * np.std(ci_widths)):
                    filtered_data[model]['upper_ci'].append(upper_ci[i])
                    filtered_data[model]['lower_ci'].append(lower_ci[i])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model, color in [('ARIMA', 'b'), ('ARIMA+GARCH', 'g')]:
            # Calculate mean and std of filtered widths
            upper = np.mean(filtered_data[model]['upper_ci'], axis=0)
            lower = np.mean(filtered_data[model]['lower_ci'], axis=0)
            widths = upper - lower
            std_widths = np.std([u - l for u, l in zip(filtered_data[model]['upper_ci'], 
                                                      filtered_data[model]['lower_ci'])], axis=0)
            
            time_points = np.arange(len(widths))
            
            ax.plot(time_points, widths, f'{color}-', label=model)
            ax.fill_between(time_points, widths - std_widths, widths + std_widths,
                           color=color, alpha=0.2)
        
        # Add original statistics to legend
        legend_text = []
        for model in ['ARIMA', 'ARIMA+GARCH']:
            stats = summary[model]
            legend_text.append(
                f"{model}:\n"
                f"Mean growth: {stats['mean_growth_rate']:.1f}%\n"
                f"Initial width: {stats['mean_initial_width']:.2f} cm³\n"
                f"Final width: {stats['mean_final_width']:.2f} cm³"
            )
        
        ax.set_xlabel('Time Steps Ahead')
        ax.set_ylabel('Confidence Interval Width (cm³)')
        ax.set_title('Growth of Forecast Uncertainty Over Time\n(Outliers removed for visualization)')
        
        # Add text boxes with original statistics
        bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="gray", alpha=0.9)
        ax.text(0.05, 0.95, legend_text[0], transform=ax.transAxes,
                verticalalignment='top', bbox=bbox_props)
        ax.text(0.95, 0.95, legend_text[1], transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/forecast_uncertainty_growth.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_forecast_fan(self, forecast_metrics: dict, output_path: str):
        """Create fan chart showing forecast uncertainty over time."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(forecast, lower, upper, threshold=2.5):
            """Filter outliers while maintaining forecast-CI pairs."""
            widths = upper - lower
            mean_width = np.mean(widths)
            std_width = np.std(widths)
            mask = np.abs(widths - mean_width) <= threshold * std_width
            return forecast[mask], lower[mask], upper[mask]
        
        # Get raw data
        raw_data = forecast_metrics['raw_data']
        summary = forecast_metrics['summary']
        
        # Select a representative case (median growth rate)
        for model in ['ARIMA', 'ARIMA+GARCH']:
            growth_rates = np.array(raw_data[model]['growth_rates'])
            median_idx = np.argsort(np.abs(growth_rates - np.median(growth_rates)))[0]
            raw_data[model]['selected_idx'] = median_idx
        
        try:
            # Get and filter data for both models
            filtered_data = {}
            for model in ['ARIMA', 'ARIMA+GARCH']:
                idx = raw_data[model]['selected_idx']
                forecast = np.array(raw_data[model]['forecasts'][idx])
                lower = np.array(raw_data[model]['lower_ci'][idx])
                upper = np.array(raw_data[model]['upper_ci'][idx])
                
                # Filter outliers
                filtered_forecast, filtered_lower, filtered_upper = filter_outliers(forecast, lower, upper)
                filtered_data[model] = {
                    'forecast': filtered_forecast,
                    'lower': filtered_lower,
                    'upper': filtered_upper
                }
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot settings for each model
            plot_settings = {
                'ARIMA': {'ax': ax1, 'color': 'b', 'alpha': 0.2},
                'ARIMA+GARCH': {'ax': ax2, 'color': 'g', 'alpha': 0.2}
            }
            
            for model, settings in plot_settings.items():
                ax = settings['ax']
                data = filtered_data[model]
                time_points = np.arange(len(data['forecast']))
                
                # Plot forecast and CI
                ax.plot(time_points, data['forecast'], f"{settings['color']}-", label='Forecast')
                ax.fill_between(time_points, data['lower'], data['upper'],
                              color=settings['color'], alpha=settings['alpha'],
                              label='95% CI')
                
                # Add original statistics
                stats = summary[model]
                stats_text = (
                    f"Original Statistics:\n"
                    f"Mean growth: {stats['mean_growth_rate']:.1f}%\n"
                    f"Initial width: {stats['mean_initial_width']:.2f} cm³\n"
                    f"Final width: {stats['mean_final_width']:.2f} cm³"
                )
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='white', alpha=0.9))
                
                ax.set_title(f'{model} Forecast\n(Outliers removed for visualization)')
                ax.set_xlabel('Time Steps Ahead')
                ax.set_ylabel('Volume (cm³)')
                ax.legend()
            
            plt.suptitle('Forecast Uncertainty Over Time\n(Representative case with median growth rate)')
            plt.tight_layout()
            plt.savefig(f"{output_path}/forecast_fan.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating fan chart: {e}")

    def plot_forecast_distributions(self, forecast_metrics: dict, output_path: str):
        """Create distribution plot of forecast changes with outlier removal."""
        plt.style.use('seaborn-v0_8')
        
        def filter_outliers(data, threshold=2.5):
            mean = np.mean(data)
            std = np.std(data)
            mask = np.abs(data - mean) <= threshold * std
            return data[mask]
        
        # Get raw data
        raw_data = forecast_metrics['raw_data']
        summary = forecast_metrics['summary']
        
        # Filter data for plotting
        filtered_data = {
            'ARIMA': {
                'changes': filter_outliers(np.array(raw_data['ARIMA']['changes'])),
                'pct_changes': filter_outliers(np.array(raw_data['ARIMA']['pct_changes']))
            },
            'ARIMA+GARCH': {
                'changes': filter_outliers(np.array(raw_data['ARIMA+GARCH']['changes'])),
                'pct_changes': filter_outliers(np.array(raw_data['ARIMA+GARCH']['pct_changes']))
            }
        }
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Absolute Changes
        positions = [1, 2]
        parts = ax1.violinplot([filtered_data['ARIMA']['changes'], filtered_data['ARIMA+GARCH']['changes']], positions, showmeans=True)
        
        # Customize violin plots
        parts['bodies'][0].set_facecolor('lightblue')
        parts['bodies'][1].set_facecolor('lightgreen')
        parts['cmeans'].set_color('black')
        
        # Add scatter points with jitter for absolute changes
        for i, (data, color) in enumerate([(filtered_data['ARIMA']['changes'], 'blue'), (filtered_data['ARIMA+GARCH']['changes'], 'green')], 1):
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax1.scatter(np.repeat(i, len(data)) + jitter, data, alpha=0.3, color=color, s=20)
        
        # Plot 2: Percentage Changes
        parts = ax2.violinplot([filtered_data['ARIMA']['pct_changes'], filtered_data['ARIMA+GARCH']['pct_changes']], positions, showmeans=True)
        
        parts['bodies'][0].set_facecolor('lightblue')
        parts['bodies'][1].set_facecolor('lightgreen')
        parts['cmeans'].set_color('black')
        
        # Add scatter points with jitter for percentage changes
        for i, (data, color) in enumerate([(filtered_data['ARIMA']['pct_changes'], 'blue'), (filtered_data['ARIMA+GARCH']['pct_changes'], 'green')], 1):
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax2.scatter(np.repeat(i, len(data)) + jitter, data, alpha=0.3, color=color, s=20)
        
        # Add statistics text boxes
        for ax, data_pairs, ylabel in [
            (ax1, [(filtered_data['ARIMA']['changes'], 'ARIMA'), (filtered_data['ARIMA+GARCH']['changes'], 'ARIMA+GARCH')], 'Absolute Change (cm³)'),
            (ax2, [(filtered_data['ARIMA']['pct_changes'], 'ARIMA'), (filtered_data['ARIMA+GARCH']['pct_changes'], 'ARIMA+GARCH')], 'Percentage Change (%)')
        ]:
            stats_text = []
            for data, label in data_pairs:
                mean = np.mean(data)
                std = np.std(data)
                median = np.median(data)
                q1, q3 = np.percentile(data, [25, 75])
                
                stats = f"{label}:\n" \
                        f"Mean: {mean:.2f}\n" \
                        f"Median: {median:.2f}\n" \
                        f"Std: {std:.2f}\n" \
                        f"IQR: [{q1:.2f}, {q3:.2f}]"
                stats_text.append(stats)
            
            # Add text boxes
            bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="gray", alpha=0.9)
            ax.text(0.05, 0.95, stats_text[0], transform=ax.transAxes, 
                    verticalalignment='top', bbox=bbox_props)
            ax.text(0.95, 0.95, stats_text[1], transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
            
            # Customize plot
            ax.set_xticks(positions)
            ax.set_xticklabels(['ARIMA', 'ARIMA+GARCH'])
            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
        
        # Add titles
        ax1.set_title('Distribution of Absolute Changes in Forecast')
        ax2.set_title('Distribution of Percentage Changes in Forecast')
        
        plt.suptitle('Forecast Change Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_path}/forecast_changes_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_forecast_uncertainty(self) -> Dict:
        """Analyze forecast uncertainty patterns."""
        uncertainty_metrics = {
            'ARIMA': {
                'initial_ci_width': [],
                'final_ci_width': [],
                'growth_rates': [],  # Store all valid growth rates
                'ci_widths': []      # Store all CI widths for each time point
            },
            'ARIMA+GARCH': {
                'initial_ci_width': [],
                'final_ci_width': [],
                'growth_rates': [],
                'ci_widths': []
            }
        }
        
        min_length = float('inf')
        # First pass to get minimum forecast length
        for idx, row in self.results.iterrows():
            try:
                arima_upper = np.array(eval(row['ARIMA_Upper_CI']))
                min_length = min(min_length, len(arima_upper))
            except Exception:
                continue
        
        for idx, row in self.results.iterrows():
            try:
                for model in ['ARIMA', 'ARIMA+GARCH']:
                    prefix = 'ARIMA+GARCH_' if model == 'ARIMA+GARCH' else 'ARIMA_'
                    upper = np.array(eval(row[f'{prefix}Upper_CI']))[:min_length] / 1000
                    lower = np.array(eval(row[f'{prefix}Lower_CI']))[:min_length] / 1000
                    
                    # Calculate CI widths for all time points
                    ci_widths = upper - lower
                    
                    if ci_widths[0] > 0:  # Only process valid cases
                        uncertainty_metrics[model]['initial_ci_width'].append(ci_widths[0])
                        uncertainty_metrics[model]['final_ci_width'].append(ci_widths[-1])
                        uncertainty_metrics[model]['ci_widths'].append(ci_widths)
                        
                        # Calculate percentage growth
                        growth = ((ci_widths[-1] - ci_widths[0]) / ci_widths[0]) * 100
                        if not np.isinf(growth) and not np.isnan(growth):
                            uncertainty_metrics[model]['growth_rates'].append(growth)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {}
        for model in ['ARIMA', 'ARIMA+GARCH']:
            ci_widths_array = np.array(uncertainty_metrics[model]['ci_widths'])
            
            summary[model] = {
                'mean_initial_width': np.mean(uncertainty_metrics[model]['initial_ci_width']),
                'median_initial_width': np.median(uncertainty_metrics[model]['initial_ci_width']),
                'mean_final_width': np.mean(uncertainty_metrics[model]['final_ci_width']),
                'median_final_width': np.median(uncertainty_metrics[model]['final_ci_width']),
                'median_growth_rate': np.median(uncertainty_metrics[model]['growth_rates']),
                'mean_growth_rate': np.mean(uncertainty_metrics[model]['growth_rates']),
                'width_over_time': np.mean(ci_widths_array, axis=0),
                'width_std_over_time': np.std(ci_widths_array, axis=0)
            }
        
        return summary

    #### REPORT GENERATION ####
    def generate_report(self, output_path: str):
        """Generate comprehensive evaluation report."""
        os.makedirs(output_path, exist_ok=True)
        
        rolling_metrics, results = self.evaluate_rolling_predictions()
        stats_results = self.perform_statistical_comparison()
    
        # Determine best model based on MAE
        arima_mae = rolling_metrics['ARIMA']['mae']
        garch_mae = rolling_metrics['ARIMA+GARCH']['mae']
        best_model = 'ARIMA+GARCH' if garch_mae < arima_mae else 'ARIMA'
        mae_improvement = abs(arima_mae - garch_mae) / max(arima_mae, garch_mae) * 100
    
    
        # Generate report
        report = ["=== Model Validation Report ===\n"]
        
        # 1. Overall Performance Metrics
        report.append("1. Model Performance Summary")
        report.append("--------------------------")
        report.append(f"Best performing model: {best_model}")
        report.append(f"Performance improvement: {mae_improvement:.2f}%\n")
        
        report.append("Mean Absolute Error (MAE):")
        report.append(f"- ARIMA: {arima_mae:.3f} cm³")
        report.append(f"- ARIMA+GARCH: {garch_mae:.3f} cm³")
        
        report.append("\nMAE Distribution:")
        for model in ['ARIMA', 'ARIMA+GARCH']:
            report.append(f"\n{model}:")
            report.append(f"- Median: {rolling_metrics[model]['mae_median']:.3f} cm³")
            report.append(f"- Std Dev: {rolling_metrics[model]['mae_std']:.3f} cm³")
            report.append(f"- Q1-Q3: [{rolling_metrics[model]['mae_q1']:.3f}, {rolling_metrics[model]['mae_q3']:.3f}] cm³")
        
        report.append("\nMean Absolute Percentage Error (MAPE):")
        report.append(f"- ARIMA: {rolling_metrics['ARIMA']['mape']:.2f}% (±{rolling_metrics['ARIMA']['mape_std']:.2f}%)")
        report.append(f"- ARIMA+GARCH: {rolling_metrics['ARIMA+GARCH']['mape']:.2f}% (±{rolling_metrics['ARIMA+GARCH']['mape_std']:.2f}%)")
        
        # 2. Model Agreement Analysis
        report.append("\n2. Model Agreement Analysis")
        report.append("--------------------------")
        for model in ['ARIMA', 'ARIMA+GARCH']:
            mean_diff = np.mean(rolling_metrics[model]['actuals'] - rolling_metrics[model]['predictions'])
            std_diff = np.std(rolling_metrics[model]['actuals'] - rolling_metrics[model]['predictions'])
            report.append(f"\n{model}:")
            report.append(f"- Mean difference: {mean_diff:.3f} cm³")
            report.append(f"- Std of differences: {std_diff:.3f} cm³")
            report.append(f"- 95% Limits of Agreement: [{mean_diff - 1.96*std_diff:.3f}, {mean_diff + 1.96*std_diff:.3f}] cm³")
        
        # 3. Clinical Interpretation
        report.append("\n3. Clinical Interpretation")
        report.append("--------------------------")
        report.append(f"- The {best_model} model shows superior performance with {mae_improvement:.1f}% lower mean absolute error")
        report.append(f"- Average prediction error: {min(arima_mae, garch_mae):.3f} cm³")
        report.append(f"- Percentage accuracy: {100 - min(rolling_metrics['ARIMA']['mape'], rolling_metrics['ARIMA+GARCH']['mape']):.1f}%")
        
        # Model Reliability
        reliability_threshold = 0.5  # cm³
        arima_reliable = np.mean(np.array(results['ARIMA']['errors']) < reliability_threshold) * 100
        garch_reliable = np.mean(np.array(results['ARIMA+GARCH']['errors']) < reliability_threshold) * 100
        
        report.append(f"\nPrediction Reliability (within {reliability_threshold} cm³):")
        report.append(f"- ARIMA: {arima_reliable:.1f}% of predictions")
        report.append(f"- ARIMA+GARCH: {garch_reliable:.1f}% of predictions")
        
        # 4. Generate Visualizations
        report.append("\n4. Visualization Files Generated")
        report.append("--------------------------")
        
        # Generate plots
        self.plot_validation_accuracy(rolling_metrics, results, output_path)
        report.append("- prediction_scatter.png: Prediction accuracy visualization")
        
        self.plot_validation_bland_altman(rolling_metrics, results, output_path)
        report.append("- bland_altman_plot.png: Model agreement analysis")
        
        self.plot_validation_error_distributions(rolling_metrics, results, output_path)
        report.append("- validation_error_distribution.png: Error distribution analysis")
        
        report.append("\n5. Model Comparison:")
        report.append("--------------------------")
        report.append("\nError Metrics (lower is better):")
        for metric in ['MAE', 'MSE', 'RMSE']:
            data = stats_results[metric]
            wins_garch = data['wins_garch']
            total_cases = data['total_cases']
            garch_percentage = (wins_garch / total_cases) * 100
            arima_percentage = 100 - garch_percentage
            
            winning_model = 'ARIMA+GARCH' if garch_percentage > 50 else 'ARIMA'
            win_percentage = max(garch_percentage, arima_percentage)
            
            report.append(f"\n{metric}:")
            report.append(f"- {winning_model} wins ({win_percentage:.1f}% of cases)")
            report.append(f"- Statistical test used: {data['test_used']}")
            report.append(f"- {data['test_used']} p-value: {data['test_result'].pvalue:.4f}")
            if not np.isnan(data['mcnemar_pvalue']):
                report.append(f"- McNemar's test p-value: {data['mcnemar_pvalue']:.4f}")
            report.append("- Interpretation:")
            report.append(f"  * {'Significant' if data['test_result'].pvalue < 0.05 else 'No significant'} difference in {metric.lower()} values")
            if not np.isnan(data['mcnemar_pvalue']):
                report.append(f"  * {'Significant' if data['mcnemar_pvalue'] < 0.05 else 'No significant'} difference in win/loss pattern")

        report.append("\nInformation Criteria (lower is better):")
        for metric in ['AIC', 'BIC', 'HQIC']:
            data = stats_results[metric]
            wins_garch = data['wins_garch']
            total_cases = data['total_cases']
            garch_percentage = (wins_garch / total_cases) * 100
            arima_percentage = 100 - garch_percentage
            
            winning_model = 'ARIMA+GARCH' if garch_percentage > 50 else 'ARIMA'
            win_percentage = max(garch_percentage, arima_percentage)
            
            report.append(f"\n{metric}:")
            report.append(f"- {winning_model} wins ({win_percentage:.1f}% of cases)")
            report.append(f"- Statistical test used: {data['test_used']}")
            report.append(f"- {data['test_used']} p-value: {data['test_result'].pvalue:.4f}")
            if not np.isnan(data['mcnemar_pvalue']):
                report.append(f"- McNemar's test p-value: {data['mcnemar_pvalue']:.4f}")
            report.append("- Interpretation:")
            report.append(f"  * {'Significant' if data['test_result'].pvalue < 0.05 else 'No significant'} difference in {metric.lower()} values")
            if not np.isnan(data['mcnemar_pvalue']):
                report.append(f"  * {'Significant' if data['mcnemar_pvalue'] < 0.05 else 'No significant'} difference in win/loss pattern")

        self.plot_win_loss_diagram(stats_results, output_path)
        report.append("- win_loss_diagram.png: Win/loss diagram between models")
        
        # 6. forecast analysis
        report.append("\n6. Forecast Analysis:")
        report.append("--------------------------")
        forecast_metrics = self.evaluate_forecasts()
        summary = forecast_metrics['summary']
        raw_data = forecast_metrics['raw_data']

        # Uncertainty Growth Analysis
        report.append("\nUncertainty Growth Analysis:")
        for model in ['ARIMA', 'ARIMA+GARCH']:
            stats = summary[model]
            report.append(f"\n{model}:")
            report.append(f"- Initial CI width: {stats['mean_initial_width']:.2f} ± {np.std(raw_data[model]['initial_ci_width']):.2f} cm³")
            report.append(f"- Final CI width: {stats['mean_final_width']:.2f} ± {np.std(raw_data[model]['final_ci_width']):.2f} cm³")
            report.append(f"- Median growth rate: {stats['median_growth_rate']:.1f}%")
            report.append(f"- Mean growth rate: {stats['mean_growth_rate']:.1f}% ± {np.std(raw_data[model]['growth_rates']):.1f}%")

        # Forecast Changes Analysis
        report.append("\nForecast Changes Analysis:")
        for model in ['ARIMA', 'ARIMA+GARCH']:
            stats = summary[model]
            report.append(f"\n{model}:")
            report.append(f"- Absolute change: {stats['mean_change']:.2f} ± {stats['std_change']:.2f} cm³")
            report.append(f"- Percentage change: {stats['mean_pct_change']:.1f}% ± {stats['std_pct_change']:.1f}%")
            report.append(f"- Median absolute change: {stats['median_change']:.2f} cm³")
            report.append(f"- Median percentage change: {stats['median_pct_change']:.1f}%")

        # Comparative Analysis
        report.append("\nComparative Uncertainty Analysis:")
        arima_growth = summary['ARIMA']['median_growth_rate']
        garch_growth = summary['ARIMA+GARCH']['median_growth_rate']
        better_model = 'ARIMA+GARCH' if garch_growth < arima_growth else 'ARIMA'
        
        report.append(f"- {better_model} shows more controlled uncertainty growth")
        report.append("- Initial uncertainty comparison:")
        report.append(f"  * ARIMA: {summary['ARIMA']['mean_initial_width']:.2f} cm³")
        report.append(f"  * ARIMA+GARCH: {summary['ARIMA+GARCH']['mean_initial_width']:.2f} cm³")
        report.append("- Final uncertainty comparison:")
        report.append(f"  * ARIMA: {summary['ARIMA']['mean_final_width']:.2f} cm³")
        report.append(f"  * ARIMA+GARCH: {summary['ARIMA+GARCH']['mean_final_width']:.2f} cm³")

        # Clinical Implications
        report.append("\nClinical Implications of Forecasts:")
        report.append(f"- Uncertainty grows {min(arima_growth, garch_growth):.1f}% - {max(arima_growth, garch_growth):.1f}% over forecast period")
        report.append(f"- Initial prediction uncertainty: {min(summary['ARIMA']['mean_initial_width'], summary['ARIMA+GARCH']['mean_initial_width']):.2f} - {max(summary['ARIMA']['mean_initial_width'], summary['ARIMA+GARCH']['mean_initial_width']):.2f} cm³")
        report.append(f"- Final prediction uncertainty: {min(summary['ARIMA']['mean_final_width'], summary['ARIMA+GARCH']['mean_final_width']):.2f} - {max(summary['ARIMA']['mean_final_width'], summary['ARIMA+GARCH']['mean_final_width']):.2f} cm³")

        # Generate forecast visualizations
        report.append("\n6. Forecast Visualizations Generated")
        report.append("--------------------------")
        
        self.plot_forecast_uncertainty_growth(forecast_metrics, output_path)
        report.append("- uncertainty_growth.png: Visualization of uncertainty growth over time")
        
        self.plot_forecast_fan(forecast_metrics, output_path)
        report.append("- forecast_fan.png: Fan chart showing forecast uncertainty for representative cases")
        
        self.plot_forecast_distributions(forecast_metrics, output_path)
        report.append("- forecast_changes_distribution.png: Distribution of forecast changes")

        # 7. Forecast Trend Analysis
        report.append("\n7. Forecast Trend Analysis:")
        report.append("--------------------------")
        trend_analysis = self.analyze_forecast_trends()
        
        report.append("\nTrend Predictions (±25% threshold):")
        report.append("\nARIMA predictions:")
        report.append(f"- Progression: {trend_analysis['percentages']['ARIMA']['progression']:.1f}%")
        report.append(f"- Stability: {trend_analysis['percentages']['ARIMA']['stability']:.1f}%")
        report.append(f"- Regression: {trend_analysis['percentages']['ARIMA']['regression']:.1f}%")
        
        report.append("\nARIMA+GARCH predictions:")
        report.append(f"- Progression: {trend_analysis['percentages']['ARIMA+GARCH']['progression']:.1f}%")
        report.append(f"- Stability: {trend_analysis['percentages']['ARIMA+GARCH']['stability']:.1f}%")
        report.append(f"- Regression: {trend_analysis['percentages']['ARIMA+GARCH']['regression']:.1f}%")
        
        report.append(f"\nModel Agreement: {trend_analysis['percentages']['agreement']:.1f}% of cases")
        report.append("\nAgreement Details:")
        for trend in ['progression', 'stability', 'regression']:
            agreement_count = trend_analysis['agreement_details'][trend]
            if agreement_count > 0:
                percentage = (agreement_count / trend_analysis['agreement']['count']) * 100
                report.append(f"- {trend.capitalize()}: {percentage:.1f}% of agreements")
        
        report.append("\nClinical Implications:")
        report.append("- Models agree on trend direction in majority of cases" if trend_analysis['percentages']['agreement'] > 50 
                    else "- Models show significant disagreement on trend direction")
        report.append(f"- Most common agreed prediction: {max(trend_analysis['agreement_details'].items(), key=lambda x: x[1])[0]}")
        
        # Add visualization of trend predictions
        self.plot_trend_analysis(trend_analysis, output_path)

        
        # 8. Save report
        with open(f"{output_path}/evaluation_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))

    
def main():
    output_path = '/home/juanqui55/git/mri-longitudinal-analysis/data/output/06_forecasting_evaluation'
    evaluator = ForecastEvaluator('/home/juanqui55/git/mri-longitudinal-analysis/data/output/05_volumetric_forecasting/joint/JOINT_forecasts.csv')
    evaluator.generate_report(output_path)

if __name__ == "__main__":
    main()