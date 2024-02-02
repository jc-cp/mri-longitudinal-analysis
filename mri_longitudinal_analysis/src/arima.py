"""
This script provides functionality for ARIMA-based time series prediction.
It supports loading data from .csv files.
"""
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from math import sqrt
from pmdarima import auto_arima
from cfg.src import arima_cfg
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf


class TimeSeriesDataHandler:
    """Loads time-series data and processes it for the prediction."""

    def __init__(self, directory, loading_limit):
        self.directory = directory
        self.loading_limit = loading_limit

    def load_data(self):
        """
        Loads time series data either from a directory or from a specified file.

        Returns:
        - list: A list of loaded time series data as DataFrames.
        """
        time_series_list = []
        file_names = []
        try:
            if os.path.isdir(self.directory):
                print("\tLoading data...")
                for idx, filename in enumerate(os.listdir(self.directory)):
                    if filename.endswith(".csv"):
                        if self.loading_limit and idx >= self.loading_limit:
                            break
                        filepath = os.path.join(self.directory, filename)
                        ts_data = pd.read_csv(filepath)
                        time_series_list.append(ts_data)
                        file_names.append(os.path.splitext(filename)[0])
            elif os.path.isfile(self.directory) and self.directory.endswith(".csv"):
                print("\tLoading data...")
                ts_data = pd.read_csv(self.directory)
                time_series_list.append(ts_data)
                file_names.append(os.path.splitext(self.directory)[0])
            return time_series_list, file_names
        except (FileNotFoundError, IOError) as error:
            print(f"Error loading time series: {error}")
            return []

    def load_data_generator(self):
        """Generate time series data on-the-fly for memory efficiency."""
        if os.path.isdir(self.directory):
            for filename in os.listdir(self.directory):
                if filename.endswith(".csv"):
                    filepath = os.path.join(self.directory, filename)
                    ts_data = pd.read_csv(filepath, usecols=[0, 1], parse_dates=[0], index_col=0)
                    yield ts_data.squeeze(), os.path.splitext(filename)[0]
        elif os.path.isfile(self.directory) and self.directory.endswith(".csv"):
            ts_data = pd.read_csv(self.directory, usecols=[0, 1], parse_dates=[0], index_col=0)
            yield ts_data.squeeze(), os.path.splitext(self.directory)[0]

    def process_and_interpolate_series(self, series_list, file_names, freq=1):
        """
        Process and interpolate the series with 'Age' as an integer index representing days.

        :param series_list: List of loaded time series data as DataFrames.
        :param file_names: List of filenames.
        :param freq: Frequency for interpolation (default is 1, representing daily data).
        :return: List of processed and interpolated series.
        """
        processed_series_list = []

        for idx, ts_data in enumerate(series_list):
            print(f"\tInterpolating data for: {file_names[idx]}")

            if "Age" not in ts_data.columns or "Volume" not in ts_data.columns:
                print(f"Warning: 'Age' or 'Volume' column missing in {file_names[idx]}")
                continue
            
            ts_data["Age"] = ts_data["Age"].astype(int)
            original_data = ts_data.set_index("Age")['Volume'].dropna()
            
            # Generate a new index that fills in the missing days
            new_index = range(ts_data["Age"].min(), ts_data["Age"].max() + freq, freq)
            ts_data_complete = pd.DataFrame(index=new_index)
            ts_data_complete = ts_data_complete.join(original_data).interpolate(method="linear")
            ts_data_complete.fillna(method='ffill', inplace=True)
            ts_data_complete.fillna(method='bfill', inplace=True)

            processed_series_list.append(ts_data_complete)
            patient_folder_path = self.ensure_patient_folder_exists(file_names[idx])
            filename = os.path.join(patient_folder_path, f"{file_names[idx]}_interpolated_vs_original.png")
            self.plot_original_and_interpolated(original_data, ts_data_complete, file_names[idx], filename)
                        
        return processed_series_list

    def process_series(self, series_list, arima_pred, file_names, target_column="Volume"):
        """
        Main method to handle series for csv data.
        :param series_list: List of series data.
        :param arima_pred: Constructor of the class
        :param file_names: List of filenames
        """
        for idx, ts_data in enumerate(series_list):
            volume_ts = ts_data[target_column]
            print(f"Preliminary check for patient: {file_names[idx]}")
            if arima_cfg.PLOTTING:
                print(f"\tCreating Autocorrelation plot for: {file_names[idx]}")
                arima_pred.generate_plot(volume_ts, "autocorrelation", file_names[idx])
                print(f"\tCreating Partial Autocorrelation plot for: {file_names[idx]}")
                arima_pred.generate_plot(volume_ts, "partial_autocorrelation", file_names[idx])

            print("\tChecking stationarity through ADF test.")
            is_stat = self.perform_dickey_fuller_test(data=volume_ts, patient_id=file_names[idx])
            if is_stat:
                print(f"\tPatient {file_names[idx]} is stationary.")
            else:
                print(f"\tPatient {file_names[idx]} is not stationary.")

            print("Starting prediction:")
            arima_pred.arima_prediction(
                data=volume_ts, patient_id=file_names[idx], is_stationary=is_stat
            )
        #arima_pred.save_forecasts_to_csv()
        #arima_pred.save_patient_metrics_to_csv()
        #arima_pred.print_and_save_cohort_summary()

    def ensure_patient_folder_exists(self, patient_id):
        """Ensure that a folder for the patient's results exists. If not, create it."""
        patient_folder_path = os.path.join(arima_cfg.OUTPUT_DIR, patient_id)
        if not os.path.exists(patient_folder_path):
            os.makedirs(patient_folder_path)
        return patient_folder_path

    def perform_dickey_fuller_test(self, data, patient_id):
        """Performing Dickey Fuller test to see the stationarity of series."""
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        adf_test_file_path = os.path.join(patient_folder_path, f"{patient_id}_adf_test.txt")

        # Augmented Dickey-Fuller test
        result = adfuller(data)
        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]
        icbest = result[5]
        is_stationary = p_value < 0.05

        with open(
            adf_test_file_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"ADF Statistic for patient {patient_id}: {adf_stat}\n")
            file.write(f"p-value for patient {patient_id}: {p_value}\n")
            file.write(f"Used lags for patient {patient_id}: {used_lag}\n")
            file.write(
                f"Number of observations for patient {patient_id}: {n_obs} (len(series)-"
                " used_lags)\n"
            )
            for key, value in critical_values.items():
                file.write(f"Critical Value ({key}) for patient {patient_id}: {value}\n")
            file.write(f"IC Best for patient {patient_id}: {icbest}\n")
            
            if is_stationary:
                file.write(f"The series is stationary for patient {patient_id}.\n")
            else:
                file.write(f"The series is not stationary for patient {patient_id}.\n")

        return is_stationary

    @staticmethod
    def plot_original_and_interpolated(original_data, interpolated_data, patient_id, filename):
        """
        Plots the original and interpolated data for a given patient.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(original_data, 'bo-', label='Original Data')
        plt.plot(interpolated_data, 'r*-', label='Interpolated Data')
        plt.title(f"Original vs. Interpolated Data for {patient_id}")
        plt.xlabel('Age')
        plt.ylabel('Volume')
        plt.legend()
        plt.savefig(filename)

class ArimaPrediction:
    """
    A class to handle ARIMA-based time series prediction.
    """

    def __init__(self):
        """
        Constructor for the Arima_prediction class.
        """
        self.patient_ids = {}
        self.cohort_summary = {}
        self.cohort_metrics = {"aic": [], "bic": [], "hqic": []}
        os.makedirs(arima_cfg.OUTPUT_DIR, exist_ok=True)

        self.plot_types = {
            "autocorrelation": {
                "function": autocorrelation_plot,
                "title": "Autocorrelation Plot",
                "xlabel": "Lag",
                "ylabel": "Autocorrelation",
            },
            "partial_autocorrelation": {
                "function": self.custom_plot_pacf,
                "title": "Partial Autocorrelation Plot",
                "xlabel": "Lag",
                "ylabel": "Partial Autocorrelation",
            },
            "residuals": {
                "function": lambda data: pd.Series(
                    data
                ).plot(),  # Use lambda to adapt the plot() method
                "title": "Residuals Plot",
                "xlabel": "Age (in days)",
                "ylabel": "Residuals",
            },
            "density": {
                "function": lambda data: pd.Series(data).plot(
                    kind="kde"
                ),  # Use lambda for density plot
                "title": "Density Plot",
                "xlabel": "Residual Value",
                "ylabel": "Density",
            },
        }

    ############
    # Plotting #
    ############

    def custom_plot_pacf(self, data):
        """
        Plots the Partial Autocorrelation Function (PACF) of the given time series data
        along with the confidence intervals. The function avoids the limitations of
        standard PACF plotting tools by generating a custom plot.

        Parameters:
        - data (array-like): Time series data.
        - alpha (float, optional): Significance level used to compute the confidence intervals.
        Defaults to 0.05, indicating 95% confidence intervals.

        Returns:
        None. The function directly plots the PACF using matplotlib.
        """
        nlags = min(len(data) // 2 - 1, 40)
        values, confint = pacf(data, nlags=nlags, alpha=0.05)
        x = range(len(values))

        plt.figure(figsize=(10, 6))
        plt.stem(x, values, basefmt="b")
        plt.errorbar(
            x,
            values,
            yerr=[values - confint[:, 0], confint[:, 1] - values],
            fmt="o",
            color="b",
            capsize=5,
        )
        plt.hlines(0, xmin=0, xmax=len(values) - 1, colors="r", linestyles="dashed")

    def generate_plot(
        self,
        data,
        plot_type,
        patient_id,
    ):
        """
        Generates and saves various plots based on the provided data and plot type.

        Parameters:
        - data (Series): Time series data.
        - plot_type (str): Type of the plot to generate.
        - patient_id (str): Name of the data file.
        """
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        figure_path = os.path.join(patient_folder_path, f"{patient_id}_{plot_type}.png")

        plt.figure(figsize=(10, 6))

        plot_func = self.plot_types[plot_type]["function"]
        plot_func(data)

        plt.title(self.plot_types[plot_type]["title"] + f" for {patient_id}")
        plt.xlabel(self.plot_types[plot_type]["xlabel"])
        plt.ylabel(self.plot_types[plot_type]["ylabel"])

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(figure_path)
        plt.close()

    def _save_arima_fig(
        self, stationary_data, rolling_index, rolling_predictions, forecast_mean, forecast_steps, conf_int, patient_id
    ):
        """
        Plot the historical data, rolling forecasts, future forecasts, and adjusted confidence intervals.
        """
        # Historical data plot
        plt.figure(figsize=(12, 6))
        plt.plot(stationary_data, label='Historical Data', color='blue')

        # Rolling forecasts plot
        plt.plot(rolling_index, rolling_predictions, label='Rolling Forecasts', color='green', linestyle='--')

        # Future forecasts plot
        last_age = stationary_data.index[-1]
        future_ages = np.arange(last_age + 1, last_age + 1 + forecast_steps)  # Generate future ages
        plt.plot(future_ages, forecast_mean, label='Future Forecast', color='red')

        # Adjusted confidence intervals plot
        lower_bounds, upper_bounds = conf_int  # Assuming conf_int is a tuple of (lower_bounds, upper_bounds)
        plt.fill_between(future_ages, lower_bounds, upper_bounds, color='pink', alpha=0.3, label='95% Confidence Interval')

        plt.title("ARIMA Forecast with Confidence Intervals")
        plt.legend()
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        figure_path = os.path.join(patient_folder_path, f"{patient_id}_forecast_plot.png")
        plt.savefig(figure_path)
        plt.close()

    def adjust_confidence_intervals(self, original_series, forecast_mean, conf_int, d_value):
        """
        Adjust the confidence intervals based on the inverted forecast mean.
        This function assumes conf_int is an array with shape (2, forecast_steps),
        where the first row contains the lower bounds, and the second row contains the upper bounds.
        """
        if d_value > 0:
            # Invert the differencing for the forecast mean to get the last actual value
            last_actual_value = original_series.iloc[-1]

            # Calculate the differences for lower and upper bounds relative to the forecast mean
            lower_bound_diff = conf_int[0, :] - forecast_mean
            upper_bound_diff = conf_int[1, :] - forecast_mean

            # Apply the differences to the last actual value to approximate the original scale
            adjusted_lower_bounds = last_actual_value + lower_bound_diff.cumsum()
            adjusted_upper_bounds = last_actual_value + upper_bound_diff.cumsum()

            return adjusted_lower_bounds, adjusted_upper_bounds
        else:
            # If no differencing was applied, use the confidence intervals as is
            return conf_int[0, :], conf_int[1, :]

    #################
    # Main function #
    #################

    def arima_prediction(
        self,
        data,
        patient_id,
        is_stationary=False,
        p_value=None,
        d_value=None,
        q_value=None,
        forecast_steps=3,
        p_range=range(5),
        q_range=range(5),
        rolling_forecast_size=0.8
    ):
        """Actual arima prediction method. Gets the corresponding
        p,d,q values from analysis and performs a prediction."""

        # Make series stationary and gets the differencing d_value
        if not is_stationary:
            stationary_data, d_value = self._make_series_stationary(data)
            print("\tMade data stationary!")

        else:
            stationary_data = data
            d_value = 0

        # Get range for p_value from partial correlation
        if p_value is None:
            suggested_p_value = self._determine_p_from_pacf(stationary_data)
            print("\tSuggested p_value:", suggested_p_value)
            p_range = range(max(0, suggested_p_value - 2), suggested_p_value + 3)
            print("\tGotten p_range!")

        # Get range for q_value from auto correlation
        if q_value is None:
            suggested_q_value = self._determine_q_from_acf(stationary_data)
            print("\tSuggested q_value:", suggested_q_value)
            q_range = range(max(0, suggested_q_value - 2), suggested_q_value + 3)
            print("\tGotten q_range!")

        try:
            # Split the data 
            split_idx = int(len(stationary_data) * rolling_forecast_size)
            training_data = stationary_data.iloc[:split_idx]
            testing_data = stationary_data.iloc[split_idx:]
            
            # Get the best ARIMA order based on training data
            if p_value is None or q_value is None:
                p_value, d_value, q_value = self.find_best_arima_order(
                    training_data, p_range, d_value, q_range
                )
                best_order = (p_value, d_value, q_value)
                print(f"\tBest ARIMA order: ({p_value}, {d_value}, {q_value})")

            # Rolling forecast
            rolling_predictions = []
            rolling_index = []
            train_model = ARIMA(training_data, order=best_order)
            train_model_fit = train_model.fit()
            for t in range(len(testing_data)):
                yhat = train_model_fit.forecast()
                yhat = self.invert_differencing(training_data, yhat, d_value)
                print(f"\tForecast for patient {patient_id} is {yhat}.")

                rolling_index.append(testing_data.index[t])
                rolling_predictions.append(yhat)
                new_observation = pd.Series([testing_data.iloc[t]], index=[testing_data.index[t]])
                train_model_fit = train_model_fit.append(new_observation, refit=True)
            
            # RMSE
            rmse = sqrt(mean_squared_error(testing_data, rolling_predictions))
            print(f'\tRolling Forecast RMSE for patient {patient_id}: {rmse}')
                        
            # Final model fitting
            final_model = ARIMA(stationary_data, order=best_order)
            final_model_fit = final_model.fit()
            print(f"\tModel fit for patient {patient_id}.")

            # Metrics
            aic = final_model_fit.aic
            bic = final_model_fit.bic
            hqic = final_model_fit.hqic

            if arima_cfg.DIAGNOSTICS:
                self._diagnostics(final_model_fit, patient_id)
                print(f"ARIMA model summary for patient {patient_id}:\n{final_model_fit.summary()}")
                # Plot residual errors
                residuals = final_model_fit.resid
                self.generate_plot(residuals, "residuals", patient_id)
                self.generate_plot(residuals, "density", patient_id)
                print(residuals.describe())
                print(f"AIC: {aic}, BIC: {bic}, HQIC: {hqic}")

            # Forecast and plotting
            forecast_steps = self._get_adaptive_forecast_steps(stationary_data)
            forecast = final_model_fit.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            stderr = forecast.se_mean
            conf_int = forecast.conf_int()
            forecast_mean = self.invert_differencing(stationary_data, forecast_mean, d_value)
            conf_int = self.adjust_confidence_intervals(stationary_data, forecast_mean, conf_int, d_value)
            
            # Save forecasts plots
            self._save_arima_fig(stationary_data, rolling_index, rolling_predictions, forecast_mean, forecast_steps, conf_int, patient_id)

            # Saving to df
            self.cohort_summary[patient_id] = {
                "forecast_mean": forecast_mean.tolist(),
                "stderr": stderr.tolist(),
                "CI": conf_int,
                "aic": aic,
                "bic": bic,
                "hqic": hqic,
                "residuals": residuals,
            }

            self.update_cohort_metrics(aic, bic, hqic)

        except (IOError, ValueError) as error:
            print("An error occurred:", str(error))

    ###########################
    # ARIMA variables methods #
    ###########################

    def _make_series_stationary(
        self,
        data,
        max_diff=3,
    ):
        """
        Returns the differenced series until it becomes stationary or reaches max
        allowed differencing.
        """
        d_value = 0  # Track differencing order
        result = adfuller(data)
        p_value = result[1]

        while p_value >= 0.05 and d_value < max_diff:
            data = data.diff().dropna()
            result = adfuller(data)
            p_value = result[1]
            d_value += 1

        if p_value < 0.01 and d_value > 0:
            data = data.diff(-1).dropna()  # Inverse the last differencing
            d_value -= 1

        return data, d_value

    def _determine_p_from_pacf(self, data, alpha=0.05):
        """
        Returns the optimal p value for ARIMA based on PACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for PACF.

        Returns:
        - p value
        """

        pacf_vals, confint = pacf(data, alpha=alpha, nlags=min(len(data) // 2 - 1, 40), method="ywmle")
        significant_lags = np.where((pacf_vals > confint[:, 1]) | (pacf_vals < confint[:, 0]))[0]
        
        # Exclude lag 0 which always significant
        significant_lags = significant_lags[significant_lags > 0]
        
        if significant_lags.size > 0:
            # Return the first significant lag as p
            return significant_lags[0] - 1  # Adjusting index to lag
        return 1

    def _determine_q_from_acf(self, data, alpha=0.05):
        """
        Returns the optimal q value for ARIMA based on ACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for ACF.

        Returns:
        - q value
        """
        acf_vals, confint = acf(data, alpha=alpha, nlags=min(len(data) // 2 - 1, 10), fft=True)

        significant_lags = np.where((acf_vals > confint[:, 1]) | (acf_vals < confint[:, 0]))[0]
        
        # Exclude lag 0 which always significant
        significant_lags = significant_lags[significant_lags > 0]
        
        if significant_lags.size > 0:
            # Return the first significant lag as q
            return significant_lags[0] - 1  # Adjusting index to lag
        return 1

    def find_best_arima_order(self, stationary_data, p_range, d_value, q_range):
        """
        Determine the best ARIMA order based on AIC.

        Parameters:
        - data (pd.Series): The time series data for which the ARIMA order needs to be determined.
        - p_range (range): The range of values for the ARIMA 'p' parameter to be tested.
        - q_range (range): The range of values for the ARIMA 'q' parameter to be tested.

        Returns:
        - tuple: The optimal (p, d, q) order for the ARIMA model.

        """
        best_aic = float("inf")
        best_order = None

        for p_value in p_range:
            for q_value in q_range:
                try:
                    model = ARIMA(stationary_data, order=(p_value, d_value, q_value))
                    model_fit = model.fit()

                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p_value, d_value, q_value)

                except (MemoryError, ModuleNotFoundError, InterruptedError) as error:
                    print(error)
                    continue

        return best_order

    def _get_adaptive_forecast_steps(self, data):
        """
        Returns the forecast frequency based on the size of the dataset.
        """
        n_steps = len(data)
        # Forecast proportionally based on data length.
        # This takes 25% of data length as forecast steps. Adjust as needed.
        return max(1, int(n_steps * 0.25))

    def invert_differencing(self, original_series, diff_values, d_order=1):
        """
        Invert the differencing process for ARIMA forecasted values.

        Parameters:
        - original_series: The original time series data.
        - diff_values: The forecasted differenced values.
        - d_order: The order of differencing applied.

        Returns:
        - A list of integrated forecasted values.
        """
        if d_order == 0:
            return diff_values

        if d_order == 1:
            last_value = original_series.iloc[-1]
            # Build a new series combining the last original value with the forecasted values
            integrated = np.r_[last_value, diff_values].cumsum()
            return integrated[1:]

        # If d_order > 1, apply the function recursively
        else:
            integrated = self.invert_differencing(
                original_series, diff_values, d_order=1
            )
            return self.invert_differencing(
                original_series, integrated, d_order=d_order - 1
            )

    ##################
    # Output methods #
    ##################

    def save_forecasts_to_csv(self):
        """Save the forecast to a .csv file."""
        forecast_df_list = []

        for patient_id, metrics in self.cohort_summary.items():
            forecast_df = pd.DataFrame(metrics["forecast_mean"], columns=["forecast_mean"])
            forecast_df["stderr"] = metrics["stderr"]
            forecast_df["lower_ci"] = metrics["CI"].iloc[:, 0]
            forecast_df["upper_ci"] = metrics["CI"].iloc[:, 1]
            forecast_df["patient_id"] = patient_id  # Add patient_id as a column for reference
            forecast_df_list.append(forecast_df)

        # Concatenate all individual DataFrames into one
        all_forecasts_df = pd.concat(forecast_df_list, ignore_index=True)
        filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_forecasts.csv")
        all_forecasts_df.to_csv(filename, index=False)

    def _diagnostics(self, model_fit, patient_id):
        """Saves the diagnostics plot."""
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_diagnostics_plot.png"))
        plt.close()

    def update_cohort_metrics(self, aic, bic, hqic):
        """Updates the cohort metrics."""
        self.cohort_metrics["aic"].append(aic)
        self.cohort_metrics["bic"].append(bic)
        self.cohort_metrics["hqic"].append(hqic)

    def save_patient_metrics_to_csv(self):
        """Saves individual patient metrics to a CSV file."""
        patient_metrics_df = pd.DataFrame.from_dict(self.cohort_summary, orient="index")
        patient_metrics_df.reset_index(inplace=True)
        patient_metrics_df.rename(columns={"index": "patient_id"}, inplace=True)
        filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_patient_metrics.csv")
        patient_metrics_df.to_csv(filename)

    def print_and_save_cohort_summary(self):
        """Calculates and prints/saves cohort-wide summary statistics."""
        summary_data = []

        # Calculate summary statistics for each metric
        for metric, values in self.cohort_metrics.items():
            summary_stats = {
                "Metric": metric.upper(),
                "Mean": np.mean(values),
                "Std Dev": np.std(values),
                "Min": np.min(values),
                "Max": np.max(values),
            }
            summary_data.append(summary_stats)

        # Convert the list of summary statistics to a DataFrame
        summary_df = pd.DataFrame(summary_data)

        print(summary_df)
        filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_cohort_summary.csv")
        summary_df.to_csv(filename)

    def ensure_patient_folder_exists(self, patient_id):
        """Ensure that a folder for the patient's results exists. If not, create it."""
        patient_folder_path = os.path.join(arima_cfg.OUTPUT_DIR, patient_id)
        if not os.path.exists(patient_folder_path):
            os.makedirs(patient_folder_path)
        return patient_folder_path


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")
    arima_prediction = ArimaPrediction()

    print("Starting ARIMA:")
    ts_handler = TimeSeriesDataHandler(arima_cfg.TIME_SERIES_DIR, arima_cfg.LOADING_LIMIT)
    ts_data_list, filenames = ts_handler.load_data()
    print("\tData loaded!")
    interp_series = ts_handler.process_and_interpolate_series(ts_data_list, filenames)
    print("\tData interpolated!")
    ts_handler.process_series(interp_series, arima_prediction, filenames)
