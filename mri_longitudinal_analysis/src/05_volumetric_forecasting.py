"""
This script provides functionality for ARIMA-based time series prediction.
It supports loading data from .csv files.
"""
import os
import warnings
from math import sqrt
import arch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg.src import arima_cfg
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
from scipy.interpolate import Akima1DInterpolator  # , PchipInterpolator, CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import acf, adfuller, pacf
import json
import pickle

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

    def process_and_interpolate_series(
        self,
        dataframe_list: list[pd.DataFrame],
        file_names: list[str],
        freq=arima_cfg.INTERPOLATION_FREQ,
    ) -> list[pd.DataFrame]:
        """
        Process and interpolate the series, keeping the original 'Age' structure intact
        and interpolating missing 'Volume' values.

        Parameters:
        - dataframe_list: List of DataFrames with 'Age' and 'Volume' columns.
        - file_names: List of file names corresponding to each series for identification.
        - freq: Not used in this corrected version, intended for defining interpolation frequency.
        - max_gap: Maximum gap for 'Age' to perform interpolation.

        Returns:
        - processed_series_list: List of DataFrames with interpolated 'Volume' data.
        """
        processed_series_list = []

        for idx, df in enumerate(dataframe_list):
            print(f"\tInterpolating data for: {file_names[idx]}")
            if (
                not isinstance(df, pd.DataFrame)
                or "Age" not in df.columns
                or "Volume" not in df.columns
            ):
                print(
                    f"\tWarning: Either 'Age' or 'Volume' column is missing in {file_names[idx]}"
                )
                continue

            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            df.dropna(
                subset=["Volume"], inplace=True
            )  # Drop rows where 'Volume' is NaN after conversion
            df.sort_values(by="Age", inplace=True)
            # Ensure there are enough points for interpolation
            if df.shape[0] < 2:
                print(
                    f"\tNot enough data points for interpolation in {file_names[idx]}"
                )
                continue

            min_age, max_age = df["Age"].min(), df["Age"].max()
            all_ages = np.arange(min_age, max_age + 1, freq)

            # Use specific interpolator # PCHIP or CubicSpline also popssible
            interpolator = Akima1DInterpolator(df["Age"], df["Volume"])
            interpolated_volumes = interpolator(all_ages)
            interpolated_volumes = np.clip(
                interpolated_volumes, a_min=0, a_max=None
            )  # Ensure non-negativity
            df_interpolated = pd.DataFrame(
                {"Age": all_ages, "Volume": interpolated_volumes}
            ).sort_values(by="Age")

            processed_series_list.append(df_interpolated)

            # Plotting before and after interpolation
            patient_id = file_names[idx]
            patient_folder_path = self.ensure_patient_folder_exists(patient_id)
            filename = os.path.join(
                patient_folder_path, f"{patient_id}_interpolated_vs_original.png"
            )
            self.plot_original_and_interpolated(
                df, df_interpolated, file_names[idx], filename
            )

        return processed_series_list

    def process_series(
        self, series_list, arima_pred, file_names, target_column="Volume"
    ):
        """
        Main method to handle series for csv data.
        :param series_list: List of series data.
        :param arima_pred: Constructor of the class
        :param file_names: List of filenames
        """
        for idx, ts_data in enumerate(series_list):
            try:
                volume_ts = ts_data[[target_column, 'Age']]
                print(f"Preliminary check for patient: {file_names[idx]}")
                if arima_cfg.PLOTTING:
                    print(f"\tCreating Autocorrelation plot for: {file_names[idx]}")
                    arima_pred.generate_plot(volume_ts, "autocorrelation", file_names[idx])
                    print(f"\tCreating Partial Autocorrelation plot for: {file_names[idx]}")
                    arima_pred.generate_plot(
                        volume_ts, "partial_autocorrelation", file_names[idx]
                    )

                print("\tChecking stationarity through ADF test.")
                is_stat = self.perform_dickey_fuller_test(
                    data=volume_ts, patient_id=file_names[idx]
                )
                if is_stat:
                    print(f"\tPatient {file_names[idx]} is stationary.")
                else:
                    print(f"\tPatient {file_names[idx]} is not stationary.")

                print("Starting prediction:")
                arima_pred.arima_prediction(
                data=volume_ts, patient_id=file_names[idx], is_stationary=is_stat
            )
            except (ValueError, KeyError) as error:
                print(f"An error occurred: {error}")
        arima_pred.save_forecasts_to_csv()
        arima_pred.print_and_save_cohort_summary()

    def ensure_patient_folder_exists(self, patient_id):
        """Ensure that a folder for the patient's results exists. If not, create it."""
        patient_folder_path = os.path.join(arima_cfg.OUTPUT_DIR, patient_id)
        if not os.path.exists(patient_folder_path):
            os.makedirs(patient_folder_path)
        return patient_folder_path

    def perform_dickey_fuller_test(self, data, patient_id):
        """Performing Dickey Fuller test to see the stationarity of series."""
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        adf_test_file_path = os.path.join(
            patient_folder_path, f"{patient_id}_adf_test.txt"
        )
        volume_data = data["Volume"]
        
        # Check for constant series
        volume_std = volume_data.std()
        volume_mean = volume_data.mean()
        
        if volume_std < 1e-10 or volume_std/volume_mean < 1e-5:
            # Handle constant series
            with open(adf_test_file_path, "w", encoding="utf-8") as file:
                file.write(f"Series for patient {patient_id} is constant or near-constant\n")
                file.write(f"Mean value: {volume_mean}\n")
                file.write(f"Standard deviation: {volume_std}\n")
                file.write("ADF test cannot be performed on constant series\n")
            
            # Return False to indicate non-stationarity (though technically constant series are stationary)
            return False
        
        try:
            # Augmented Dickey-Fuller test
            result = adfuller(volume_data)
            adf_stat = result[0]
            p_value = result[1]
            used_lag = result[2]
            n_obs = result[3]
            critical_values = result[4]
            icbest = result[5]
            is_stationary = p_value < 0.05
            
            with open(adf_test_file_path, "w", encoding="utf-8") as file:
                file.write(f"ADF Statistic for patient {patient_id}: {adf_stat}\n")
                file.write(f"p-value for patient {patient_id}: {p_value}\n")
                file.write(f"Used lags for patient {patient_id}: {used_lag}\n")
                file.write(
                    f"Number of observations for patient {patient_id}: {n_obs} (len(series)-"
                    " used_lags)\n"
                )
                for key, value in critical_values.items():
                    file.write(
                        f"Critical Value ({key}) for patient {patient_id}: {value}\n"
                    )
                file.write(f"IC Best for patient {patient_id}: {icbest}\n")

                if is_stationary:
                    file.write(f"The series is stationary for patient {patient_id}.\n")
                else:
                    file.write(f"The series is not stationary for patient {patient_id}.\n")

            return is_stationary
        
        except ValueError as e:
            print(f"\tError in ADF test for {patient_id}: {str(e)}")
            with open(adf_test_file_path, "w", encoding="utf-8") as file:
                file.write(f"Error in ADF test: {str(e)}\n")
            return False

    @staticmethod
    def plot_original_and_interpolated(
        original_data, interpolated_data, patient_id, filename
    ):
        """
        Plots the original and interpolated data for a given patient.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            original_data["Age"], original_data["Volume"], "bo-", label="Original Data"
        )
        plt.plot(
            interpolated_data["Age"],
            interpolated_data["Volume"],
            "r*-",
            label="Interpolated Data",
        )
        plt.title(f"Original vs. Interpolated Data for {patient_id}")
        plt.xlabel("Age")
        plt.ylabel("Volume")
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
    def plot_differencing_effect(
        self, patient_id, original_series, d_value, max_diff=3
    ):
        """
        Differentiates the original series and plots the differenced series.
        """
        _, ax = plt.subplots(max_diff + 1, 1, figsize=(10, 5 * (max_diff + 1)))

        # Original Series
        ax[0].plot(original_series, label="Original Series")
        ax[0].set_title("Original Series")
        ax[0].legend()

        # Differenced Series
        differenced_series = original_series.copy()
        for d in range(1, max_diff + 1):
            differenced_series = differenced_series.diff().dropna()
            adf_result = adfuller(differenced_series)
            ax[d].plot(differenced_series, label=f"Differenced Series (d={d_value})")
            ax[d].set_title(
                f"Step{d}- Differenced Series (p-value={adf_result[1]:.4f})"
            )
            ax[d].legend()

        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        figure_path = os.path.join(
            patient_folder_path, f"{patient_id}_differentiating.png"
        )

        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

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
        volume_data, _ = data["Volume"], data["Age"]
        nlags = min(len(volume_data) // 2 - 1, 40)
        values, confint = pacf(volume_data, nlags=nlags, alpha=0.05)
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
        self,
        data,
        rolling_predictions_arima,
        rolling_predictions_combined,
        forecast_mean_arima,
        forecast_combined,
        forecast_steps,
        lower_bounds_arima,
        upper_bounds_arima,
        lower_bounds_combined,
        upper_bounds_combined,
        patient_id,
        split_idx,
    ):
        """
        Plot the historical data, rolling forecasts, future forecasts, and adjusted confidence intervals.
        """
        # Historical data plot
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
        for ax, title, rolling_predictions, forecast_mean, lower_bounds, upper_bounds in [
            (ax1, "ARIMA Forecast", rolling_predictions_arima, forecast_mean_arima, lower_bounds_arima, upper_bounds_arima),
            (ax2, "ARIMA+GARCH Forecast", rolling_predictions_combined, forecast_combined, lower_bounds_combined, upper_bounds_combined)
        ]:
            # Historical data plot
            ax.plot(data['Age'], data['Volume'], label="Historical Data", color="blue")
            
            # Rolling predictions plot
            if isinstance(rolling_predictions, (np.ndarray, list)) and len(rolling_predictions) > 0:
                rolling_index = data['Age'].iloc[split_idx:split_idx + len(rolling_predictions)]
                ax.plot(rolling_index, rolling_predictions, label="Rolling Predictions", color="green", linestyle="--")
            
            # Future forecasts plot
            last_age = data['Age'].iloc[-1]
            future_index = np.arange(last_age + 1, last_age + 1 + forecast_steps)
            
            # Ensure all arrays are numpy arrays and have the same length
            forecast_mean = np.asarray(forecast_mean)
            lower_bounds = np.asarray(lower_bounds)
            upper_bounds = np.asarray(upper_bounds)
            
            min_len = min(len(future_index), len(forecast_mean), len(lower_bounds), len(upper_bounds))
            future_index = future_index[:min_len]
            forecast_mean = forecast_mean[:min_len]
            lower_bounds = lower_bounds[:min_len]
            upper_bounds = upper_bounds[:min_len]
            
            ax.plot(future_index, forecast_mean, label="Future Forecast", color="red")
            
            # Confidence intervals plot
            ax.fill_between(future_index, lower_bounds, upper_bounds, color='pink', alpha=0.3, label='95% Confidence Interval')
            
            ax.set_title(title)
            ax.legend()
            ax.set_ylabel("Volume [mm3]")
            ax.set_xlabel("Age [days]")
        
        plt.tight_layout()
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        figure_path = os.path.join(patient_folder_path, f"{patient_id}_forecast_plot_comparison.png")
        plt.savefig(figure_path, dpi=300)
        plt.close()
    
    def _adjust_confidence_intervals(
        self, original_series, forecast_mean, conf_int, d_value
    ):
        """
        Adjust the confidence intervals based on the inverted forecast mean.
        This function assumes conf_int is an dataframe.
        """
        if d_value > 0:
            last_observation = original_series.iloc[-1]
            # Assuming forecast_mean is a series and conf_int is a DataFrame
            first_forecast_value = forecast_mean[0]
            offset = last_observation - first_forecast_value

            # Invert differencing for the confidence interval
            lower_ci = self.invert_differencing(
                original_series[-d_value:], conf_int.iloc[:, 0], d_value
            )
            upper_ci = self.invert_differencing(
                original_series[-d_value:], conf_int.iloc[:, 1], d_value
            )

            lower_ci = lower_ci - offset
            upper_ci = upper_ci - offset
            return lower_ci, upper_ci
        else:
            return conf_int.iloc[:, 0], conf_int.iloc[:, 1]

    #################
    # Main function #
    #################

    def test_arima(self, data):
        """
        Test function.
        """
        # split into train and test sets
        X = data.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size : len(X)]
        history = [x for x in train]
        predictions = list()
        # walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(0, 2, 1))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        # evaluate forecasts
        rmse = sqrt(mean_squared_error(test, predictions))
        print("Test RMSE: %.3f" % rmse)
        # plot forecasts against actual outcomes
        plt.plot(test, color="blue")
        plt.plot(predictions, color="red")
        plt.savefig("test_arima.png")

    def arima_prediction(self, data, patient_id, is_stationary=False, rolling_forecast_size=0.8):
        """
        Fits ARIMA and ARIMA+GARCH models to the data and saves predictions.
        """
        try:
            # 1. Data Preparation and Stationarity Check
            if data["Volume"].std() == 0:
                print(f"\tWarning: Constant series detected for {patient_id}. Adding small noise.")
                data["Volume"] = data["Volume"] + np.random.normal(0, data["Volume"].mean() * 0.001, len(data["Volume"]))

            original_series = data["Volume"].copy()

            # 2. Make series stationary if needed
            if not is_stationary:
                stationary_data, d_value = self._make_series_stationary(data)
                d_value = min(d_value, 2)  # Limit differencing to 1
                print(f"\tMade data stationary! D-value: {d_value}")
            else:
                stationary_data = original_series.copy()
                d_value = 0

            # 3. Split data for model validation
            split_idx = int(len(stationary_data) * rolling_forecast_size)
            training_data = stationary_data.iloc[:split_idx]
            testing_data = stationary_data.iloc[split_idx:]

            # 4. Model Parameter Selection and Fitting
            try:
                auto_model = auto_arima(
                    training_data,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    d=d_value,
                    seasonal=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    method='lbfgs',
                )
                best_order = auto_model.order
                print(f"\tBest ARIMA order found for {patient_id}: {best_order}")
            except Exception as e:
                print(f"\tAuto ARIMA failed: {e}. Using default order.")
                best_order = (1, d_value, 1)

            # 5. Generate rolling predictions
            rolling_predictions_arima = []
            rolling_predictions_combined = []
            original_test = original_series.iloc[split_idx:]
            
            for i in range(len(testing_data)):
                try:
                    # Get training window
                    train_end = split_idx + i
                    if train_end <= d_value:  # Skip if we don't have enough data points
                        continue
                    
                    # For ARIMA, use stationary data for training
                    train_stationary = stationary_data.iloc[:train_end]
                    if len(train_stationary) <= d_value + 1:  # Ensure enough data points for differencing
                        continue
                    
                    # Fit ARIMA model on stationary data
                    try:
                        model = ARIMA(train_stationary, order=best_order)
                        model_fit = model.fit()
                    except (ValueError, np.linalg.LinAlgError) as e:
                        print(f"\tARIMA fitting failed at step {i}: {e}")
                        if rolling_predictions_arima:
                            pred_arima = rolling_predictions_arima[-1]
                        else:
                            pred_arima = float(original_series.iloc[train_end-1])
                        rolling_predictions_arima.append(pred_arima)
                        rolling_predictions_combined.append(pred_arima)
                        continue
                    
                    # Make one-step ahead prediction in stationary space
                    try:
                                            
                        forecast_result = model_fit.forecast(steps=1)
                        
                        # Handle different return types from forecast
                        if isinstance(forecast_result, pd.Series):
                            pred_stationary = forecast_result.iloc[0]
                        elif isinstance(forecast_result, np.ndarray):
                            pred_stationary = forecast_result[0]
                        else:
                            pred_stationary = forecast_result
                                                
                        # Validate prediction
                        if pred_stationary is None or (isinstance(pred_stationary, (int, float)) and pred_stationary == 0):
                            raise ValueError(f"Invalid forecast value: {pred_stationary}")
                        
                        if not np.isfinite(float(pred_stationary)):
                            raise ValueError(f"Non-finite forecast value: {pred_stationary}")
                            
                    except Exception as e:
                        print(f"\tForecast failed at step {i}: {e}")
                        print(f"\tModel fit parameters: {model_fit.params}")
                        print(f"\tModel fit AIC: {model_fit.aic}")
                        if rolling_predictions_arima:
                            pred_arima = rolling_predictions_arima[-1]
                        else:
                            pred_arima = float(original_series.iloc[train_end-1])
                        rolling_predictions_arima.append(pred_arima)
                        rolling_predictions_combined.append(pred_arima)
                        continue
                    
                    # If we differenced the data, we need to invert the transformation
                    if d_value > 0:
                        try:
                            # Get the appropriate window of original data for inverting
                            original_window = original_series.iloc[max(0, train_end-d_value):train_end]
                            if len(original_window) < d_value:
                                raise ValueError("Not enough data points for inverting differencing")
                                
                            # Invert the differencing
                            pred_original = self.invert_differencing(
                                original_window,
                                np.array([pred_stationary]),
                                d_value
                            )
                            pred_arima = float(pred_original[-1])
                        except Exception as e:
                            print(f"\tInverting differencing failed at step {i}: {e}")
                            if rolling_predictions_arima:
                                pred_arima = rolling_predictions_arima[-1]
                            else:
                                pred_arima = float(original_series.iloc[train_end-1])
                    else:
                        pred_arima = float(pred_stationary)
                    
                    rolling_predictions_arima.append(pred_arima)
                    
                    # GARCH prediction
                    try:
                        residuals = model_fit.resid
                        if isinstance(residuals, pd.Series):
                            residuals = residuals.values
                        if len(residuals) < 4:  # Minimum required for GARCH(1,1)
                            raise ValueError("Not enough residuals for GARCH modeling")
                            
                        garch_model = arch.arch_model(residuals, vol='Garch', p=1, q=1)
                        garch_result = garch_model.fit(disp='off', show_warning=False)
                        garch_forecast = garch_result.forecast(horizon=1)
                        garch_variance = float(garch_forecast.variance.values[-1, -1])
                        
                        pred_combined = pred_arima + float(np.sqrt(garch_variance))
                        rolling_predictions_combined.append(pred_combined)
                    except Exception as e:
                        print(f"\tGARCH prediction failed at step {i}: {e}")
                        rolling_predictions_combined.append(pred_arima)
                
                except Exception as e:
                    print(f"\tError in rolling prediction {i}: {e}")
                    # Use last prediction or original value if no predictions yet
                    if rolling_predictions_arima:
                        last_pred_arima = rolling_predictions_arima[-1]
                        last_pred_combined = rolling_predictions_combined[-1]
                    else:
                        last_pred_arima = float(original_series.iloc[max(0, train_end-1)])
                        last_pred_combined = last_pred_arima
                    
                    rolling_predictions_arima.append(last_pred_arima)
                    rolling_predictions_combined.append(last_pred_combined)

            # Ensure predictions match test data length
            rolling_predictions_arima = np.array(rolling_predictions_arima)
            rolling_predictions_combined = np.array(rolling_predictions_combined)
            print(f"\tRolling predictions for {patient_id} done!")
            # Ensure test values are properly aligned
            test_values = original_series.iloc[split_idx:split_idx + len(rolling_predictions_arima)].values

            # 6. Calculate performance metrics using aligned data
            metrics_arima = {
                'mse': mean_squared_error(test_values, rolling_predictions_arima),
                'rmse': np.sqrt(mean_squared_error(test_values, rolling_predictions_arima)),
                'mae': mean_absolute_error(test_values, rolling_predictions_arima)
            }
            
            metrics_combined = {
                'mse': mean_squared_error(test_values, rolling_predictions_combined),
                'rmse': np.sqrt(mean_squared_error(test_values, rolling_predictions_combined)),
                'mae': mean_absolute_error(test_values, rolling_predictions_combined)
            }

            # 7. Final forecast using full data
            final_model = ARIMA(stationary_data, order=best_order)
            final_model_fit = final_model.fit()
            
            forecast_steps = max(1, int(len(data) * 0.25))
            forecast_arima = final_model_fit.get_forecast(steps=forecast_steps)
            forecast_mean_arima = forecast_arima.predicted_mean
            conf_int_arima = forecast_arima.conf_int()

            # 8. Invert differencing for final forecasts if needed
            if d_value > 0:
                forecast_mean_arima = self.invert_differencing(
                    original_series[-d_value:],
                    forecast_mean_arima,
                    d_value
                )
                lower_b_arima, upper_b_arima = self._adjust_confidence_intervals(
                    original_series,
                    forecast_mean_arima,
                    conf_int_arima,
                    d_value
                )
            else:
                lower_b_arima = conf_int_arima.iloc[:, 0]
                upper_b_arima = conf_int_arima.iloc[:, 1]

            print(f"\tFinal ARIMA forecast done.")
            # 9. Generate GARCH forecasts
            try:
                residuals = final_model_fit.resid
                # Ensure residuals are properly aligned
                if isinstance(residuals, pd.Series):
                    residuals = residuals.values
                
                garch_model = arch.arch_model(residuals, vol='Garch', p=1, q=1)
                garch_result = garch_model.fit(disp='off')
                garch_forecast = garch_result.forecast(horizon=forecast_steps)
                garch_variance = garch_forecast.variance.values[-1, :forecast_steps]
                
                # Ensure arrays are the same length before combining
                min_len = min(len(forecast_mean_arima), len(garch_variance))
                forecast_mean_arima = forecast_mean_arima[:min_len]
                garch_variance = garch_variance[:min_len]
                
                # Combine ARIMA and GARCH forecasts
                forecast_combined = forecast_mean_arima + np.sqrt(garch_variance)
                
                # Calculate combined confidence intervals
                stderr_combined = np.sqrt(forecast_arima.se_mean[:min_len]**2 + garch_variance)
                lower_b_combined = forecast_combined - 1.96 * stderr_combined
                upper_b_combined = forecast_combined + 1.96 * stderr_combined
                
                # Ensure all arrays are the same length
                lower_b_arima = lower_b_arima[:min_len]
                upper_b_arima = upper_b_arima[:min_len]
                
            except Exception as e:
                print(f"\tGARCH fitting failed: {e}")
                forecast_combined = forecast_mean_arima
                lower_b_combined = lower_b_arima
                upper_b_combined = upper_b_arima
            
            print(f"\tFinal GARCH forecast done.")

            # 10. Save results and visualization
            arima_model_metrics = self._collect_model_metrics(final_model_fit)
            garch_model_metrics = self._collect_model_metrics(garch_result)
            combined_metrics = self._collect_combined_metrics(
                arima_model_metrics, 
                garch_model_metrics, 
                len(data)
            )
            
            # Collect all results
            results = {
                'arima': self._collect_forecast_results(
                    rolling_predictions_arima,
                    metrics_arima,
                    forecast_mean_arima,
                    forecast_arima.se_mean,
                    lower_b_arima,
                    upper_b_arima,
                    arima_model_metrics
                ),
                'combined': self._collect_forecast_results(
                    rolling_predictions_combined,
                    metrics_combined,
                    forecast_combined,
                    stderr_combined,
                    lower_b_combined,
                    upper_b_combined,
                    combined_metrics
                ),
                'metadata': {
                    'split_idx': split_idx,
                    'forecast_steps': forecast_steps,
                    'original_data': data['Volume'].tolist(),
                    'validation_data': original_test.tolist() if isinstance(original_test, (pd.Series, np.ndarray)) else original_test,  # Convert to list if needed,
                    'timestamps': data['Age'].tolist()
                }
            }
            
            # Save all results
            self._save_results(patient_id, data, results)

            print(f"\nRolling Forecast Comparison for patient {patient_id}:")
            print(f"ARIMA - MSE: {metrics_arima['mse']:.4f}, RMSE: {metrics_arima['rmse']:.4f}, MAE: {metrics_arima['mae']:.4f}")
            print(f"ARIMA+GARCH - MSE: {metrics_combined['mse']:.4f}, RMSE: {metrics_combined['rmse']:.4f}, MAE: {metrics_combined['mae']:.4f}")



        except Exception as e:
            print(f"Failed to process {patient_id}: {str(e)}")
            return None

    ###########################
    # ARIMA variables methods #
    ###########################

    def _make_series_stationary(self, data, max_diff=3):
        """Returns the differenced series until it becomes stationary."""
        volume_data = data["Volume"]
        
        # Check for constant series
        if volume_data.std() < 1e-10:
            return volume_data, 0
        
        d_value = 0
        try:
            result = adfuller(volume_data)
            p_value = result[1]
        except ValueError as e:
            if "constant" in str(e):
                return volume_data, 0
            raise e

        while p_value >= 0.05 and d_value < max_diff:
            try:
                volume_data = volume_data.diff().dropna()
                if len(volume_data) < 2:  # Check if too many values were lost
                    break
                result = adfuller(volume_data)
                p_value = result[1]
                d_value += 1
            except ValueError as e:
                if "constant" in str(e):
                    break
                raise e

        return volume_data, d_value

    def _determine_p_from_pacf(self, data, alpha=0.05):
        """
        Returns the optimal p value for ARIMA based on PACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for PACF.
        """

        pacf_vals, confint = pacf(
            data, alpha=alpha, nlags=min(len(data) // 2 - 1, 40), method="ywmle"
        )
        significant_lags = np.where(
            (pacf_vals > confint[:, 1]) | (pacf_vals < confint[:, 0])
        )[0]

        # Exclude lag 0 which always significant
        significant_lags = significant_lags[significant_lags > 0]

        if significant_lags.size > 0:
            # Return the first significant lag as p
            p_val = significant_lags[-1] - 1
            max_p = significant_lags.max()
            return p_val, max_p  # Adjusting index to lag
        return 1, 1

    def _determine_q_from_acf(self, data, alpha=0.05):
        """
        Returns the optimal q value for ARIMA based on ACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for ACF.

        Returns:
        - q value
        """
        acf_vals, confint = acf(
            data, alpha=alpha, nlags=min(len(data) // 2 - 1, 10), fft=True
        )

        significant_lags = np.where(
            (acf_vals > confint[:, 1]) | (acf_vals < confint[:, 0])
        )[0]

        # Exclude lag 0 which always significant
        significant_lags = significant_lags[significant_lags > 0]

        if significant_lags.size > 0:
            # Return the first significant lag as q
            q_value = significant_lags[-1] - 1
            max_q = significant_lags.max()
            return q_value, max_q
        return 1, 1

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
        return max(1, int(n_steps * 0.75))

    def invert_differencing(self, history, forecast, d_order=1):
        """
        Invert the differencing process for ARIMA forecasted values.

        Parameters:
        - history: The original time series data.
        - forecast: The forecasted differenced values.
        - d_value: The order of differencing applied.

        Returns:
        - A list of integrated forecasted values.
        """
        if d_order == 0:
            return forecast  # Directly return the forecast for no differencing

        if d_order == 1:
            last_value = history.iloc[-1]
            return np.cumsum(np.insert(forecast, 0, last_value))

        # For d_order=2 and above
        if d_order >= 2:
            forecast = np.insert(forecast, 0, [history.iloc[-d] for d in range(d_order, 0, -1)])
            for d in range(d_order):
                forecast = np.cumsum(forecast)
            return forecast[d_order:]

    ##################
    # Output methods #
    ##################
    def _diagnostics(self, model_fit, patient_id):
        """Saves the diagnostics plot."""
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.savefig(
            os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_diagnostics_plot.png"))
        plt.close()

    def save_forecasts_to_csv(self):
        """Save the forecasts for both ARIMA and ARIMA+GARCH to a .csv file."""
        forecast_df_list = []
        for patient_id, results in self.cohort_summary.items():
            arima_results = results['arima']
            combined_results = results['combined']
            
            forecast_df = pd.DataFrame({
                'Patient_ID': [patient_id],
                'ARIMA_Rolling_Predictions': [arima_results['rolling_predictions']],
                'ARIMA_MSE': [arima_results['rolling_metrics']['mse']],
                'ARIMA_RMSE': [arima_results['rolling_metrics']['rmse']],
                'ARIMA_MAE': [arima_results['rolling_metrics']['mae']],
                'ARIMA_Forecast': [arima_results['forecast']['mean']],
                'ARIMA_Lower_CI': [arima_results['forecast']['conf_int_lower']],
                'ARIMA_Upper_CI': [arima_results['forecast']['conf_int_upper']],
                'ARIMA_AIC': [arima_results['model_metrics']['aic']],
                'ARIMA_BIC': [arima_results['model_metrics']['bic']],
                'ARIMA_HQIC': [arima_results['model_metrics']['hqic']],
                'ARIMA+GARCH_Rolling_Predictions': [combined_results['rolling_predictions']],
                'ARIMA+GARCH_MSE': [combined_results['rolling_metrics']['mse']],
                'ARIMA+GARCH_RMSE': [combined_results['rolling_metrics']['rmse']],
                'ARIMA+GARCH_MAE': [combined_results['rolling_metrics']['mae']],
                'ARIMA+GARCH_Forecast': [combined_results['forecast']['mean']],
                'ARIMA+GARCH_Lower_CI': [combined_results['forecast']['conf_int_lower']],
                'ARIMA+GARCH_Upper_CI': [combined_results['forecast']['conf_int_upper']],
                'ARIMA+GARCH_AIC': [combined_results['model_metrics']['aic']],
                'ARIMA+GARCH_BIC': [combined_results['model_metrics']['bic']],
                'ARIMA+GARCH_HQIC': [combined_results['model_metrics']['hqic']],
                'Original_Data': [results['metadata']['original_data']],
                'Validation_Data': [results['metadata']['validation_data']],
                'Timestamps': [results['metadata']['timestamps']]
            })
            forecast_df_list.append(forecast_df)

        if forecast_df_list:
            all_forecasts_df = pd.concat(forecast_df_list, ignore_index=True)
            filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_forecasts.csv")
            all_forecasts_df.to_csv(filename, index=False)
            
            # Also save as pickle for easier loading later
            pickle_filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_forecasts.pkl")
            with open(pickle_filename, 'wb') as f:
                pickle.dump(all_forecasts_df, f)
            
            print(f"Forecasts saved to {filename} and {pickle_filename}")
        else:
            print("Warning: No forecasts to save.")

    def update_cohort_metrics(self, metrics_arima, metrics_combined):
        """Updates the cohort metrics for both ARIMA and ARIMA+GARCH models."""
        for model, metrics in [('arima', metrics_arima), ('arimagarch', metrics_combined)]:
            for metric in ['aic', 'bic', 'hqic']:
                key = f"{metric}_{model}"
                if key not in self.cohort_metrics:
                    self.cohort_metrics[key] = []
                self.cohort_metrics[key].append(metrics[metric])

    def print_and_save_cohort_summary(self):
        """Calculates and prints/saves cohort-wide summary statistics."""
        summary_data = []
        
        # List of metrics and models
        metrics = ['AIC', 'BIC', 'HQIC']
        models = ['ARIMA', 'ARIMA+GARCH']
        
        # Calculate summary statistics for each metric and model
        for metric in metrics:
            for model in models:
                key = f"{metric.lower()}_{model.lower().replace('+', '')}"
                if key in self.cohort_metrics:
                    values = self.cohort_metrics[key]
                    summary_stats = {
                        "Metric": f"{metric} ({model})",
                        "Mean": np.mean(values),
                        "Std Dev": np.std(values),
                        "Min": np.min(values),
                        "Max": np.max(values),
                    }
                    summary_data.append(summary_stats)
                else:
                    print(f"Warning: {key} not found in cohort_metrics")
        
        # Convert the list of summary statistics to a DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Print the summary
        if not summary_df.empty:
            print("\nCohort Summary Statistics:")
            print(summary_df.to_string(index=False))
            
            # Save the summary DataFrame
            if not summary_df.empty:
                # Save as CSV
                csv_filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_cohort_summary.csv")
                summary_df.to_csv(csv_filename, index=False)
                
                # Save as pickle
                pickle_filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_cohort_summary.pkl")
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(summary_df, f)
                
                # Save full cohort data
                cohort_filename = os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_full_results.pkl")
                with open(cohort_filename, 'wb') as f:
                    pickle.dump(self.cohort_summary, f)
                
                print(f"\nResults saved to:")
                print(f"- CSV: {csv_filename}")
                print(f"- Summary pickle: {pickle_filename}")
                print(f"- Full results pickle: {cohort_filename}")
        else:
            print("No summary data available.")

        # Additional analysis: Compare ARIMA vs ARIMA+GARCH
        print("\nComparison of ARIMA vs ARIMAGARCH:")
        for metric in metrics:
            arima_key = f"{metric.lower()}_arima"
            combined_key = f"{metric.lower()}_arimagarch"
            if arima_key in self.cohort_metrics and combined_key in self.cohort_metrics:
                arima_values = self.cohort_metrics[arima_key]
                combined_values = self.cohort_metrics[combined_key]
                
                better_count = sum(c < a for a, c in zip(arima_values, combined_values))
                total_count = len(arima_values)
                
                print(f"{metric}:")
                print(f"  ARIMA+GARCH better in {better_count}/{total_count} cases ({better_count/total_count*100:.2f}%)")
                print(f"  Average improvement: {np.mean(np.array(arima_values) - np.array(combined_values)):.2f}")
            else:
                print(f"Warning: {arima_key} or {combined_key} not found in cohort_metrics")
    
    def ensure_patient_folder_exists(self, patient_id):
        """Ensure that a folder for the patient's results exists. If not, create it."""
        patient_folder_path = os.path.join(arima_cfg.OUTPUT_DIR, patient_id)
        if not os.path.exists(patient_folder_path):
            os.makedirs(patient_folder_path)
        return patient_folder_path

    def _collect_model_metrics(self, model_fit):
        """
        Collect basic model metrics for both ARIMA and GARCH models.
        
        Parameters:
            model_fit: Either ARIMAResults or ARCHModelResult object
        """
        if hasattr(model_fit, 'aic'):  # Common attribute for both
            metrics = {
                'aic': model_fit.aic,
                'bic': model_fit.bic,
            }
            
            # ARIMA specific attributes
            if hasattr(model_fit, 'df_model'):
                metrics.update({
                    'hqic': getattr(model_fit, 'hqic', None),
                    'df_model': model_fit.df_model,
                    'llf': model_fit.llf
                })
            # GARCH specific attributes
            else:
                metrics.update({
                    'hqic': getattr(model_fit, 'hqic', None),
                    'df_model': model_fit.num_params,  # GARCH uses num_params instead of df_model
                    'llf': model_fit.loglikelihood  # GARCH uses loglikelihood instead of llf
                })
            
            return metrics
        else:
            # Return default values if model doesn't have these attributes
            return {
                'aic': None,
                'bic': None,
                'hqic': None,
                'df_model': 0,
                'llf': 0
            }

    def _collect_combined_metrics(self, arima_metrics, garch_metrics, n_samples):
        """
        Compute combined metrics for ARIMA+GARCH.
        
        Parameters:
            arima_metrics: Dict of ARIMA model metrics
            garch_metrics: Dict of GARCH model metrics
            n_samples: Number of samples in the dataset
        
        Returns:
            Dict of combined model metrics
        """
        # Start with basic combined metrics
        combined = {
            'aic': arima_metrics['aic'] + garch_metrics['aic'],
            'bic': arima_metrics['bic'] + garch_metrics['bic'],
        }
        
        # Calculate combined HQIC if we have all necessary components
        if all(metric is not None for metric in [arima_metrics['df_model'], 
                                               garch_metrics['df_model'], 
                                               arima_metrics['llf'], 
                                               garch_metrics['llf']]):
            k_combined = arima_metrics['df_model'] + garch_metrics['df_model']
            llf_combined = arima_metrics['llf'] + garch_metrics['llf']
            combined['hqic'] = -2 * llf_combined + 2 * k_combined * np.log(np.log(n_samples))
        else:
            combined['hqic'] = None
        
        # Add additional combined metrics
        combined.update({
            'df_model': arima_metrics['df_model'] + garch_metrics['df_model'],
            'llf': arima_metrics['llf'] + garch_metrics['llf']
        })
        
        return combined

    def _collect_forecast_results(self, predictions, metrics, forecast_mean, stderr, lower_bounds, upper_bounds, model_metrics):
        """
        Collect all results for a single model and ensure they're JSON serializable.
        
        Parameters should be numpy arrays, pandas Series, or basic Python types.
        Returns a dictionary with all values converted to lists or basic Python types.
        """
        def to_serializable(obj):
            """Convert an object to a JSON serializable format."""
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_serializable(item) for item in obj]
            return obj

        return {
            "rolling_predictions": to_serializable(predictions),
            "rolling_metrics": {
                "mse": to_serializable(metrics['mse']),
                "rmse": to_serializable(metrics['rmse']),
                "mae": to_serializable(metrics['mae'])
            },
            "forecast": {
                "mean": to_serializable(forecast_mean),
                "stderr": to_serializable(stderr),
                "conf_int_lower": to_serializable(lower_bounds),
                "conf_int_upper": to_serializable(upper_bounds)
            },
            "model_metrics": to_serializable(model_metrics)
        }

    def _save_results(self, patient_id, data, results_dict, save_plots=True):
        """Save all results for a patient."""
        # Save to class attribute for later use
        self.cohort_summary[patient_id] = results_dict
        
        # Create patient directory if it doesn't exist
        patient_folder_path = self.ensure_patient_folder_exists(patient_id)
        
        # Save patient results to JSON
        results_file = os.path.join(patient_folder_path, f"{patient_id}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyEncoder)
        
        # Save patient results to pickle (preserves numpy arrays)
        pickle_file = os.path.join(patient_folder_path, f"{patient_id}_results.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(results_dict, f)
        
        # Update cohort metrics
        self.update_cohort_metrics(
            results_dict['arima']['model_metrics'],
            results_dict['combined']['model_metrics']
        )
        
        # Save plots if requested
        if save_plots:
            self._save_arima_fig(
                data=data,
                rolling_predictions_arima=results_dict['arima']['rolling_predictions'],
                rolling_predictions_combined=results_dict['combined']['rolling_predictions'],
                forecast_mean_arima=results_dict['arima']['forecast']['mean'],
                forecast_combined=results_dict['combined']['forecast']['mean'],
                forecast_steps=len(results_dict['arima']['forecast']['mean']),
                lower_bounds_arima=results_dict['arima']['forecast']['conf_int_lower'],
                upper_bounds_arima=results_dict['arima']['forecast']['conf_int_upper'],
                lower_bounds_combined=results_dict['combined']['forecast']['conf_int_lower'],
                upper_bounds_combined=results_dict['combined']['forecast']['conf_int_upper'],
                patient_id=patient_id,
                split_idx=results_dict['metadata']['split_idx']
            )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    arima_prediction = ArimaPrediction()

    print("Starting ARIMA:")
    ts_handler = TimeSeriesDataHandler(
        arima_cfg.TIME_SERIES_DIR_COHORT, arima_cfg.LOADING_LIMIT
    )
    ts_data_list, filenames = ts_handler.load_data()
    print("\tData loaded!")
    interp_series = ts_handler.process_and_interpolate_series(ts_data_list, filenames)
    print("\tData interpolated!")
    ts_handler.process_series(interp_series, arima_prediction, filenames)
