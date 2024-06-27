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
                    ts_data = pd.read_csv(
                        filepath,
                        dtype={"Age": np.int64, "Volume": np.float64},
                        parse_dates=["Age"],
                        infer_datetime_format=False,
                    )
                    yield ts_data, os.path.splitext(filename)[0]

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
        # Augmented Dickey-Fuller test
        result = adfuller(volume_data)
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
                file.write(
                    f"Critical Value ({key}) for patient {patient_id}: {value}\n"
                )
            file.write(f"IC Best for patient {patient_id}: {icbest}\n")

            if is_stationary:
                file.write(f"The series is stationary for patient {patient_id}.\n")
            else:
                file.write(f"The series is not stationary for patient {patient_id}.\n")

        return is_stationary

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
            if rolling_predictions:
                rolling_index = data['Age'].iloc[split_idx:split_idx + len(rolling_predictions)]
                ax.plot(rolling_index, rolling_predictions, label="Rolling Predictions", color="green", linestyle="--")
            
            # Future forecasts plot
            last_age = data['Age'].iloc[-1]
            future_index = np.arange(last_age + 1, last_age + 1 + forecast_steps)
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

    def arima_prediction(
        self,
        data,
        patient_id,
        is_stationary=False,
        autoarima=True,
        p_value=0,
        d_value=0,
        q_value=0,
        forecast_steps=3,
        p_range=range(5),
        q_range=range(5),
        rolling_forecast_size=0.8,
    ):
        """Actual arima prediction method. Gets the corresponding
        p,d,q values from analysis and performs a prediction."""

        # Make series stationary and gets the differencing d_value
        if not is_stationary:
            stationary_data, d_value = self._make_series_stationary(data)
            self.plot_differencing_effect(patient_id, data["Volume"], d_value)
            print("\tMade data stationary! D-value:", d_value)
            if d_value > 1:
                d_value = 1

        else:
            stationary_data = data["Volume"]
            d_value = 0

        try:
            # Split the data
            original_index = stationary_data.index
            split_idx = int(len(stationary_data) * rolling_forecast_size)
            training_data = stationary_data.iloc[:split_idx]
            testing_data = stationary_data.iloc[split_idx:]

            # other values
            p_value, max_p = self._determine_p_from_pacf(stationary_data)
            q_value, max_q = self._determine_q_from_acf(stationary_data)

            rolling_predictions_arima = []
            rolling_predictions_combined = []
            accumulated_diffs = []
            if autoarima:
                auto_model = auto_arima(
                    training_data,
                    start_p=p_value,
                    start_q=q_value,
                    max_p=max_p,
                    max_q=max_q,
                    d=d_value,
                    seasonal=False,
                    trace=True,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                    scoring="mse",
                    information_criterion="aic",
                    with_intercept="auto",
                )
                best_order = auto_model.order
            else:
                print("\tSuggested p_value: ", p_value)
                p_range = range(max(0, p_value - 2), p_value + 3)
                print("\tp_range: ", p_range)
                print("\tSuggested q_value:", q_value)
                q_range = range(max(0, q_value - 2), q_value + 3)
                print("\tq_range: ", q_range)

                # Get the best ARIMA order based on training data
                p_value, d_value, q_value = self.find_best_arima_order(
                    training_data, p_range, d_value, q_range
                )
                best_order = (p_value, d_value, q_value)
                print(f"\tBest ARIMA order: ({p_value}, {d_value}, {q_value})")

            if d_value > 0:
                trend_option = "n"  # No trend
            else:
                trend_option = "c"  # Include a constant as trend

            for t in range(len(testing_data)):
                train_model = ARIMA(training_data, order=best_order, trend=trend_option)
                train_model_fit = train_model.fit()
                yhat_arima = train_model_fit.get_forecast(steps=1)
                yhat_arima = yhat_arima.predicted_mean.iloc[0]
                accumulated_diffs.append(yhat_arima)

                # GARCH model on ARIMA residuals
                residuals = train_model_fit.resid
                garch_model = arch.arch_model(residuals, vol="Garch", p=p_value, q=q_value)
                garch_result = garch_model.fit(disp='off')
                garch_forecast = garch_result.forecast(horizon=1)
                garch_variance = garch_forecast.variance.values[-1, -1]
                yhat_combined = yhat_arima + sqrt(garch_variance)
                
                if len(accumulated_diffs) == d_value:
                    diff_data = data["Volume"].iloc[split_idx + t - d_value + 1 : split_idx + t + 1]
                    yhat_arima_inverted = self.invert_differencing(
                        diff_data,
                        accumulated_diffs,
                        d_value,
                    )
                    yhat_combined_inverted = self.invert_differencing(
                        diff_data,
                        [yhat_combined] + accumulated_diffs[:-1],
                        d_value,
                    )
                    rolling_predictions_arima.append(yhat_arima_inverted[-1])
                    rolling_predictions_combined.append(yhat_combined_inverted[-1])
                    accumulated_diffs.pop(0)

                next_index = original_index[split_idx + t]
                new_observation = pd.Series(testing_data.iloc[t], index=[next_index])
                training_data = pd.concat([training_data, new_observation])

            # Metrics
            actual_observed_values = data["Volume"].iloc[
                split_idx : split_idx + len(rolling_predictions_arima)
            ].values
            
            mse_arima = mean_squared_error(actual_observed_values, rolling_predictions_arima)
            rmse_arima = sqrt(mse_arima)
            mae_arima = mean_absolute_error(actual_observed_values, rolling_predictions_arima)
            mse_combined = mean_squared_error(actual_observed_values, rolling_predictions_combined)
            rmse_combined = sqrt(mse_combined)
            mae_combined = mean_absolute_error(actual_observed_values, rolling_predictions_combined)
            print(f"\nRolling Forecast Comparison for patient {patient_id}:")
            print(f"ARIMA - MSE: {mse_arima:.4f}, RMSE: {rmse_arima:.4f}, MAE: {mae_arima:.4f}")
            print(f"ARIMA+GARCH - MSE: {mse_combined:.4f}, RMSE: {rmse_combined:.4f}, MAE: {mae_combined:.4f}")

            
            # Final model fitting and out-of-sample forecasting
            final_model = ARIMA(stationary_data, order=best_order)
            final_model_fit = final_model.fit()
            residuals = final_model_fit.resid
            forecast_steps = self._get_adaptive_forecast_steps(data["Volume"])
            # ARIMA-only forecast
            forecast_arima = final_model_fit.get_forecast(steps=forecast_steps)
            forecast_mean_arima = forecast_arima.predicted_mean
            stderr_arima = forecast_arima.se_mean
            conf_int_arima = forecast_arima.conf_int()
            forecast_mean_arima = self.invert_differencing(
                data["Volume"][-d_value:], forecast_mean_arima, d_value
            )[: len(conf_int_arima)]
            (upper_b_arima, lower_b_arima) = self._adjust_confidence_intervals(
                data["Volume"], forecast_mean_arima, conf_int_arima, d_value
            )
            print(f"\tModel fit for patient {patient_id}.")
            

            # ARIMA+GARCH forecast
            model_garch = arch.arch_model(residuals, vol="Garch", p=p_value, q=q_value) 
            result_garch = model_garch.fit()
            forecast_garch = result_garch.forecast(horizon=forecast_steps)
            forecast_garch_var = forecast_garch.variance.values[-1, :]
            inverted_garch_var = self.invert_differencing(data["Volume"][-d_value:], forecast_garch_var, d_value)[:len(forecast_mean_arima)]
            forecast_combined = forecast_mean_arima + np.sqrt(inverted_garch_var)
            stderr_combined = np.sqrt(stderr_arima**2 + forecast_garch_var)
            lower_b_comb = forecast_combined - 1.96 * stderr_combined
            upper_b_comb = forecast_combined + 1.96 * stderr_combined

            # Information criteria  
            aic_arima = final_model_fit.aic
            bic_arima = final_model_fit.bic
            hqic_arima = final_model_fit.hqic
            aic_garch = result_garch.aic
            bic_garch = result_garch.bic
            aic_combined = aic_arima + aic_garch
            bic_combined = bic_arima + bic_garch

            # Estimate HQIC for combined model
            # HQIC = -2 * log-likelihood + 2 * k * log(log(n))
            # where k is the number of parameters and n is the sample size
            n = len(data)
            k_arima = final_model_fit.df_model
            k_garch = result_garch.num_params
            k_combined = k_arima + k_garch
            log_likelihood_combined = final_model_fit.llf + result_garch.loglikelihood
            hqic_combined = -2 * log_likelihood_combined + 2 * k_combined * np.log(np.log(n))
            
            if arima_cfg.DIAGNOSTICS:
                self._diagnostics(final_model_fit, patient_id)
                # print(
                #     f"ARIMA model summary for patient {patient_id}:\n{final_model_fit.summary()}"
                # )
                # Plot residual errors
                self.generate_plot(residuals, "residuals", patient_id)
                self.generate_plot(residuals, "density", patient_id)
                print(residuals.describe())
                print(f"AIC: {aic_arima}, BIC: {bic_arima}, HQIC: {hqic_arima}")
            comparison_results = {
                'comparison':{
                    "ARIMA": {
                        "rolling_predictions": rolling_predictions_arima,
                        "rolling_mse": mse_arima,
                        "rolling_rmse": rmse_arima,
                        "rolling_mae": mae_arima,
                        "final_forecast": forecast_mean_arima.tolist(),
                        "stderr": stderr_arima.tolist(),
                        "conf_int_lower": lower_b_arima.tolist(),
                        "conf_int_upper": upper_b_arima.tolist(),
                        "aic": aic_arima,
                        "bic": bic_arima,
                        "hqic": hqic_arima
                    },
                    "ARIMA+GARCH": {
                        "rolling_predictions": rolling_predictions_combined,
                        "rolling_mse": mse_combined,
                        "rolling_rmse": rmse_combined,
                        "rolling_mae": mae_combined,
                        "final_forecast": forecast_combined.tolist(),
                        "stderr": stderr_combined.tolist(),
                        "conf_int_lower": lower_b_comb.tolist(),
                        "conf_int_upper": upper_b_comb.tolist(),
                        "aic": aic_combined,
                        "bic": bic_combined,
                        "hqic": hqic_combined
                    }
                },
                "validation_data" : actual_observed_values,
                }
            # Save forecasts plots
            self._save_arima_fig(
                data,
                rolling_predictions_arima,
                rolling_predictions_combined,
                forecast_mean_arima,
                forecast_combined,
                forecast_steps,
                lower_b_arima,
                upper_b_arima,
                lower_b_comb,
                upper_b_comb,
                patient_id,
                split_idx,
            )
            print("Figures with forecast saved!")

            
            # Saving to df
            self.cohort_summary[patient_id] = comparison_results
            metrics_arima = {'aic': aic_arima, 'bic': bic_arima, 'hqic': hqic_arima}
            metrics_combined = {'aic': aic_combined, 'bic': bic_combined, 'hqic': hqic_combined}
            self.update_cohort_metrics(metrics_arima, metrics_combined)

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
        volume_data = data["Volume"]
        d_value = 0  # Track differencing order
        result = adfuller(volume_data)
        p_value = result[1]

        while p_value >= 0.05 and d_value < max_diff:
            volume_data = volume_data.diff().dropna()
            result = adfuller(volume_data)
            p_value = result[1]
            d_value += 1

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
            os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_diagnostics_plot.png")
        )
        plt.close()

    def save_forecasts_to_csv(self):
        """Save the forecasts for both ARIMA and ARIMA+GARCH to a .csv file."""
        forecast_df_list = []
        for patient_id, metrics in self.cohort_summary.items():
            comparison = metrics.get('comparison', {})
            arima_metrics = comparison.get('ARIMA', {})
            combined_metrics = comparison.get('ARIMA+GARCH', {})

            forecast_df = pd.DataFrame({
                'Patient_ID': [patient_id],
                'ARIMA_Forecast': [arima_metrics.get('final_forecast', [])],
                'ARIMA_Stderr': [arima_metrics.get('stderr', [])],
                'ARIMA_Lower_CI': [arima_metrics.get('conf_int_lower', [])],
                'ARIMA_Upper_CI': [arima_metrics.get('conf_int_upper', [])],
                'ARIMA_Rolling_Predictions': [arima_metrics.get('rolling_predictions', [])],
                'ARIMA_MSE': [arima_metrics.get('rolling_mse', None)],
                'ARIMA_RMSE': [arima_metrics.get('rolling_rmse', None)],
                'ARIMA_MAE': [arima_metrics.get('rolling_mae', None)],
                'ARIMA_AIC': [arima_metrics.get('aic', None)],
                'ARIMA_BIC': [arima_metrics.get('bic', None)],
                'ARIMA_HQIC': [arima_metrics.get('hqic', None)],
                'ARIMA+GARCH_Forecast': [combined_metrics.get('final_forecast', [])],
                'ARIMA+GARCH_Stderr': [combined_metrics.get('stderr', [])],
                'ARIMA+GARCH_Lower_CI': [combined_metrics.get('conf_int_lower', [])],
                'ARIMA+GARCH_Upper_CI': [combined_metrics.get('conf_int_upper', [])],
                'ARIMA+GARCH_Rolling_Predictions': [combined_metrics.get('rolling_predictions', [])],
                'ARIMA+GARCH_MSE': [combined_metrics.get('rolling_mse', None)],
                'ARIMA+GARCH_RMSE': [combined_metrics.get('rolling_rmse', None)],
                'ARIMA+GARCH_MAE': [combined_metrics.get('rolling_mae', None)],
                'ARIMA+GARCH_AIC': [combined_metrics.get('aic', None)],
                'ARIMA+GARCH_BIC': [combined_metrics.get('bic', None)],
                'ARIMA+GARCH_HQIC': [combined_metrics.get('hqic', None)],
                'Validation_Data': [metrics.get('validation_data', [])]
            })
            
            forecast_df_list.append(forecast_df)

        if not forecast_df_list:
            print("Warning: No forecasts to save. forecast_df_list is empty.")
            return

        try:
            # Concatenate all individual DataFrames into one
            all_forecasts_df = pd.concat(forecast_df_list, ignore_index=True)
            filename = os.path.join(
                arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_forecasts.csv"
            )
            all_forecasts_df.to_csv(filename, index=False)
            print(f"Forecasts saved to {filename}")
        except ExceptionGroup as e:
            print(f"Error in save_forecasts_to_csv: {str(e)}")
            print(f"Number of DataFrames in forecast_df_list: {len(forecast_df_list)}")
            for i, df in enumerate(forecast_df_list):
                print(f"DataFrame {i} shape: {df.shape}")
                
    def update_cohort_metrics(self, metrics_arima, metrics_combined):
        """Updates the cohort metrics for both ARIMA and ARIMA+GARCH models."""
        for model, metrics in [('arima', metrics_arima), ('arimagarch', metrics_combined)]:
            for metric in ['aic', 'bic', 'hqic']:
                key = f"{metric}_{model}"
                if key not in self.cohort_metrics:
                    self.cohort_metrics[key] = []
                self.cohort_metrics[key].append(metrics[metric])

    def print_and_save_cohort_summary(self):
        """Calculates and prints/saves cohort-wide summary statistics for both ARIMA and ARIMA+GARCH models."""
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
        
            # Save the summary to a CSV file
            filename = os.path.join(
                arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_cohort_summary.csv"
            )
            summary_df.to_csv(filename, index=False)
            print(f"\nCohort summary saved to: {filename}")
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
