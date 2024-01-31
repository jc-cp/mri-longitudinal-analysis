"""
This script provides functionality for ARIMA-based time series prediction.
It supports loading data from .csv files.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg.src import arima_cfg
from pandas.plotting import autocorrelation_plot
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
            print(f"Interpolating data for: {file_names[idx]}")

            # Set 'Age' as the index if it's not already
            if "Age" in ts_data.columns:
                ts_data["Age"] = ts_data["Age"].astype(int)
                ts_data.set_index("Age", inplace=True)

            # Generate a new index that fills in the missing days
            new_index = range(ts_data.index.min(), ts_data.index.max() + freq, freq)
            ts_data = ts_data.reindex(new_index)
            ts_data.interpolate(method="linear", inplace=True)

            if "Volume" not in ts_data.columns:
                print(f"Warning: 'Volume' column missing after interpolation for {file_names[idx]}")
                continue
            processed_series_list.append(ts_data)

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

    def perform_dickey_fuller_test(self, data, patient_id):
        """Performing Dickey Fuller test to see the stationarity of series."""

        # Augmented Dickey-Fuller test
        result = adfuller(data)
        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]
        icbest = result[5]
        with open(
            os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_adf_test.txt"),
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

        is_stationary = p_value < 0.05
        return is_stationary


class ArimaPrediction:
    """
    A class to handle ARIMA-based time series prediction.
    """

    def __init__(self):
        """
        Constructor for the Arima_prediction class.
        """
        self.patient_ids = {}
        self.cohort_summary = pd.DataFrame()
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
        nlags = min(len(data) // 2 - 1, 10)
        values, confint = pacf(data, nlags=nlags, alpha=0.05)
        plt.bar(range(len(values)), values)
        plt.fill_between(range(len(confint)), confint[:, 0], confint[:, 1], color="pink", alpha=0.3)

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
        plt.figure(figsize=(10, 6))

        plot_func = self.plot_types[plot_type]["function"]
        plot_func(data)

        plt.title(self.plot_types[plot_type]["title"] + f" for {patient_id}")
        plt.xlabel(self.plot_types[plot_type]["xlabel"])
        plt.ylabel(self.plot_types[plot_type]["ylabel"])

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_{plot_type}.png"))
        plt.close()

    def _save_forecast_fig(
        self,
        data,
        forecast,
        forecast_steps,
        conf_int,
        patient_id,
    ):
        plt.figure(figsize=(12, 6))
        plt.plot(data, color="blue", label="Historical Data")
        plt.plot(
            range(len(data), len(data) + forecast_steps), forecast, color="red", label="Forecast"
        )
        plt.fill_between(
            range(len(data), len(data) + forecast_steps),
            conf_int[:, 0],
            conf_int[:, 1],
            color="pink",
            alpha=0.3,
        )
        plt.title("ARIMA Forecast with Confidence Intervals")
        plt.legend()
        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_forecast_plot.png"))
        plt.close()

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
            p_range = range(max(0, suggested_p_value - 2), suggested_p_value + 3)
            print("\tGotten p_range!")

        # Get range for q_value from auto correlation
        if q_value is None:
            suggested_q_value = self._determine_q_from_acf(stationary_data)
            q_range = range(max(0, suggested_q_value - 2), suggested_q_value + 3)
            print("\tGotten q_range!")

        # Get the best ARIMA order
        if p_value is None or q_value is None:
            p_value, d_value, q_value = self.find_best_arima_order(
                stationary_data, p_range, d_value, q_range
            )
            print(f"\tBest ARIMA order: ({p_value}, {d_value}, {q_value})")

        try:
            # Get adaptive forecast steps
            forecast_steps = self._get_adaptive_forecast_steps(stationary_data)

            model = ARIMA(stationary_data, order=(p_value, d_value, q_value))
            model_fit = model.fit()
            print(f"\tModel fit for patient {patient_id}.")
            if arima_cfg.DIAGNOSTICS:
                self._diagnostics(model_fit, patient_id)
                print(f"ARIMA model summary for patient {patient_id}:\n{model_fit.summary()}")

            # Metrics
            aic = model_fit.aic
            bic = model_fit.bic
            hqic = model_fit.hqic

            # Forecast and plotting
            forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)
            if d_value > 0:
                forecast = ArimaPrediction.invert_differencing(data, forecast, d_value)
            forecast_series = pd.Series(forecast, name="Predictions")

            # Save forecasts plots
            self._save_forecast_fig(data, forecast, forecast_steps, conf_int, patient_id)

            # plot residual errors
            residuals = model_fit.resid
            self.generate_plot(residuals, "residuals", patient_id)
            self.generate_plot(residuals, "density", patient_id)
            if arima_cfg.DIAGNOSTICS:
                print(residuals.describe())

            # Saving to df
            self.cohort_summary[patient_id] = {
                "forecast": forecast_series.tolist(),
                "stderr": stderr,
                "CI": conf_int,
                "aic": aic,
                "bic": bic,
                "hqic": hqic,
                "residuals": residuals,
            }
            self._save_cohort_summary_to_csv()

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

        pacf_vals, confint = pacf(data, alpha=alpha, nlags=min(len(data) // 2 - 1, 10))
        significant_lags = [
            lag
            for lag, pacf_val in enumerate(pacf_vals)
            if pacf_val > confint[lag, 1] or pacf_val < confint[lag, 0]
        ]

        if significant_lags:
            return max(significant_lags)
        return 1  # Default to 1 if none are significant

    def _determine_q_from_acf(self, data, alpha=0.05):
        """
        Returns the optimal q value for ARIMA based on ACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for ACF.

        Returns:
        - q value
        """
        acf_vals, confint = acf(data, alpha=alpha, nlags=min(len(data) // 2 - 1, 10))

        significant_lags = [
            lag
            for lag, acf_val in enumerate(acf_vals)
            if acf_val > confint[lag, 1] or acf_val < confint[lag, 0]
        ]

        if significant_lags:
            return max(significant_lags)
        return 1  # Default to 1 if none are significant

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

    @staticmethod
    def invert_differencing(original_series, diff_values, d_order=1):
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
            integrated = ArimaPrediction.invert_differencing(
                original_series, diff_values, d_order=1
            )
            return ArimaPrediction.invert_differencing(
                original_series, integrated, d_order=d_order - 1
            )

    ##################
    # Output methods #
    ##################

    def _save_cohort_summary_to_csv(self):
        """Save the forecast to a .csv file."""
        self.cohort_summary.to_csv(
            os.path.join(arima_cfg.OUTPUT_DIR, f"{arima_cfg.COHORT}_summary.csv"),
            index=False,
        )

    def _diagnostics(self, model_fit, patient_id):
        """Saves the diagnostics plot."""
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{patient_id}_diagnostics_plot.png"))
        plt.close()


if __name__ == "__main__":
    arima_prediction = ArimaPrediction()

    print("Starting ARIMA:")
    ts_handler = TimeSeriesDataHandler(arima_cfg.TIME_SERIES_DIR, arima_cfg.LOADING_LIMIT)
    ts_data_list, filenames = ts_handler.load_data()
    print("\tData loaded!")
    processed_series = ts_handler.process_and_interpolate_series(ts_data_list, filenames)
    print("\tData interpolated!")
    ts_handler.process_series(processed_series, arima_prediction, filenames)
