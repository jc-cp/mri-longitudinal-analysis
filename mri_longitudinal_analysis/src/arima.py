"""
This script provides functionality for ARIMA-based time series prediction.
It supports loading data from both images and CSV files.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import arima_cfg
from pandas.plotting import autocorrelation_plot
from PIL import Image
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf


class DataHandler:
    """Abstract class for both loading DataHandler child classes."""

    def __init__(self, directory, loading_limit):
        self.directory = directory
        self.loading_limit = loading_limit

    def load_data(self):
        """Load data from the source. Should be implemented by subclasses."""
        raise NotImplementedError

    def process_series(self, series_list, arima_pred, file_names):
        """
        Main method to handle series for both images and csv data.
        :param series_list: List of series data.
        :param arima_pred: Constructor of the class
        :param file_names: List of filenames
        """
        for idx, ts_data in enumerate(series_list):
            arima_pred.generate_plot(ts_data, "autocorrelation", file_names[idx])
            arima_pred.generate_plot(ts_data, "partial_autocorrelation", file_names[idx])
            arima_pred.perform_dickey_fuller_test(data=ts_data)
            arima_pred.arima_prediction(data=ts_data)

    def load_data_generator(self):
        """Generate data on-the-fly for memory efficiency. Should be implemented by subclasses."""
        raise NotImplementedError


class ImageDataHandler(DataHandler):
    """Loads image data and processes it for the prediction."""

    def load_data(self):
        """Loads images from the specified directory up to the loading limit."""
        imgs = []
        file_names = []
        try:
            loaded_images = 0
            for filename in os.listdir(self.directory):
                if filename.endswith(".png"):
                    img = Image.open(os.path.join(self.directory, filename)).convert("L")
                    imgs.append(list(img.getdata()))
                    file_names.append(os.path.splitext(filename)[0])
                    loaded_images += 1
                    if self.loading_limit and loaded_images >= self.loading_limit:
                        break
            return imgs, file_names
        except (FileNotFoundError, IOError) as error:
            print(f"Error loading images: {error}")
            return [], []

    def load_data_generator(self):
        """Generate image data on-the-fly for memory efficiency."""
        loaded_images = 0
        for filename in os.listdir(self.directory):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(self.directory, filename)).convert("L")
                yield list(img.getdata()), os.path.splitext(filename)[0]
                loaded_images += 1
                if self.loading_limit and loaded_images >= self.loading_limit:
                    break


class TimeSeriesDataHandler(DataHandler):
    """Loads time-series data and processes it for the prediction."""

    def load_data(self):
        """
        Loads time series data either from a directory or from a specified file.

        Returns:
        - list: A list of loaded time series data.
        """
        time_series_list = []
        file_names = []
        try:
            if os.path.isdir(self.directory):
                for filename in os.listdir(self.directory):
                    if filename.endswith(".csv"):
                        filepath = os.path.join(self.directory, filename)
                        ts_data = pd.read_csv(
                            filepath, usecols=[0, 1], parse_dates=[0], index_col=0
                        )
                        time_series_list.append(ts_data.squeeze())
                        file_names.append(os.path.splitext(filename)[0])
            elif os.path.isfile(self.directory) and self.directory.endswith(".csv"):
                ts_data = pd.read_csv(self.directory, usecols=[0, 1], parse_dates=[0], index_col=0)
                time_series_list.append(ts_data.squeeze())
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


class ArimaPrediction:
    """
    A class to handle ARIMA-based time series prediction.
    """

    PLOT_TYPES = {
        "autocorrelation": {
            "function": autocorrelation_plot,
            "title": "Autocorrelation Plot",
            "xlabel": "Lag",
            "ylabel": "Autocorrelation",
        },
        "partial_autocorrelation": {
            "function": plot_pacf,
            "title": "Partial Autocorrelation Plot",
            "xlabel": "Lag",
            "ylabel": "Partial Autocorrelation",
        },
        "residuals": {
            "function": lambda data: pd.Series(
                data
            ).plot(),  # Use lambda to adapt the plot() method
            "title": "Residuals Plot",
            "xlabel": "Date/Time",
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

    def __init__(self):
        """
        Constructor for the Arima_prediction class.
        """
        self.images = []
        self.filenames = []
        os.makedirs(arima_cfg.OUTPUT_DIR, exist_ok=True)

    def generate_plot(self, data, plot_type, filename):
        """
        Generates and saves various plots based on the provided data and plot type.

        Parameters:
        - data (Series): Time series data.
        - plot_type (str): Type of the plot to generate.
        - filename (str): Name of the data file.
        """
        plt.figure(figsize=(10, 6))

        plot_func = self.PLOT_TYPES[plot_type]["function"]
        plot_func(data)

        plt.title(self.PLOT_TYPES[plot_type]["title"] + f" for {filename}")
        plt.xlabel(self.PLOT_TYPES[plot_type]["xlabel"])
        plt.ylabel(self.PLOT_TYPES[plot_type]["ylabel"])

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{filename}_{plot_type}.png"))
        plt.close()

    def arima_prediction(
        self,
        data,
        filename,
        p_value=None,
        d_value=None,
        q_value=None,
        forecast_steps=10,
        p_range=range(5),
        q_range=range(5),
    ):
        """Actual arima prediction method. Gets the corresponding
        p,d,q values from analysis and performs a prediction."""

        suffix = "from_image" if "image" in filename else "from_csv"

        # Make series stationary and get the differencing d_value
        stationary_data, d_value = self._make_series_stationary(data)

        # Get p_value from partial correlation
        if p_value is None:
            suggested_p_value = self._determine_p_from_pacf(stationary_data)
            p_range = range(max(0, suggested_p_value - 2), suggested_p_value + 3)

        if p_value is None or d_value is None or q_value is None:
            p_value, d_value, q_value = self.find_best_arima_order(
                stationary_data, p_range, d_value, q_range
            )

        try:
            # Get adaptive forecast steps
            forecast_steps = self._get_adaptive_forecast_steps(stationary_data)

            model = ARIMA(stationary_data, order=(p_value, d_value, q_value))
            model_fit = model.fit()

            # Display AIC and BIC metrics
            print(f"ARIMA AIC for image {filename}: {model_fit.aic}")
            print(f"ARIMA BIC for image {filename}: {model_fit.bic}")
            print(f"ARIMA HQIC for image {filename}: {model_fit.hqic}")
            print(f"ARIMA model summary for image {filename}:\n{model_fit.summary()}")

            # Forecast next `forecast_steps` points
            forecast, _, conf_int = model_fit.forecast(steps=forecast_steps)
            self._diagnostics(model_fit, filename)

            forecast = ArimaPrediction.invert_differencing(data, forecast, d_value)
            forecast_series = pd.Series(forecast, name="Predictions")

            # Save forecasts
            self._save_forecast_fig(data, forecast, forecast_steps, conf_int, filename)
            self._save_forecast_csv(forecast_series, filename, suffix)

            # plot residual errors
            self.generate_plot(model_fit.resid, "residuals", filename)
            self.generate_plot(model_fit.resid, "density", filename)

            # optional description
            residuals = model_fit.resid
            print(residuals.describe())

        except IOError as error:
            print("An error occurred:", str(error))

    def _save_forecast_fig(
        self,
        data,
        forecast,
        forecast_steps,
        conf_int,
        filename,
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
        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{filename}_forecast_plot.png"))
        plt.close()

    def _save_forecast_csv(self, forecast_series, filename, suffix):
        """Save the forecast to a .csv file."""
        forecast_series.to_csv(
            os.path.join(arima_cfg.OUTPUT_DIR, f"{filename}{suffix}_forecast.csv"),
            index=False,
        )

    def _diagnostics(self, model_fit, filename):
        """Saves the diagnostics plot."""
        model_fit.plot_diagnostics(figsize=(12, 8), legend=True)
        plt.savefig(os.path.join(arima_cfg.OUTPUT_DIR, f"{filename}_diagnostics_plot.png"))
        plt.close()

    def perform_dickey_fuller_test(self, data, filename):
        """Performing Dickey Fuller test to see the stationarity of series."""

        suffix = "from_image" if "image" in filename else "from_csv"

        # Augmented Dickey-Fuller test
        result = adfuller(data)
        with open(
            os.path.join(arima_cfg.OUTPUT_DIR, f"{filename}{suffix}_adf_test.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"ADF Statistic for image {filename}: {result[0]}\n")
            file.write(f"p-value for image {filename}: {result[1]}\n")
            for key, value in result[4].items():
                file.write(f"Critical Value ({key}) for image {filename}: {value}\n")

        print(f"ADF Statistic for image {filename}: {result[0]}")
        print(f"p-value for image {filename}: {result[1]}")
        for key, value in result[4].items():
            print(f"Critical Value ({key}) for image {filename}: {value}")

    def _make_series_stationary(self, data, max_diff=3):
        """
        Returns the differenced series until it becomes stationary or reaches max
        allowed differencing.
        """
        d_value = 0  # Track differencing order
        p_value = 1
        result = adfuller(data)

        while p_value >= 0.05 and d_value < max_diff:
            data = data.diff().dropna()
            result = adfuller(data)
            p_value = result[1]
            d_value += 1

        return data, d_value

    def _get_adaptive_forecast_steps(self, data):
        """
        Returns the forecast frequency based on the size of the dataset.
        """
        n_steps = len(data)
        # Forecast proportionally based on data length.
        # This takes 5% of data length as forecast steps. Adjust as needed.
        return max(1, int(n_steps * 0.05))

    def _determine_p_from_pacf(self, data, alpha=0.05):
        """
        Returns the optimal p value for ARIMA based on PACF.
        Parameters:
        - data: Time series data.
        - alpha: Significance level for PACF.

        Returns:
        - p value
        """

        pacf_vals, confint = pacf(data, alpha=alpha, nlags=len(data) - 1)
        significant_lags = np.where(pacf_vals > confint[:, 1]) or np.where(
            pacf_vals < confint[:, 0]
        )

        if significant_lags[0].any():
            return significant_lags[0][-1]
        return 1  # Default to 1 if none are significant

    def find_best_arima_order(self, stationary_data, p_range, d_value, q_range):
        """
        Determine the best ARIMA order based on AIC, BIC, and HQIC criteria.

        Parameters:
        - data (pd.Series): The time series data for which the ARIMA order needs to be determined.
        - p_range (range): The range of values for the ARIMA 'p' parameter to be tested.
        - d_range (range): The range of values for the ARIMA 'd' parameter to be tested.
        - q_range (range): The range of values for the ARIMA 'q' parameter to be tested.

        Returns:
        - tuple: The optimal (p, d, q) order for the ARIMA model.

        Note:
        The function attempts to fit an ARIMA model for each combination of p, d, q values provided.
        The combination that minimizes the AIC, BIC, and HQIC values is selected as the best order.
        """
        best_aic, best_bic, best_hqic = float("inf"), float("inf"), float("inf")
        best_order = None

        for p_value in p_range:
            for q_value in q_range:
                try:
                    model = ARIMA(stationary_data, order=(p_value, d_value, q_value))
                    model_fit = model.fit()

                    if (
                        (model_fit.aic < best_aic)
                        and (model_fit.bic < best_bic)
                        and (model_fit.hqic < best_hqic)
                    ):
                        best_aic, best_bic, best_hqic = (
                            model_fit.aic,
                            model_fit.bic,
                            model_fit.hqic,
                        )
                        best_order = (p_value, d_value, q_value)

                        # [FIXME]:
                        # avg_score = (model_fit.aic + model_fit.bic + model_fit.hqic) / 3
                        # if avg_score < (best_aic + best_bic + best_hqic) / 3:
                        #     best_aic = model_fit.aic
                        #     best_bic = model_fit.bic
                        #     best_hqic = model_fit.hqic
                        #     best_order = (p_value, d_value, q_value)
                except (MemoryError, ModuleNotFoundError, InterruptedError) as error:
                    print(error)
                    continue

        return best_order

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


if __name__ == "__main__":
    arima_prediction = ArimaPrediction()

    if arima_cfg.FROM_IMAGES:
        image_handler = ImageDataHandler(arima_cfg.PLOTS_DIR, arima_cfg.LOADING_LIMIT)
        images, filenames = image_handler.load_data()
        image_handler.process_series(images, arima_prediction, filenames)

    if arima_cfg.FROM_DATA:
        ts_handler = TimeSeriesDataHandler(arima_cfg.TIME_SERIES_DIR, arima_cfg.LOADING_LIMIT)
        ts_data_list = ts_handler.load_data()
        ts_handler.process_series(ts_data_list, arima_prediction, filenames)
