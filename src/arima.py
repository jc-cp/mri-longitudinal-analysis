import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller, pacf
import numpy as np


from cfg.arima_cfg import (
    LOADING_LIMIT,
    OUTPUT_DIR,
    PLOTS_DIR,
    FROM_IMAGES,
    FROM_DATA,
    TIME_SERIES_DIR,
)

# Constants
PLOT_TYPE_AUTO = "autocorrelation"
PLOT_TYPE_PACF = "partial_autocorrelation"
PLOT_TYPE_RESIDUALS = "residuals"
PLOT_TYPE_DENSITY = "density"


# Look at the AIC, BIC, and HQIC metrics
class Arima_prediction:
    def __init__(self, loading_limit, image_directory, time_series_directory):
        self.image_directory = image_directory
        self.loading_limit = loading_limit
        self.time_series_directory = time_series_directory
        self.time_series_list = (
            self._load_time_series() if time_series_directory else []
        )
        self.images = []
        self.filenames = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _process_series(self, series_list, source):
        """
        Main method to handle series for both images and csv data.
        :param series_list: List of series data.
        :param source: Source type (from_image or from_csv).
        """
        for ts_data in series_list:
            self._generate_plot(ts_data, PLOT_TYPE_AUTO, source)
            self._generate_plot(ts_data, PLOT_TYPE_PACF, source)
            self.dickey_fuller_test(data=ts_data, source=source)
            self.arima_prediction(data=ts_data, source=source)

    def _load_images(self):
        loaded_images = 0
        for filename in os.listdir(self.image_directory):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(self.image_directory, filename)).convert(
                    "L"
                )
                self.images.append(list(img.getdata()))
                print("Got filename", filename)
                self.filenames.append(os.path.splitext(filename)[0])
                loaded_images += 1
                if self.loading_limit and loaded_images >= self.loading_limit:
                    break

    def _load_time_series(self):
        time_series_list = []

        # Check if the path is a directory
        if os.path.isdir(self.time_series_directory):
            for filename in os.listdir(self.time_series_directory):
                if filename.endswith(".csv"):
                    filepath = os.path.join(self.time_series_directory, filename)
                    ts_data = pd.read_csv(
                        filepath, usecols=[0, 1], parse_dates=[0], index_col=0
                    )
                    time_series_list.append(ts_data.squeeze())

        # Check if the path is a file
        elif os.path.isfile(
            self.time_series_directory
        ) and self.time_series_directory.endswith(".csv"):
            ts_data = pd.read_csv(
                self.time_series_directory, usecols=[0, 1], parse_dates=[0], index_col=0
            )
            time_series_list.append(ts_data.squeeze())

        return time_series_list

    def _generate_plot(self, data, plot_type, source):
        filename = self._get_filename(data, source)
        plt.figure(figsize=(10, 6))

        if plot_type == PLOT_TYPE_AUTO:
            autocorrelation_plot(data)
            print(f"Performing autocorrelation plot for patient {filename}.")
            plt.title(f"Autocorrelation Plot for {filename}")
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")

        elif plot_type == PLOT_TYPE_PACF:
            plot_pacf(data)
            print(f"Performing partial autocorrelation plot for patient {filename}.")
            plt.title(f"Partial Autocorrelation plot for {filename}")

        elif plot_type == PLOT_TYPE_RESIDUALS:
            residuals = pd.Series(data)
            residuals.plot(title=f"Residuals Plot for {filename}")
            plt.xlabel("Date/Time")
            plt.ylabel("Residuals")

        elif plot_type == PLOT_TYPE_DENSITY:
            residuals = pd.Series(data)
            residuals.plot(kind="kde", title=f"Density Plot for {filename}")
            plt.xlabel("Residual Value")
            plt.ylabel("Density")

        plt.legend([f"{plot_type.capitalize()} of {filename}"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_{source}_{plot_type}.png"))
        plt.close()

    def _get_filename(self, data, source):
        # Retrieve filename based on data and source
        if source == "from_image":
            return self.filenames[self.images.index(data)]
        elif source == "from_csv":
            return "time_series_data_{}".format(self.time_series_list.index(data))
        else:
            raise ValueError(f"Unknown source type: {source}")

    def arima_prediction(
        self, p=None, d=1, q=0, from_image=True, data=None, forecast_steps=10
    ):  # , source):
        if from_image:
            series_list = [pd.Series(img) for img in self.images]
            suffix = "_from_image"

        else:
            series_list = [data]
            suffix = "_from_csv"

        try:
            for i, data in enumerate(series_list):
                # Make series stationary and get the differencing order
                stationary_data, d = self.make_series_stationary(data)

                # If p isn't provided, determine it using PACF
                if p is None:
                    p = self.determine_p_from_pacf(stationary_data)

                # Get adaptive forecast steps
                forecast_steps = self.get_adaptive_forecast_steps(stationary_data)

                model = ARIMA(stationary_data, order=(p, d, q))
                model_fit = model.fit()

                # Display AIC and BIC metrics
                print(f"ARIMA AIC for image {i+1}: {model_fit.aic}")
                print(f"ARIMA BIC for image {i+1}: {model_fit.bic}")
                print(f"ARIMA HQIC for image {i+1}: {model_fit.hqic}")
                print(f"ARIMA model summary for image {i+1}:\n{model_fit.summary()}")

                # Forecast next `forecast_steps` points
                forecast = model_fit.forecast(steps=forecast_steps)
                forecast_series = pd.Series(forecast, name="Predictions")

                # Save the forecast to a CSV file
                filename = (
                    self.filenames[i] if from_image else "time_series_data_{}".format(i)
                )
                forecast_series.to_csv(
                    os.path.join(OUTPUT_DIR, f"{filename}{suffix}_forecast.csv"),
                    index=False,
                )

                # plot residual errors
                self._generate_plot(model_fit.resid, PLOT_TYPE_RESIDUALS, source)
                self._generate_plot(model_fit.resid, PLOT_TYPE_DENSITY, source)

                # optional description
                residuals = model_fit.resid
                print(residuals.describe())

        except Exception as e:
            print("An error occurred:", str(e))

    def dickey_fuller_test(self, from_image=True, data=None):
        if from_image:
            series_list = [pd.Series(img) for img in self.images]
            suffix = "_from_image"
        else:
            series_list = [data]
            suffix = "_from_csv"

        print("Performing Dickey Fuller test!")

        for i, data in enumerate(series_list):
            # Augmented Dickey-Fuller test
            result = adfuller(data)
            filename = (
                self.filenames[i] if from_image else "time_series_data_{}".format(i)
            )

            with open(
                os.path.join(OUTPUT_DIR, f"{filename}{suffix}_adf_test.txt"), "w"
            ) as file:
                file.write(f"ADF Statistic for image {i+1}: {result[0]}\n")
                file.write(f"p-value for image {i+1}: {result[1]}\n")
                for key, value in result[4].items():
                    file.write(f"Critical Value ({key}) for image {i+1}: {value}\n")

            print(f"ADF Statistic for image {i+1}: {result[0]}")
            print(f"p-value for image {i+1}: {result[1]}")
            for key, value in result[4].items():
                print(f"Critical Value ({key}) for image {i+1}: {value}")

    def make_series_stationary(self, data, max_diff=3):
        """
        Returns the differenced series until it becomes stationary or reaches max allowed differencing.
        """
        d = 0  # Track differencing order
        p_value = 1
        result = adfuller(data)

        while p_value >= 0.05 and d < max_diff:
            data = data.diff().dropna()
            result = adfuller(data)
            p_value = result[1]
            d += 1

        return data, d

    def get_adaptive_forecast_steps(self, data):
        """
        Returns the forecast frequency based on the size of the dataset.
        """
        n = len(data)
        # Forecast proportionally based on data length.
        # This takes 5% of data length as forecast steps. Adjust as needed.
        return max(1, int(n * 0.05))

    def determine_p_from_pacf(self, data, alpha=0.05):
        """
        Returns the optimal p value for ARIMA based on PACF.
        """
        # Threshold for significance
        threshold = 1.96 / np.sqrt(len(data))

        pacf_vals = pacf(data)

        # Find where PACF values are greater than the threshold
        significant_lags = np.where(np.abs(pacf_vals) > threshold)[0]

        if significant_lags.any():
            return significant_lags[-1]
        return 1  # Default to 1 if none are significant


if __name__ == "__main__":
    image_analysis = Arima_prediction(LOADING_LIMIT, PLOTS_DIR, TIME_SERIES_DIR)

    if FROM_IMAGES:
        image_analysis._load_images()
        image_analysis._process_series(image_analysis.images, "from_image")

    if FROM_DATA:
        image_analysis._process_series(image_analysis.time_series_list, "from_csv")
