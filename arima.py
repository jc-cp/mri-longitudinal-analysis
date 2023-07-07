import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA

from cfg.arima_cfg import LOADING_LIMIT, OUTPUT_DIR, PLOTS_DIR


# Look at the AIC, BIC, and HQIC metrics
class Arima_prediction:
    def __init__(self, loading_limit, image_directory):
        self.image_directory = image_directory
        self.loading_limit = LOADING_LIMIT
        self.images = []
        self.filenames = []

    def load_images(self):
        loaded_images = 0
        for filename in os.listdir(self.image_directory):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(self.image_directory, filename)).convert(
                    "L"
                )
                self.images.append(list(img.getdata()))
                print("Got filename", filename)
                self.filenames.append(
                    os.path.splitext(filename)[0]
                )  # Save the filename without extension
                loaded_images += 1
                if self.loading_limit and loaded_images >= self.loading_limit:
                    break

    def autocorrelation_plots(self):
        try:
            for i, image in enumerate(self.images):
                image_series = pd.Series(image)
                plt.figure()
                print("here")
                autocorrelation_plot(image_series)
                print("here")
                filename = self.filenames[i]
                print("Creating plot for file", filename)  # Move this line here
                plt.title(f"Autocorrelation plot for image {i+1}")
                plt.savefig(os.path.join(OUTPUT_DIR, filename))
                plt.close()
        except Exception as e:
            print("An error occurred:", str(e))

    def arima_prediction(self, p=5, d=1, q=0):
        for i, image in enumerate(self.images):
            image_series = pd.Series(image)
            model = ARIMA(image_series, order=(p, d, q))
            model_fit = model.fit(disp=0)
            print(f"ARIMA model summary for image {i+1}:\n{model_fit.summary()}")

            # plot residual errors
            residuals = pd.DataFrame(model_fit.resid)
            residuals.plot()
            plt.title(f"Residuals plot for image {i+1}")
            plt.show()
            residuals.plot(kind="kde")
            plt.title(f"Density plot for image {i+1}")
            plt.show()
            print(residuals.describe())


if __name__ == "__main__":
    # Usage
    image_analysis = Arima_prediction(LOADING_LIMIT, PLOTS_DIR)
    image_analysis.load_images()
    image_analysis.autocorrelation_plots()
    # image_analysis.arima_prediction(p=5, d=1, q=0)
