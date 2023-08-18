import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller


from cfg.arima_cfg import LOADING_LIMIT, OUTPUT_DIR, PLOTS_DIR, FROM_IMAGES, FROM_DATA, TIME_SERIES_FILE


# Look at the AIC, BIC, and HQIC metrics
class Arima_prediction:
    def __init__(self, loading_limit, image_directory, time_series):
        self.image_directory = image_directory
        self.loading_limit = LOADING_LIMIT
        self.time_series = time_series
        self.images = []
        self.filenames = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                # Regular autocorrelation plot
                plt.figure()
                autocorrelation_plot(image_series)
                filename = self.filenames[i]
                print(f"Performing autocorrelation plot for patient {filename}.")
                plt.title(f"Autocorrelation plot for image {i+1}")
                plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_autocorrelation.png"))
                plt.close()
                
                # Partial autocorrelation plot
                plt.figure()
                plot_pacf(image_series)
                print(f"Performing partial autocorrelation plot for patient {filename}.")
                plt.title(f"Partial Autocorrelation plot for image {i+1}")
                plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_partial_autocorrelation.png"))
                plt.close()
        except Exception as e:
            print("An error occurred:", str(e))

    def arima_prediction(self, p=5, d=1, q=0):
        for i, image in enumerate(self.images):
            image_series = pd.Series(image)
            model = ARIMA(image_series, order=(p, d, q))
            model_fit = model.fit(disp=0)
            
            # Display AIC and BIC metrics
            print(f"ARIMA AIC for image {i+1}: {model_fit.aic}")
            print(f"ARIMA BIC for image {i+1}: {model_fit.bic}")
            print(f"ARIMA HQIC for image {i+1}: {model_fit.hqic}")
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

    def dickey_fuller_test(self):
        print("Performing Dickey Fuller test!")
        for i, image in enumerate(self.images):
            image_series = pd.Series(image)
            
            # Augmented Dickey-Fuller test
            result = adfuller(image_series)
            print(f"ADF Statistic for image {i+1}: {result[0]}")
            print(f"p-value for image {i+1}: {result[1]}")
            for key, value in result[4].items():
                print(f"Critical Value ({key}) for image {i+1}: {value}")


if __name__ == "__main__":
    
    time_series_data = pd.read_csv(TIME_SERIES_FILE, index_col=0, squeeze=True)
    image_analysis = Arima_prediction(LOADING_LIMIT, PLOTS_DIR, TIME_SERIES_FILE)
    
    if FROM_IMAGES:
        image_analysis.load_images()
        image_analysis.autocorrelation_plots()
    
    if FROM_DATA:
        image_analysis.dickey_fuller_test()
        # image_analysis.arima_prediction(p=5, d=1, q=0)
