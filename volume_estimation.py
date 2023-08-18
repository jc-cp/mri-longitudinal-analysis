import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from PIL import Image
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import SimpleITK as sitk
import csv

from cfg.volume_est_cfg import (LIMIT_LOADING, PLOTS_DIR, POLY_SMOOTHING,
                                REDCAP_FILE, SEG_DIR, TEST_DATA, CSV_DIR)


class VolumeEstimator:
    def __init__(self, segmentations_path, dob_file):
        self.path = segmentations_path

        if not TEST_DATA:
            try:
                # Process the redacap .csv with clinical data
                self.dob_df = pd.read_csv(dob_file, sep=",", encoding="UTF-8")
                self.dob_df = self.dob_df[self.dob_df["no ops cohort"] == "NAN"]
                print(f"The length of the total csv dataset is: {len(self.dob_df)}")
                if len(self.dob_df) != 89:
                    print("Warning: The length of the filtered dataset is not 60")
                self.dob_df["Date of Birth"] = pd.to_datetime(
                    self.dob_df["Date of Birth"], dayfirst=True
                )
                self.dob_df["BCH MRN"] = self.dob_df["BCH MRN"].astype(int)
            except Exception as e:
                print(f"Error processing DOB file: {e}")

        self.volumes = defaultdict(list)

    @staticmethod
    def estimate_volume(segmentation_path):
        try:
            segmentation = sitk.ReadImage(segmentation_path)
            voxel_spacing = segmentation.GetSpacing()
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            num_voxels = (segmentation_array > 0).sum()
            total_volume = num_voxels * voxel_volume
        except Exception as e:
            print(f"Error estimating volume for {segmentation_path}: {e}")
            return None

        return total_volume

    def process_files(self, max_patients=None):
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        patient_ids = set()
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            patient_id, scan_id = file_name.split("_")[0], file_name.split("_")[1]
            patient_ids.add(patient_id)

        if max_patients is not None and max_patients < len(patient_ids):
            patient_ids = list(patient_ids)[:max_patients]

        filtered_file_paths = [
            fp for fp in file_paths if os.path.basename(fp).split("_")[0] in patient_ids
        ]

        if not TEST_DATA:
            filtered_df = self.dob_df[self.dob_df["BCH MRN"].astype(str).isin(patient_ids)]
            print(f"The length of the filtered dataset is: {len(filtered_df)}")

        with Pool(cpu_count()) as p:
            results = p.map(self.estimate_volume, filtered_file_paths)

        for file_path, volume in zip(filtered_file_paths, results):
            file_name = os.path.basename(file_path)
            patient_id, date_str = file_name.split("_")[0], file_name.split("_")[1]
            date_str = date_str.replace(".nii.gz", "")
            date = datetime.strptime(date_str, "%Y%m%d")

            if not TEST_DATA and patient_id in filtered_df["BCH MRN"].astype(str).values:
                dob = self.dob_df.loc[
                    self.dob_df["BCH MRN"] == int(patient_id), "Date of Birth"
                ].iloc[0]
                age = (date - dob).days / 365.25
                self.volumes[patient_id].append((date, volume, age))
            else:
                self.volumes[patient_id].append((date, volume))
    
    def plot_with_poly_smoothing(self, patient_id, dates, volumes, ages=None):
        # Polynomial interpolation
        poly_degree = 5  # Degree of the polynomial
        poly_coeff = np.polyfit(mdates.date2num(dates), volumes, poly_degree)
        poly_interp = np.poly1d(poly_coeff)

        # Generate interpolated values
        num_points = 50  # Number of points for interpolation
        start = mdates.date2num(min(dates))
        end = mdates.date2num(max(dates))
        interpolated_dates = mdates.num2date(np.linspace(start, end, num_points))
        interpolated_volumes_poly = poly_interp(mdates.date2num(interpolated_dates))

        fig, ax1 = self.setup_plot_base()

        ax1.plot(interpolated_dates, interpolated_volumes_poly, color="tab:blue", marker="o")
        ax1.set_xticks(dates)
        ax1.set_xticklabels([dt.strftime("%m/%d/%Y") for dt in dates], rotation=90, fontsize=8)

        self.add_volume_change_to_plot(ax1, dates, volumes)

        if ages:
            self.add_age_to_plot(ax1, dates, ages)

        plt.title(f"Patient ID: {patient_id}")
        return fig

    def setup_plot_base(self):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel("Scan Date")
        ax1.set_ylabel("Volume (mm³)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        ax1.xaxis.set_tick_params(pad=5)
        return fig, ax1

    def add_volume_change_to_plot(self, ax, dates, volumes):
        volume_changes = [0]
        for i, v in enumerate(volumes[1:], 1):
            if volumes[i - 1] != 0:
                volume_change = ((v - volumes[i - 1]) / volumes[i - 1]) * 100
            else:
                volume_change = np.nan
            volume_changes.append(volume_change)

        for i, (date, volume, volume_change) in enumerate(zip(dates, volumes, volume_changes)):
            ax.text(date, volume, f"{volume_change:.2f}%", fontsize=8, va="bottom", ha="left")

    def add_age_to_plot(self, ax, dates, ages):
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position("top")
        ax2.xaxis.set_label_position("top")
        ax2.set_xlabel("Patient Age (Years)")
        ax2.set_xlim(ax.get_xlim())
        date_nums = mdates.date2num(dates)
        ax2.set_xticks(date_nums)
        ax2.set_xticklabels([f"{age:.1f}" for age in ages])
        ax2.xaxis.set_tick_params(labelsize=8)

    def plot_volumes(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        for patient_id, volumes_data in self.volumes.items():
            volumes_data.sort(key=lambda x: x[0])  # sort by date
            
            if POLY_SMOOTHING:
                if not TEST_DATA:
                    dates, volumes, ages = zip(*volumes_data)  # unzip to three lists
                    fig = self.plot_with_poly_smoothing(patient_id, dates, volumes, ages)
                else:
                    dates, volumes = zip(*volumes_data)
                    fig = self.plot_with_poly_smoothing(patient_id, dates, volumes)
            else:
                fig, ax1 = self.setup_plot_base()
                
                if not TEST_DATA:
                    dates, volumes, ages = zip(*volumes_data)
                    ax1.plot(dates, volumes, color="tab:blue", marker="o")
                    self.add_volume_change_to_plot(ax1, dates, volumes)
                    self.add_age_to_plot(ax1, dates, ages)
                else:
                    dates, volumes = zip(*volumes_data)
                    ax1.plot(dates, volumes, color="tab:blue", marker="o")
                    self.add_volume_change_to_plot(ax1, dates, volumes)

                plt.title(f"Patient ID: {patient_id}")

            fig.tight_layout()

            date_range = f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
            plt.savefig(os.path.join(output_path, f"volume_{patient_id}_{date_range}.png"))
            plt.close()

    def calculate_volume_change(self, previous, current):
        return ((current - previous) / previous) * 100 if previous != 0 else 0

    def generate_csv(self, output_folder):
        """
        Generate .csv files for each patient with time series volume data.
        
        Parameters:
            - output_folder: Path to the folder where the .csv files should be saved.
        """

        # Ensure output directory exists, if not, create it.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for patient_id, volume_data in self.volumes.items():
            csv_file_path = os.path.join(output_folder, f"{patient_id}.csv")
            
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # CSV header
                if not TEST_DATA:
                    csv_writer.writerow(["Date", "Volume", "Age", "Growth[%]"])
                else:
                    csv_writer.writerow(["Date", "Volume", "Growth[%]"]) 

                # Sort data by date to ensure time series is in order
                sorted_volume_data = sorted(volume_data, key=lambda x: x[0])

                previous_volume = None
                for entry in sorted_volume_data:
                    if not TEST_DATA:
                        date, volume, age = entry
                        if previous_volume:
                            percentage_growth = ((volume - previous_volume) / previous_volume) * 100
                        else:
                            percentage_growth = 0  # No growth for the first data point
                        csv_writer.writerow([date.strftime('%Y-%m-%d'), volume, age, percentage_growth])
                        previous_volume = volume

                    else:
                        date, volume = entry
                        if previous_volume:
                            percentage_growth = ((volume - previous_volume) / previous_volume) * 100
                        else:
                            percentage_growth = 0  # No growth for the first data point
                        csv_writer.writerow([date.strftime('%Y-%m-%d'), volume, percentage_growth])
                        previous_volume = volume

if __name__ == "__main__":
    ve = VolumeEstimator(SEG_DIR, REDCAP_FILE)
    print("Getting prediction masks.")
    ve.process_files(max_patients=LIMIT_LOADING)
    print("Saving data.")
    ve.plot_volumes(output_path=PLOTS_DIR)
    print("Generating time-series csv's.")
    ve.generate_csv(output_folder=CSV_DIR)
