"""
Main script to initialize and run the VolumeEstimator.
    
This script initializes the VolumeEstimator with appropriate configuration settings,
processes provided segmentation files, visualizes the volume estimations, and 
exports the results to specified directories.
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from cfg import volume_est_cfg


class VolumeEstimator:
    """
    Class to estimate and visualize volumes based on segmentation files.

    This class allows for volume estimation based on provided segmentations,
    and it supports visualization and data export capabilities to aid analysis.
    If not in test mode, it also uses patient's Date of Birth (DoB) for more detailed analysis.

    Attributes:
        path (str): Path to the directory containing segmentation files.
        dob_df (pd.DataFrame): DataFrame containing Date of Birth information, if available.
        volumes (dict): Dictionary storing volume data for each patient.
    """

    def __init__(self, segmentations_path, dob_file):
        """
        Initialize the VolumeEstimator with given paths.

        Args:
            segmentations_path (str): Path to the directory containing segmentation files.
            dob_file (str): Path to the CSV file containing date of birth data.
        """
        self.path = segmentations_path

        if not volume_est_cfg.TEST_DATA:
            try:
                # Process the redacap .csv with clinical data
                self.dob_df = pd.read_csv(dob_file, sep=",", encoding="UTF-8")
                print(f"The length of the total csv dataset is: {len(self.dob_df)}")
                if len(self.dob_df) != 89:
                    print(
                        "Warning: The length of the filtered dataset is not 89. Check the csv"
                        " again."
                    )
                    sys.exit(1)
                self.dob_df["Date of Birth"] = pd.to_datetime(
                    self.dob_df["Date of Birth"], format="%d/%m/%y"
                )
                self.dob_df["BCH MRN"] = self.dob_df["BCH MRN"].astype(int)
            except FileNotFoundError as error:
                print(f"Error processing DOB file: {error}")
                sys.exit(1)

        self.volumes = defaultdict(list)

    @staticmethod
    def estimate_volume(segmentation_path):
        """
        Estimate the volume of the given segmentation file.

        Args:
            segmentation_path (str): Path to the segmentation file.

        Returns:
            float: Total volume of the segmentation.
        """
        try:
            segmentation = sitk.ReadImage(segmentation_path)
            voxel_spacing = segmentation.GetSpacing()
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            num_voxels = (segmentation_array > 0).sum()
            total_volume = num_voxels * voxel_volume
        except FileNotFoundError as error:
            print(f"Error estimating volume for {segmentation_path}: {error}")
            sys.exit(1)

        return total_volume

    def process_files(self, max_patients=None):
        """
        Process segmentation files, estimating volume for each one.

        Args:
            max_patients (int, optional): Maximum number of patients to process. Defaults to None.
        """
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        patient_ids = set()
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            patient_id, _ = file_name.split("_")[0], file_name.split("_")[1]  # scan_id irrelevant
            patient_ids.add(patient_id)

        if max_patients is not None and max_patients < len(patient_ids):
            patient_ids = list(patient_ids)[:max_patients]

        filtered_file_paths = [
            fp for fp in file_paths if os.path.basename(fp).split("_")[0] in patient_ids
        ]

        if not volume_est_cfg.TEST_DATA:
            filtered_df = self.dob_df[self.dob_df["BCH MRN"].astype(str).isin(patient_ids)]
            print(f"The length of the filtered dataset is: {len(filtered_df)}")

        with Pool(cpu_count()) as pool:
            results = pool.map(self.estimate_volume, filtered_file_paths)

        for file_path, volume in zip(filtered_file_paths, results):
            file_name = os.path.basename(file_path)
            patient_id, date_str = file_name.split("_")[0], file_name.split("_")[1]
            date_str = date_str.replace(".nii.gz", "")
            date = datetime.strptime(date_str, "%Y%m%d")

            if (
                not volume_est_cfg.TEST_DATA
                and patient_id in filtered_df["BCH MRN"].astype(str).values
            ):
                dob = self.dob_df.loc[
                    self.dob_df["BCH MRN"] == int(patient_id), "Date of Birth"
                ].iloc[0]
                age = (date - dob).days / 365.25
                self.volumes[patient_id].append((date, volume, age))
            else:
                self.volumes[patient_id].append((date, volume))

    def plot_with_poly_smoothing(self, patient_id, dates, volumes, ages=None):
        """
        Create a plot for the given patient data using polynomial smoothing.

        Args:
            patient_id (str): ID of the patient.
            dates (list): List of dates corresponding to volumes.
            volumes (list): List of volumes for each date.
            ages (list, optional): List of ages for each date. Defaults to None.

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot.
        """
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

        fig, a_x1 = self.setup_plot_base()

        a_x1.plot(interpolated_dates, interpolated_volumes_poly, color="tab:blue", marker="o")
        a_x1.set_xticks(dates)
        a_x1.set_xticklabels([dt.strftime("%m/%d/%Y") for dt in dates], rotation=90, fontsize=8)

        self.add_volume_change_to_plot(a_x1, dates, volumes)

        if ages:
            self.add_age_to_plot(a_x1, dates, ages)

        plt.title(f"Patient ID: {patient_id}")
        return fig

    def setup_plot_base(self):
        """
        Setup the base of a plot with the necessary properties.

        Returns:
            tuple: A tuple containing the figure and axis objects of the plot.
        """
        fig, a_x1 = plt.subplots(figsize=(12, 8))
        a_x1.set_xlabel("Scan Date")
        a_x1.set_ylabel("Volume (mm³)", color="tab:blue")
        a_x1.tick_params(axis="y", labelcolor="tab:blue")
        a_x1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        a_x1.xaxis.set_tick_params(pad=5)
        return fig, a_x1

    def add_volume_change_to_plot(self, a_x, dates, volumes):
        """
        Add volume change annotations to the plot.

        Args:
            a_x (matplotlib.axes.Axes): Axis object of the plot.
            dates (list): List of dates.
            volumes (list): List of volumes.
        """
        volume_changes = [0]
        for i, vol in enumerate(volumes[1:], 1):
            if volumes[i - 1] != 0:
                volume_change = ((vol - volumes[i - 1]) / volumes[i - 1]) * 100
            else:
                volume_change = np.nan
            volume_changes.append(volume_change)

        for i, (date, volume, volume_change) in enumerate(zip(dates, volumes, volume_changes)):
            a_x.text(
                date,
                volume,
                f"{volume_change:.2f}%",
                fontsize=8,
                va="bottom",
                ha="left",
            )

    def add_age_to_plot(self, a_x, dates, ages):
        """
        Add age annotations to the top of the plot.

        Args:
            a_x (matplotlib.axes.Axes): Axis object of the plot.
            dates (list): List of dates.
            ages (list): List of ages.
        """
        a_x2 = a_x.twiny()
        a_x2.xaxis.set_ticks_position("top")
        a_x2.xaxis.set_label_position("top")
        a_x2.set_xlabel("Patient Age (Years)")
        a_x2.set_xlim(a_x.get_xlim())
        date_nums = mdates.date2num(dates)
        a_x2.set_xticks(date_nums)
        a_x2.set_xticklabels([f"{age:.1f}" for age in ages])
        a_x2.xaxis.set_tick_params(labelsize=8)

    def plot_volumes(self, output_path):
        """
        Generate plots for each patient based on their volumes and save to the given directory.

        Args:
            output_path (str): Path to the directory where plots should be saved.
        """
        os.makedirs(output_path, exist_ok=True)

        for patient_id, volumes_data in self.volumes.items():
            volumes_data.sort(key=lambda x: x[0])  # sort by date

            if volume_est_cfg.POLY_SMOOTHING:
                if not volume_est_cfg.TEST_DATA:
                    dates, volumes, ages = zip(*volumes_data)  # unzip to three lists
                    fig = self.plot_with_poly_smoothing(patient_id, dates, volumes, ages)
                else:
                    dates, volumes = zip(*volumes_data)
                    fig = self.plot_with_poly_smoothing(patient_id, dates, volumes)
            else:
                fig, a_x1 = self.setup_plot_base()

                if not volume_est_cfg.TEST_DATA:
                    dates, volumes, ages = zip(*volumes_data)
                    a_x1.plot(dates, volumes, color="tab:blue", marker="o")
                    self.add_volume_change_to_plot(a_x1, dates, volumes)
                    self.add_age_to_plot(a_x1, dates, ages)
                else:
                    dates, volumes = zip(*volumes_data)
                    a_x1.plot(dates, volumes, color="tab:blue", marker="o")
                    self.add_volume_change_to_plot(a_x1, dates, volumes)

                plt.title(f"Patient ID: {patient_id}")

            fig.tight_layout()

            date_range = f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
            plt.savefig(os.path.join(output_path, f"volume_{patient_id}_{date_range}.png"))
            plt.close()

    def calculate_volume_change(self, previous, current):
        """
        Calculate the percentage change in volume from previous to current.

        Args:
            previous (float): Previous volume.
            current (float): Current volume.

        Returns:
            float: Percentage change in volume.
        """
        return ((current - previous) / previous) * 100 if previous != 0 else 0

    def generate_csv(self, output_folder):
        """
        Generate a CSV file for each patient containing their time series volume data.

        Args:
            output_folder (str): Path to the directory where CSV files should be saved.
        """

        # Ensure output directory exists, if not, create it.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for patient_id, volume_data in self.volumes.items():
            csv_file_path = os.path.join(output_folder, f"{patient_id}.csv")

            with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)

                # CSV header
                if not volume_est_cfg.TEST_DATA:
                    csv_writer.writerow(["Date", "Volume", "Age", "Growth[%]"])
                else:
                    csv_writer.writerow(["Date", "Volume", "Growth[%]"])

                # Sort data by date to ensure time series is in order
                sorted_volume_data = sorted(volume_data, key=lambda x: x[0])

                previous_volume = None
                for entry in sorted_volume_data:
                    if not volume_est_cfg.TEST_DATA:
                        date, volume, age = entry
                        if previous_volume:
                            percentage_growth = ((volume - previous_volume) / previous_volume) * 100
                        else:
                            percentage_growth = 0  # No growth for the first data point
                        csv_writer.writerow(
                            [date.strftime("%Y-%m-%d"), volume, age, percentage_growth]
                        )
                        previous_volume = volume

                    else:
                        date, volume = entry
                        if previous_volume:
                            percentage_growth = ((volume - previous_volume) / previous_volume) * 100
                        else:
                            percentage_growth = 0  # No growth for the first data point
                        csv_writer.writerow([date.strftime("%Y-%m-%d"), volume, percentage_growth])
                        previous_volume = volume


if __name__ == "__main__":
    ve = VolumeEstimator(volume_est_cfg.SEG_DIR, volume_est_cfg.REDCAP_FILE)
    print("Volume Estimator initialized.")
    print("Getting prediction masks.")
    ve.process_files(max_patients=volume_est_cfg.LIMIT_LOADING)
    print("Saving data.")
    ve.plot_volumes(output_path=volume_est_cfg.PLOTS_DIR)
    print("Generating time-series csv's.")
    ve.generate_csv(output_folder=volume_est_cfg.CSV_DIR)
