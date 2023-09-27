"""
Main script to initialize and run the VolumeEstimator.
    
This script initializes the VolumeEstimator with appropriate configuration settings,
processes provided segmentation files, visualizes the volume estimations, and 
exports the results to specified directories.
"""
import csv
import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import norm
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
        self.raw_data = defaultdict(list)
        self.filtered_data = defaultdict(list)
        self.poly_smoothing_data = defaultdict(list)
        self.kernel_smoothing_data = defaultdict(list)
        self.data_sources = {
            "raw": {},
            "filtered": {},
            "poly_smoothing": {},
            "kernel_smoothing": {},
        }

        if not volume_est_cfg.TEST_DATA:
            try:
                # Process the redacap .csv with clinical data
                self.dob_df = pd.read_csv(dob_file, sep=",", encoding="UTF-8")
                print(f"The length of the total csv dataset is: {len(self.dob_df)}")
                if len(self.dob_df) != volume_est_cfg.NUMBER_TOTAL_PATIENTS:
                    raise ValueError(
                        "Warning: The length of the filtered dataset is not"
                        f" {volume_est_cfg.NUMBER_TOTAL_PATIENTS}. Check the csv again."
                    )
                self.dob_df["Date of Birth"] = pd.to_datetime(
                    self.dob_df["Date of Birth"], format="%d/%m/%y"
                )
                self.dob_df["BCH MRN"] = self.dob_df["BCH MRN"].astype(int)
            except FileNotFoundError as error:
                print(f"Error processing DOB file: {error}")

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

        return total_volume

    @staticmethod
    def calculate_volume_change(previous, current):
        """
        Calculate the percentage change in volume from previous to current.

        Args:
            previous (float): Previous volume.
            current (float): Current volume.

        Returns:
            float: Percentage change in volume.
        """
        return ((current - previous) / previous) * 100 if previous != 0 else 0

    def process_files(self, max_patients=None):
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        all_ids = defaultdict(list)
        all_scans = defaultdict(list)
        filtered_scans = defaultdict(list)

        # Calculate volumes and filter out NaN ones first
        with Pool(cpu_count()) as pool:
            for file_path, volume in zip(file_paths, pool.map(self.estimate_volume, file_paths)):
                patient_id, scan_id = os.path.basename(file_path).split("_")[:2]

                # take into account the ones for filtering
                all_ids[patient_id].append(scan_id)
                all_scans[patient_id].append((file_path, volume))
                if volume != 0:
                    filtered_scans[patient_id].append((file_path, volume))

        self.raw_data = self.process_scans(all_scans)
        self.data_sources["raw"] = self.raw_data

        if max_patients is not None and max_patients < len(self.raw_data):
            self.raw_data = dict(list(self.raw_data.items())[:max_patients])

        print(f"Set the max number of patient to be loaded to: {max_patients}")
        print(f"The length of the filtered dataset is: {len(self.raw_data)}")

        # Additional logic to process and store other states of data
        # (filtered, poly-smoothed, kernel-smoothed)
        if volume_est_cfg.FILTERED:
            self.filtered_data = self.apply_filtering(all_ids, filtered_scans)
            self.data_sources["filtered"] = self.filtered_data
        if volume_est_cfg.POLY_SMOOTHING:
            self.poly_smoothing_data = self.apply_polysmoothing(
                poly_degree=volume_est_cfg.POLY_SMOOTHING_DEGREE
            )
            self.data_sources["poly_smoothing"] = self.poly_smoothing_data
        if volume_est_cfg.KERNEL_SMOOTHING:
            self.kernel_smoothing_data = self.apply_kernel_smoothing(
                bandwidth=volume_est_cfg.BANDWIDTH
            )
            self.data_sources["kernel_smoothing"] = self.kernel_smoothing_data

    def process_scans(self, all_scans) -> defaultdict(list):
        scan_dict = defaultdict(list)
        for patient_id, scans in all_scans.items():
            for file_path, volume in scans:
                date_str = os.path.basename(file_path).split("_")[1].replace(".nii.gz", "")
                date = datetime.strptime(date_str, "%Y%m%d")

                if (
                    not volume_est_cfg.TEST_DATA
                    and patient_id in self.dob_df["BCH MRN"].astype(str).values
                ):
                    dob = self.dob_df.loc[
                        self.dob_df["BCH MRN"] == int(patient_id), "Date of Birth"
                    ].iloc[0]
                    age = (date - dob).days / 365.25
                    scan_dict[patient_id].append((date, volume, age))
                else:
                    scan_dict[patient_id].append((date, volume))
        return scan_dict

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
        volume_changes = [
            self.calculate_volume_change(volumes[i - 1], vol)
            for i, vol in enumerate(volumes[1:], 1)
        ]
        volume_changes.insert(0, 0)

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
        if ages:
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
        for data_type, data in self.data_sources.items():
            if getattr(volume_est_cfg, data_type.upper(), None):
                self.plot_each_type(data, output_path, data_type)

    def plot_each_type(self, data, output_path, data_type):
        os.makedirs(output_path, exist_ok=True)
        for patient_id, volumes_data in data.items():
            volumes_data.sort(key=lambda x: x[0])  # sort by date

            if not volume_est_cfg.TEST_DATA:
                dates, volumes, ages = zip(*volumes_data)
            else:
                dates, volumes = zip(*volumes_data)

            self.plot_data(
                data_type,
                output_path,
                patient_id,
                dates,
                volumes,
                ages if not volume_est_cfg.TEST_DATA else None,
            )

    def plot_data(self, data_type, output_path, patient_id, dates, volumes, ages=None):
        os.makedirs(output_path, exist_ok=True)

        fig, a_x1 = self.setup_plot_base()

        a_x1.plot(dates, volumes, color="tab:blue", marker="o")
        self.add_volume_change_to_plot(a_x1, dates, volumes)

        if ages:
            self.add_age_to_plot(a_x1, dates, ages)

        plt.title(f"Patient ID: {patient_id} - {data_type}")
        fig.tight_layout()

        date_range = f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
        plt.savefig(os.path.join(output_path, f"volume_{data_type}_{patient_id}_{date_range}.png"))
        plt.close()

    def generate_csv(self, output_folder):
        """
        Generate a CSV file for each patient containing their time series volume data.

        Args:
            output_folder (str): Path to the directory where CSV files should be saved.
        """

        # Ensure output directory exists, if not, create it.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for patient_id, volume_data in self.raw_data.items():
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

    def apply_filtering(self, all_ids, filtered_scans) -> defaultdict(list):
        filtered_data = defaultdict(list)
        with open(volume_est_cfg.FEW_SCANS_FILE, "w", encoding="utf-8") as file:
            for patient_id, scans in list(all_ids.items()):
                if len(filtered_scans.get(patient_id, [])) < 3:
                    file.write(f"{patient_id}\n")
                    for scan_id in scans:
                        file.write(f"---- {scan_id}\n")
                    if patient_id in filtered_scans:
                        del filtered_scans[patient_id]

        filtered_data = self.process_scans(filtered_scans)
        return filtered_data

    def apply_polysmoothing(self, poly_degree=3):
        polysmoothed_data = defaultdict(list)
        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[0])  # Sort by date

            dates, volumes, ages = zip(*[(date, volume, age) for date, volume, age in scans])
            # Polynomial interpolation
            poly_coeff = np.polyfit(mdates.date2num(dates), volumes, poly_degree)
            poly_interp = np.poly1d(poly_coeff)

            # Polynomial interpolation for ages
            poly_coeff_age = np.polyfit(mdates.date2num(dates), ages, poly_degree)
            poly_interp_age = np.poly1d(poly_coeff_age)

            # Generate interpolated values
            num_points = 50  # Number of points for interpolation
            start = mdates.date2num(min(dates))
            end = mdates.date2num(max(dates))
            interpolated_dates = mdates.num2date(np.linspace(start, end, num_points))

            # Generate smoothed volumes and ages
            smoothed_volumes = poly_interp(mdates.date2num(interpolated_dates))
            smoothed_ages = poly_interp_age(mdates.date2num(interpolated_dates))

            # Applying the polynomial smoothing
            polysmoothed_data[patient_id] = [
                (interpolated_date, smoothed_vol, smoothed_age)
                for interpolated_date, smoothed_vol, smoothed_age in zip(
                    interpolated_dates, smoothed_volumes, smoothed_ages
                )
            ]
        return polysmoothed_data

    def gaussian_kernel(self, x_, x_i, bandwidth):
        """Gaussian Kernel Function"""
        return norm.pdf((x_ - x_i) / bandwidth)

    def apply_kernel_smoothing(self, bandwidth=30):
        kernelsmoothed_data = defaultdict(list)
        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[0])  # Sort by date

            dates, volumes, ages = zip(*scans)
            num_dates = mdates.date2num(dates)

            # Preallocate arrays for smoothed volumes and ages
            smoothed_volumes = np.zeros(len(volumes))
            smoothed_ages = np.zeros(len(ages))

            # Apply kernel smoothing to volumes and ages
            for i, (date, volume, age) in enumerate(zip(num_dates, volumes, ages)):
                weights = np.array(
                    [self.gaussian_kernel(date, date_i, bandwidth) for date_i in num_dates]
                )
                weights = weights / np.sum(weights)

                smoothed_volumes[i] = np.sum(weights * volumes)
                smoothed_ages[i] = np.sum(weights * ages)

            # Store smoothed data
            kernelsmoothed_data[patient_id] = [
                (mdates.num2date(date), smoothed_vol, smoothed_age)
                for date, smoothed_vol, smoothed_age in zip(
                    num_dates, smoothed_volumes, smoothed_ages
                )
            ]
        return kernelsmoothed_data

    def plot_comparison(self, output_path):
        unique_patient_ids = set(self.data_sources["raw"].keys())

        for patient_id in unique_patient_ids:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

            for i, (key, data) in enumerate(self.data_sources.items()):
                ax = axs[i]

                patient_data = data.get(patient_id, [])

                if not patient_data:
                    ax.set_title(f"No Data: {key}")
                    continue

                dates, volumes, _ = zip(*patient_data)

                ax.plot(dates, volumes, label=f"{key} data")
                ax.set_title(f"{key} Data for Patient {patient_id}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volume")
                ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"volume_comparison_{patient_id}.png"))
            plt.close(fig)


if __name__ == "__main__":
    print("Initializing Volume Estimator.")
    ve = VolumeEstimator(volume_est_cfg.SEG_DIR, volume_est_cfg.REDCAP_FILE)
    print("Getting prediction masks.")
    ve.process_files(max_patients=volume_est_cfg.LIMIT_LOADING)
    print("All files processed, generating plots.")
    ve.plot_volumes(output_path=volume_est_cfg.PLOTS_DIR)
    print("Saved all plots.")

    if volume_est_cfg.PLOT_COMPARISON:
        print("Generating volume comparison.")
        ve.plot_comparison(output_path=volume_est_cfg.PLOTS_DIR)
        print("Saved comparison.")
    print("Generating time-series csv's.")
    # ve.generate_csv(output_folder=volume_est_cfg.CSV_DIR)
