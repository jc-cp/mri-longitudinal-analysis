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
from math import isfinite
from multiprocessing import Pool, cpu_count

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import norm

from cfg import volume_est_cfg
from utils.helper_functions import (
    gaussian_kernel,
    weighted_median,
    prefix_zeros_to_six_digit_ids,
    normalize_data,
    compute_95_ci,
)


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
        self.window_smoothing_data = defaultdict(list)
        self.volume_rate_data = defaultdict(list)

        self.data_sources = {
            "raw": {},
            "filtered": {},
            "poly_smoothing": {},
            "kernel_smoothing": {},
            "window_smoothing": {},
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
                    self.dob_df["Date of Birth"], format="%d/%m/%Y"
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
        """
        Processes the .nii.gz files from a specified directory to generate volume estimates
        and apply various filters.
        Populates raw, filtered, poly-smoothed, and kernel-smoothed data.

        Args:
            max_patients (int, optional): Maximum number of patients to load. Defaults to None.
        """
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        all_ids = defaultdict(list)
        all_scans = defaultdict(list)
        filtered_scans = defaultdict(list)
        zero_volume_scans = defaultdict(list)
        zero_volume_counter = 0

        # Calculate volumes and filter out NaN ones first
        with Pool(cpu_count()) as pool:
            for file_path, volume in zip(file_paths, pool.map(self.estimate_volume, file_paths)):
                patient_id, scan_id = os.path.basename(file_path).split("_")[:2]
                patient_id = prefix_zeros_to_six_digit_ids(patient_id)

                # take into account the ones for filtering
                all_ids[patient_id].append(scan_id)
                all_scans[patient_id].append((file_path, volume))
                if volume != 0:
                    filtered_scans[patient_id].append((file_path, volume))
                else:
                    zero_volume_scans[patient_id].append(scan_id)
                    zero_volume_counter += 1
        # write zero_volume_scans to a file
        with open(volume_est_cfg.ZERO_VOLUME_FILE, "w", encoding="utf-8") as file:
            file.write(f"Total zero volume scans: {zero_volume_counter}\n")
            for patient_id, scans in zero_volume_scans.items():
                patient_id = prefix_zeros_to_six_digit_ids(patient_id)

                file.write(f"{patient_id}\n")
                for scan_id in scans:
                    file.write(f"---- {scan_id}\n")

        # store raw data
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
            self.poly_smoothing_data = self.apply_polysmoothing()
            self.data_sources["poly_smoothing"] = self.poly_smoothing_data
        if volume_est_cfg.KERNEL_SMOOTHING:
            self.kernel_smoothing_data = self.apply_kernel_smoothing(
                bandwidth=volume_est_cfg.BANDWIDTH
            )
            self.data_sources["kernel_smoothing"] = self.kernel_smoothing_data
        if volume_est_cfg.WINDOW_SMOOTHING:
            self.window_smoothing_data = self.apply_sliding_window_interpolation()
            self.data_sources["window_smoothing"] = self.window_smoothing_data

        # Additionally, process the volume rate data
        print("Adding volume rate data!")
        self.volume_rate_data = self.calculate_volume_rate_of_change(filtered_scans)

    def process_scans(self, all_scans) -> defaultdict(list):
        """
        Processes scan information to calculate volumes and optionally age.

        Args:
            all_scans (dict): Dictionary containing scan information.

        Returns:
            defaultdict(list): Processed scan information, including date,
            volume, and optionally age.
        """
        scan_dict = defaultdict(list)
        id_list = list(self.dob_df["BCH MRN"])
        id_list = [
            f"0{str(id_item)}" if len(str(id_item)) == 6 else str(id_item)
            for id_item in self.dob_df["BCH MRN"]
        ]

        for patient_id, scans in all_scans.items():
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)

            volumes = [volume for _, volume in scans]
            if volume_est_cfg.NORMALIZE:
                # Extracting just the volume data for normalization
                normalized_volumes = normalize_data(volumes)
            else:
                normalized_volumes = volumes

            for (file_path, _), morm_volume in zip(scans, normalized_volumes):
                date_str = os.path.basename(file_path).split("_")[1].replace(".nii.gz", "")
                date = datetime.strptime(date_str, "%Y%m%d")

                if not volume_est_cfg.TEST_DATA and patient_id in id_list:
                    dob = self.dob_df.loc[
                        self.dob_df["BCH MRN"] == int(patient_id), "Date of Birth"
                    ].iloc[0]
                    age = (date - dob).days / 365.25
                    scan_dict[patient_id].append((date, morm_volume, age))
                else:
                    scan_dict[patient_id].append((date, morm_volume))
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
        """
        Plots the volumes data for each data type (raw, filtered, etc.)

        Args:
            output_path (str): The directory where plots should be saved.
        """
        for data_type, data in self.data_sources.items():
            if getattr(volume_est_cfg, data_type.upper(), None):
                self.plot_each_type(data, output_path, data_type)

    def plot_each_type(self, data, output_path, data_type):
        """
        Plots the volumes data for a specific data type.

        Args:
            data (dict): Data to plot.
            output_path (str): The directory where plots should be saved.
            data_type (str): The type of data ('raw', 'filtered', etc.)
        """
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
        """
        Plots and saves the volume data for a single patient.

        Args:
            data_type (str): The type of data ('raw', 'filtered', etc.)
            output_path (str): The directory where plots should be saved.
            patient_id (str): The ID of the patient.
            dates (list): List of dates.
            volumes (list): List of volumes.
            ages (list, optional): List of ages. Defaults to None.
        """
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

        for patient_id, volume_data in self.filtered_data.items():
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
        """
        Applies filtering to exclude scans with less than a certain number of points.

        Args:
            all_ids (dict): Dictionary of all patient IDs.
            filtered_scans (dict): Dictionary of filtered scans.

        Returns:
            defaultdict(list): Filtered scan data.
        """
        filtered_data = defaultdict(list)

        # Counters for patients with too few scans and such scans
        total_patients_with_few_scans = 0
        total_scans_ignored = 0

        with open(volume_est_cfg.FEW_SCANS_FILE, "w", encoding="utf-8") as file:
            for patient_id, scans in list(all_ids.items()):
                patient_id = prefix_zeros_to_six_digit_ids(patient_id)
                if len(filtered_scans.get(patient_id, [])) < 3:
                    total_patients_with_few_scans += 1  # Increment the patient counter
                    total_scans_ignored += len(scans)
                    file.write(f"{patient_id}\n")
                    for scan_id in scans:
                        file.write(f"---- {scan_id}\n")
                    if patient_id in filtered_scans:
                        del filtered_scans[patient_id]

            # Write total counts to the file
            file.write(f"\nTotal patients with too few scans: {total_patients_with_few_scans}\n")
            file.write(f"Total scans not considered due to too few scans: {total_scans_ignored}\n")

        filtered_data = self.process_scans(filtered_scans)
        return filtered_data

    def apply_polysmoothing(self, poly_degree=3):
        """
        Applies polynomial smoothing to the volume data, dynamically selecting
        polynomial degree based on the number of scans for each patient.

        Returns:
            defaultdict(list): Data after polynomial smoothing.
        """

        average_scans = np.mean([len(scans) for scans in self.filtered_data.values()])
        print("Average scan number per patient:", average_scans)

        polysmoothed_data = defaultdict(list)
        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[0])  # Sort by date
            n_scans = len(scans)

            # Dynamic selection of polynomial degree
            if n_scans < 4:
                poly_degree = 1
            elif 4 <= n_scans <= 5:
                poly_degree = 2
            elif 6 <= n_scans <= 7:
                poly_degree = 3
            elif 8 <= n_scans <= 9:
                poly_degree = 4
            else:
                poly_degree = 5

            dates, volumes, ages = zip(*[(date, volume, age) for date, volume, age in scans])
            # Polynomial interpolation
            poly_coeff = np.polyfit(mdates.date2num(dates), volumes, poly_degree)
            poly_interp = np.poly1d(poly_coeff)

            # Polynomial interpolation for ages
            poly_coeff_age = np.polyfit(mdates.date2num(dates), ages, poly_degree)
            poly_interp_age = np.poly1d(poly_coeff_age)

            # Generate interpolated values
            num_points = 25  # Number of points for interpolation
            start = mdates.date2num(min(dates))
            end = mdates.date2num(max(dates))
            interpolated_dates = mdates.num2date(np.linspace(start, end, num_points))

            # Generate smoothed volumes and ages
            smoothed_volumes = poly_interp(mdates.date2num(interpolated_dates))
            smoothed_volumes = np.maximum(smoothed_volumes, 0)
            smoothed_ages = poly_interp_age(mdates.date2num(interpolated_dates))

            # Applying the polynomial smoothing
            polysmoothed_data[patient_id] = [
                (interpolated_date, smoothed_vol, smoothed_age)
                for interpolated_date, smoothed_vol, smoothed_age in zip(
                    interpolated_dates, smoothed_volumes, smoothed_ages
                )
            ]
        return polysmoothed_data

    def apply_kernel_smoothing(self, bandwidth=None):
        """
        Applies kernel smoothing to the volume data.

        Args:
            bandwidth (int, optional): The bandwidth for the Gaussian kernel. Defaults to 30.

        Returns:
            defaultdict(list): Data after kernel smoothing.
        """
        kernelsmoothed_data = defaultdict(list)

        if bandwidth is None:
            all_volumes = [
                volume for _, scans in self.filtered_data.items() for _, volume, _ in scans
            ]
            max_volume = max(all_volumes)
            bandwidth = 0.2 * max_volume

        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[0])  # Sort by date

            dates, volumes, ages = zip(*scans)
            num_dates = mdates.date2num(dates)

            # Preallocate arrays for smoothed volumes and ages
            smoothed_volumes = np.zeros(len(volumes))
            smoothed_ages = np.zeros(len(ages))

            # Apply kernel smoothing to volumes and ages
            for i, (date, _, _) in enumerate(zip(num_dates, volumes, ages)):
                weights = np.array(
                    [gaussian_kernel(date, date_i, bandwidth) for date_i in num_dates]
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

    def apply_sliding_window_interpolation(self, window_size=3):
        """
        Apply sliding window interpolation to smooth volumetric data of patients.

        This method applies a Gaussian-weighted median-based smoothing technique
        on a time series of scan volumes. The weighted median is calculated for
        each scan based on neighboring scans within a defined window.

        Parameters:
            window_size (int): Size of the window used to calculate the weighted
                            median. Should be an odd number for balanced
                            calculation. Default is 3.

        Returns:
            dict: A dictionary containing the interpolated scan data for each
                patient, sorted by scan date.

        Example:
            interpolated_data = self.apply_sliding_window_interpolation(window_size=5)
        """
        interpolated_data = defaultdict(list)

        for patient_id, scans in self.filtered_data.items():
            # Sort by date just in case
            scans.sort(key=lambda x: x[0])

            # Only volumes are taken for weighted median
            volumes = [volume for _, volume, _ in scans]

            for i, (date, volume, age) in enumerate(scans):
                # Find indices of scans within the window
                left = max(0, i - window_size // 2)
                right = min(len(scans), i + window_size // 2 + 1)

                # Extract neighboring scans and their temporal closeness
                neighbors = volumes[left:right]
                neighbors.sort()  # Sort for median calculation

                # Weights based on Gaussian distribution
                middle = (right - left) // 2
                weights = [norm.pdf(x, middle, window_size // 3) for x in range(len(neighbors))]
                if not all(isfinite(w) for w in weights):
                    print(f"Skipping invalid weights for patient {patient_id}, index {i}")
                    continue
                neighbors, weights = zip(*sorted(zip(neighbors, weights)))

                # Compute the weighted median
                weighted_vol = weighted_median(neighbors, weights)
                if weighted_vol is None:
                    print(f"Skipping None value for patient {patient_id}, index {i}")
                    continue
                # Replace volume with weighted median
                interpolated_data[patient_id].append((date, weighted_vol, age))
                # print(f"Interpolated data for patient {patient_id}, index {i}: {weighted_vol}")
        return interpolated_data

    def plot_comparison(self, output_path):
        """
        Plots a comparison of all the data types for each unique patient ID.

        Args:
            output_path (str): The directory where the comparison plots should be saved.
        """
        unique_patient_ids = set(self.data_sources["raw"].keys())

        for patient_id in unique_patient_ids:
            fig, axs = plt.subplots(1, len(self.data_sources), figsize=(20, 5))

            for i, (key, data) in enumerate(self.data_sources.items()):
                a_x = axs[i]

                patient_data = data.get(patient_id, [])

                if not patient_data:
                    a_x.set_title(f"No Data: {key}")
                    continue

                dates, volumes, _ = zip(*patient_data)

                a_x.plot(dates, volumes, label=f"{key} data")
                a_x.set_title(f"{key} Data for Patient {patient_id}")
                a_x.set_xlabel("Date")
                a_x.set_ylabel("Volume")
                a_x.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"volume_comparison_{patient_id}.png"))
            plt.close(fig)

    def analyze_volume_changes(self):
        """
        Analyze volume changes and compute the 95% confidence interval.
        """
        scan_data = self.filtered_data

        # Calculate volume changes for each patient
        volume_changes = []
        for _, scans in scan_data.items():
            for i in range(1, len(scans)):
                prev_vol = scans[i - 1][1]  # Previous volume
                curr_vol = scans[i][1]  # Current volume
                volume_changes.append(curr_vol - prev_vol)

        # Compute the 95% confidence interval
        lower, upper = compute_95_ci(volume_changes)
        print(f"95% CI for average volume change: ({lower:.2f}, {upper:.2f})")

        volume_rates = [
            rate[1]
            for sublist in self.volume_rate_data.values()
            for rate in sublist
            if rate[1] is not None
        ]

        # Calculate mean and standard deviation
        mean_rate = np.mean(volume_rates)
        std_rate = np.std(volume_rates)

        # Calculate the 95% confidence interval
        margin_of_error = 1.96 * (std_rate / np.sqrt(len(volume_rates)))
        confidence_interval = (mean_rate - margin_of_error, mean_rate + margin_of_error)

        print(f"95% CI for average volume rate of change: {confidence_interval}")

    def calculate_volume_rate_of_change(self, scans) -> defaultdict(list):
        """
        Calculates the rate of volume change (normalized by time span) for each patient.

        Args:
            all_scans (dict): Dictionary containing scan information.

        Returns:
            defaultdict(list): Processed scan information with rate of volume change and date.
        """
        rate_of_change_dict = defaultdict(list)

        for patient_id, scans in scans.items():
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)

            # sort scans by date for accurate calculation
            scans = sorted(scans, key=lambda x: x[0])

            # initialize previous volume and date
            prev_volume = None
            prev_date = None

            for file_path, volume in scans:
                date_str = os.path.basename(file_path).split("_")[1].replace(".nii.gz", "")
                date = datetime.strptime(date_str, "%Y%m%d")

                if prev_date is not None:
                    days_diff = (date - prev_date).days
                    volume_rate_of_change = (volume - prev_volume) / days_diff
                else:
                    volume_rate_of_change = None  # or you can set it to 0 if you prefer

                prev_volume = volume
                prev_date = date

                rate_of_change_dict[patient_id].append((date, volume_rate_of_change))

        return rate_of_change_dict


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

    if volume_est_cfg.CONFIDENCE_INTERVAL:
        print("Analyzing volume changes.")
        ve.analyze_volume_changes()

    print("Generating time-series csv's.")
    ve.generate_csv(output_folder=volume_est_cfg.CSV_DIR)
