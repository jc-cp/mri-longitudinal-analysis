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

    If not in test mode and not dealing with CBTN,
    it also uses BCH patient's Date of Birth (DoB) for a consistent analysis.

    Attributes:
        path (str): Path to the directory containing segmentation files.
        dob_df (pd.DataFrame): DataFrame containing Date of Birth information, if available.
        volumes (dict): Dictionary storing volume data for each patient.
    """

    def __init__(self, segmentations_path, clinical_data_file):
        """
        Initialize the VolumeEstimator with given paths.

        Args:
            segmentations_path (str): Path to the directory containing segmentation files.
        """
        self.raw_data = defaultdict(list)
        self.filtered_data = defaultdict(list)
        self.poly_smoothing_data = defaultdict(list)
        self.kernel_smoothing_data = defaultdict(list)
        self.window_smoothing_data = defaultdict(list)
        self.volume_rate_data = defaultdict(list)

        os.makedirs(volume_est_cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(volume_est_cfg.PLOTS_DIR, exist_ok=True)
        os.makedirs(volume_est_cfg.CSV_DIR, exist_ok=True)

        self.data_sources = {
            "raw": {},
            "filtered": {},
            "poly_smoothing": {},
            "kernel_smoothing": {},
            "window_smoothing": {},
        }

        self.segmentations_path = segmentations_path

        if not volume_est_cfg.TEST_DATA:
            # Get the ids from the directory to process
            identifiers_in_dir = set()
            for file_name in os.listdir(self.segmentations_path):
                if file_name.endswith("_mask.nii.gz"):
                    patient_id, _ = self.get_identifier(file_name)
                    identifiers_in_dir.add(patient_id)

            clinical_data = pd.read_csv(clinical_data_file, sep=",", encoding="utf-8")

            if volume_est_cfg.CBTN_DATA:
                # Process the .csv with clinical data
                clinical_data["CBTN Subject ID"] = clinical_data["CBTN Subject ID"].astype(str)
                final_clinical_data = clinical_data[
                    clinical_data["CBTN Subject ID"].isin(identifiers_in_dir)
                ]
                mismatch_ids = len(clinical_data) - len(final_clinical_data)
                print(f"Number of unique patient IDs in the original CSV: {len(clinical_data)}")
                print(f"Number of unique patient IDs in the directory: {len(identifiers_in_dir)}")
                print(f"Number of unique patient IDs in the final CSV: {len(final_clinical_data)}")
                print(f"Number of reduced patient IDs: {mismatch_ids}")

                assert (
                    len(final_clinical_data) == volume_est_cfg.NUMBER_TOTAL_CBTN_PATIENTS
                ), "Warning: The length of the filtered dataset is not 115. Check the csv again."

                self.clinical_data = final_clinical_data

            if volume_est_cfg.BCH_DATA:
                clinical_data["BCH MRN"] = clinical_data["BCH MRN"].astype(str)
                final_clinical_data = clinical_data[
                    clinical_data["BCH MRN"].isin(identifiers_in_dir)
                ]
                mismatch_ids = len(clinical_data) - len(final_clinical_data)
                print(f"\tNumber of unique patient IDs in the original CSV: {len(clinical_data)}")
                print(f"\tNumber of unique patient IDs in the directory: {len(identifiers_in_dir)}")
                print(
                    f"\tNumber of unique patient IDs in the final CSV: {len(final_clinical_data)}"
                )
                print(f"\tNumber of reduced patient IDs: {mismatch_ids}")

                assert (
                    len(final_clinical_data) == volume_est_cfg.NUMBER_TOTAL_BCH_PATIENTS
                ), "Warning: The length of the filtered dataset is not 85. Check the csv again."

                clinical_data["Date of Birth"] = pd.to_datetime(
                    clinical_data["Date of Birth"], format="%d/%m/%Y"
                )
                self.clinical_data = final_clinical_data

    @staticmethod
    def estimate_volume(segmentation_path):
        """
        Estimate the volume of the given segmentation file.

        Args:
            segmentation_path (str): Path to the segmentation file.

        Returns:
            float: Total volume of the segmentation.
        """
        total_volume = 0
        try:
            segmentation = sitk.ReadImage(segmentation_path)
            voxel_spacing = segmentation.GetSpacing()
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            num_voxels = (segmentation_array > 0).sum()
            total_volume = num_voxels * voxel_volume
        except Exception as error:
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
        if previous is None:
            previous = 0
        return ((current - previous) / previous) * 100 if previous != 0 else 0

    @staticmethod
    def get_identifier(file_name):
        """Extracts patient ID and scan ID from the file name."""
        # file format are:
        # imageXYZ_patientID_scanID_mask.nii.gz
        # imageXYZ_patientID_scanID.nii.gz
        parts = file_name.split("_")
        return parts[1], parts[2]

    #############################
    # Data processing functions #
    #############################

    def process_files(self, max_patients=None):
        """
        Processes the .nii.gz files from a specified directory to generate volume estimates
        and apply various filters.
        Populates raw, filtered, poly-smoothed, and kernel-smoothed data.

        Args:
            max_patients (int, optional): Maximum number of patients to load. Defaults to None.
        """
        file_paths = glob.glob(os.path.join(self.segmentations_path, "*_mask.nii.gz"))

        # all_ids = defaultdict(list)
        # filtered_scans = defaultdict(list)

        with Pool(cpu_count()) as pool:
            volumes = pool.map(self.estimate_volume, file_paths)

        all_scans = defaultdict(list)
        zero_volume_scans = defaultdict(list)
        for file_path, volume in zip(file_paths, volumes):
            basename = os.path.basename(file_path)
            patient_id, scan_id = self.get_identifier(basename)
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)
            all_scans[patient_id].append((file_path, volume, scan_id))
            if volume == 0:
                zero_volume_scans[patient_id].append(scan_id)

        # store raw data
        self.raw_data = self.process_scans(dict(all_scans))
        self.data_sources["raw"] = self.raw_data

        if volume_est_cfg.FILTERED:
            filtered_data = self.apply_filtering(all_scans, zero_volume_scans)
            self.filtered_data = self.process_scans(filtered_data)
            self.data_sources["filtered"] = self.filtered_data
            print("\tAdded filtered data!")

            if max_patients is not None and max_patients < len(self.filtered_data):
                self.filtered_data = dict(list(self.filtered_data.items())[:max_patients])
            print(f"\tSet the max number of patient to be loaded to: {max_patients}")
            print(f"\tThe length of the filtered dataset is: {len(self.filtered_data)}")

        # Additional logic to process and store other states of data
        # (poly-smoothed, kernel-smoothed, window-smoothed)
        if volume_est_cfg.POLY_SMOOTHING:
            self.poly_smoothing_data = self.apply_polysmoothing()
            self.data_sources["poly_smoothing"] = self.poly_smoothing_data
            print("\tAdded polynomial smoothing data!")
        if volume_est_cfg.KERNEL_SMOOTHING:
            self.kernel_smoothing_data = self.apply_kernel_smoothing(
                bandwidth=volume_est_cfg.BANDWIDTH
            )
            self.data_sources["kernel_smoothing"] = self.kernel_smoothing_data
            print("\tAdded kernel smoothing data!")
        if volume_est_cfg.WINDOW_SMOOTHING:
            self.window_smoothing_data = self.apply_sliding_window_interpolation()
            self.data_sources["window_smoothing"] = self.window_smoothing_data
            print("\tAdded sliding window smoothing data!")

        # Additionally, process the volume rate data
        self.volume_rate_data = self.calculate_volume_rate_of_change(
            self.filtered_data, use_age=True
        )
        print("\tAdded volume rate data!")

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
        if volume_est_cfg.BCH_DATA:
            column = "BCH MRN"
        elif volume_est_cfg.CBTN_DATA:
            column = "CBTN Subject ID"
        else:
            column = "Patient_ID"

        id_list = list(self.clinical_data[column])
        id_list = [
            f"0{str(id_item)}" if len(str(id_item)) == 6 else str(id_item)
            for id_item in self.clinical_data[column]
        ]

        for patient_id, scans in all_scans.items():
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)

            volumes = [volume for _, volume, _ in scans]
            if volume_est_cfg.NORMALIZE:
                # Extracting just the volume data for normalization
                normalized_volumes = normalize_data(volumes)
            else:
                normalized_volumes = volumes

            for (file_path, _, _), morm_volume in zip(scans, normalized_volumes):
                scan_id = os.path.basename(file_path).split("_")[2].replace(".nii.gz", "")

                if (
                    not volume_est_cfg.TEST_DATA
                    and volume_est_cfg.BCH_DATA
                    and patient_id in id_list
                ):
                    date = datetime.strptime(scan_id, "%Y%m%d")
                    dob = self.clinical_data.loc[
                        self.clinical_data[column] == int(patient_id), "Date of Birth"
                    ].iloc[0]
                    age = (date - dob).days
                    scan_dict[patient_id].append((date, morm_volume, age))
                elif (
                    not volume_est_cfg.TEST_DATA
                    and volume_est_cfg.CBTN_DATA
                    and patient_id in id_list
                ):
                    age = scan_id
                    scan_dict[patient_id].append((morm_volume, age))
                else:
                    scan_dict[patient_id].append((morm_volume))
        return scan_dict

    def apply_filtering(self, all_scans, zero_volume_scans) -> defaultdict(list):
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
        few_scans_file_path = volume_est_cfg.FEW_SCANS_FILE
        zero_volume_file_path = volume_est_cfg.ZERO_VOLUME_FILE

        with open(zero_volume_file_path, "w", encoding="utf-8") as file:
            for patient_id, scans in zero_volume_scans.items():
                file.write(f"{patient_id}\n")
                for scan_id in scans:
                    file.write(f"---- {scan_id}\n")
            file.write(
                "\nTotal zero volume scans:"
                f" {sum(len(scans) for scans in zero_volume_scans.values())}\n"
            )

        with open(few_scans_file_path, "w", encoding="utf-8") as file:
            for patient_id, scans in all_scans.items():
                if len(scans) >= 3:
                    filtered_data[patient_id] = scans
                else:
                    total_patients_with_few_scans += 1  # Increment the patient counter
                    file.write(f"{patient_id} - Too Few Scans: {len(scans)}\n")
                    for _, _, scan_id in scans:
                        file.write(f"---- {scan_id}\n")
            file.write(f"\nTotal patients with too few scans: {total_patients_with_few_scans}\n")

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

    #############################
    # Output related functions  #
    #############################

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

    def calculate_volume_rate_of_change(self, scans, use_age=False) -> defaultdict(list):
        """
        Calculates the rate of volume change (normalized by time span) for each patient.

        Args:
            all_scans (dict): Dictionary containing scan information.

        Returns:
            defaultdict(list): Processed scan information with rate of volume change and date.
        """
        rate_of_change_dict = defaultdict(list)

        for patient_id, patient_scans in scans.items():
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)

            sorted_scans = sorted(patient_scans, key=lambda x: x[0] if not use_age else x[-1])
            prev_volume = None
            prev_time_point = None

            for volume, time_point in sorted_scans:
                if prev_time_point is not None:
                    days_diff = float(time_point) - float(prev_time_point)

                    volume_rate_of_change = (
                        (volume - prev_volume) / days_diff if days_diff != 0 else 0
                    )
                else:
                    volume_rate_of_change = None  # or you can set it to 0 if you prefer

                prev_volume = volume
                prev_time_point = time_point

                rate_of_change_dict[patient_id].append((time_point, volume_rate_of_change))

        return rate_of_change_dict

    def generate_csv(self, output_folder):
        """
        Generate a CSV file for each patient containing their time series volume data.

        Args:
            output_folder (str): Path to the directory where CSV files should be saved.
        """

        # Ensure output directory exists, if not, create it.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for patient_id, volume_data in self.kernel_smoothing_data.items():
            csv_file_path = os.path.join(output_folder, f"{patient_id}.csv")

            with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)

                # Determine header based on whether it's test data or not
                header = [
                    "Date",
                    "Volume",
                    "Normalized Volume",
                    "Growth[%]",
                    "Normalized Growth[%]",
                ]
                if not volume_est_cfg.TEST_DATA:
                    header.insert(3, "Age")  # Insert age before Growth[%]
                csv_writer.writerow(header)
                # Sort data by date to ensure time series is in order
                sorted_volume_data = sorted(volume_data, key=lambda x: x[0])
                initial_volume = sorted_volume_data[0][1] if sorted_volume_data else None
                previous_volume = None

                for entry in sorted_volume_data:
                    if not volume_est_cfg.TEST_DATA:
                        date, volume, age = entry
                    else:
                        date, volume = entry

                    normalized_volume = volume / initial_volume if initial_volume else 0
                    percentage_growth = self.calculate_volume_change(previous_volume, volume)
                    normalized_growth = percentage_growth / volume if volume else 0

                    row = [
                        date.strftime("%Y-%m-%d"),
                        volume,
                        normalized_volume,
                        percentage_growth,
                        normalized_growth,
                    ]
                    if not volume_est_cfg.TEST_DATA:
                        row.insert(3, age)  # Insert age before Growth[%]

                    csv_writer.writerow(row)
                    previous_volume = volume

    ############################
    # Plotting-related methods #
    ############################

    def setup_plot_base(self, normalize=False):
        """
        Setup the base of a plot with the necessary properties.

        Returns:
            tuple: A tuple containing the figure and axis objects of the plot.
        """
        plt.close("all")
        fig, a_x1 = plt.subplots(figsize=(12, 8))
        a_x1.set_xlabel("Scan Date")
        if normalize:
            a_x1.set_ylabel("Normalized Volume (mm³)", color="tab:blue")
        else:
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
            self.plot_normalized_data(
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
        fig, a_x1 = self.setup_plot_base(normalize=False)

        a_x1.plot(dates, volumes, color="tab:blue", marker="o")
        self.add_volume_change_to_plot(a_x1, dates, volumes)

        if ages:
            self.add_age_to_plot(a_x1, dates, ages)

        plt.title(f"Patient ID: {patient_id} - {data_type}")
        fig.tight_layout()

        date_range = f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
        plt.savefig(os.path.join(output_path, f"volume_{data_type}_{patient_id}_{date_range}.png"))
        plt.close(fig)

    def plot_normalized_data(self, data_type, output_path, patient_id, dates, volumes, ages=None):
        """
        Plots and saves the normalized volume data for a single patient.
        Args:
            data_type (str): The type of data ('raw', 'filtered', etc.)
            output_path (str): The directory where plots should be saved.
            patient_id (str): The ID of the patient.
            dates (list): List of dates.
            volumes (list): List of normalized volumes.
            ages (list, optional): List of ages. Defaults to None.
        """
        fig, ax1 = self.setup_plot_base(normalize=True)

        patient_data = self.clinical_data[self.clinical_data["Patient_ID"] == int(patient_id)]
        if not patient_data.empty:
            dates = [d.replace(tzinfo=None) for d in dates]

            treatment_date = patient_data["Date First Treatment"].iloc[0]
            treatment_date = pd.to_datetime(treatment_date, format="%Y-%m-%d").replace(tzinfo=None)

            initial_volume = volumes[0] if volumes[0] not in [0, np.nan] else 1
            normalized_volumes = [v / initial_volume for v in volumes]

            if treatment_date:
                first_post_treatment_index = next(
                    (i for i, date in enumerate(dates) if date >= treatment_date), len(dates)
                )

                # Split the data into pre-treatment and post-treatment
                pre_treatment_dates = dates[
                    : first_post_treatment_index + 1
                ]  # Includes the last point before treatment
                pre_treatment_volumes = normalized_volumes[: first_post_treatment_index + 1]

                post_treatment_dates = dates[
                    first_post_treatment_index:
                ]  # Starts from the overlapping point
                post_treatment_volumes = normalized_volumes[first_post_treatment_index:]

                # Plot pre-treatment data in blue
                ax1.plot(
                    pre_treatment_dates,
                    pre_treatment_volumes,
                    color="tab:blue",
                    marker="o",
                    linestyle="-",
                )

                # Plot post-treatment data in grey, starting with the overlapping point
                ax1.plot(
                    post_treatment_dates,
                    post_treatment_volumes,
                    color="grey",
                    marker="o",
                    linestyle="-",
                )
                ax1.axvline(x=treatment_date, color="red", linestyle="--", label="Treatment Date")
            else:
                # Plot all data in blue if no treatment date
                print("here")
                ax1.plot(dates, normalized_volumes, color="tab:blue", marker="o", linestyle="-")

            self.add_volume_change_to_plot(ax1, dates, normalized_volumes)

            if ages:
                self.add_age_to_plot(ax1, dates, ages)

            ax1.legend(loc="best")
            plt.title(f"Patient ID: {patient_id} - Normalized Volume {data_type}")
            fig.tight_layout()
            date_range = f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
            plt.savefig(
                os.path.join(
                    output_path, f"normalized_volume_{data_type}_{patient_id}_{date_range}.png"
                )
            )
            plt.close(fig)
        else:
            return

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


if __name__ == "__main__":
    print("Initializing Volume Estimator:")
    ve = VolumeEstimator(volume_est_cfg.SEG_DIR, volume_est_cfg.CLINICAL_DATA_FILE)
    print("\tAdjusted the clinical data.")
    print("Processing files:")
    if volume_est_cfg.BCH_DATA:
        ve.process_files(max_patients=volume_est_cfg.NUMBER_TOTAL_BCH_PATIENTS)
    elif volume_est_cfg.CBTN_DATA:
        ve.process_files(max_patients=volume_est_cfg.NUMBER_TOTAL_CBTN_PATIENTS)
    else:
        ve.process_files()
    print("All files processed, generating plots.")
    # ve.plot_volumes(output_path=volume_est_cfg.PLOTS_DIR)
    # print("Saved all plots.")

    # if volume_est_cfg.PLOT_COMPARISON:
    #     print("Generating volume comparison.")
    #     ve.plot_comparison(output_path=volume_est_cfg.PLOTS_DIR)
    #     print("Saved comparison.")

    # if volume_est_cfg.CONFIDENCE_INTERVAL:
    #     print("Analyzing volume changes.")
    #     ve.analyze_volume_changes()

    # print("Generating time-series csv's.")
    # ve.generate_csv(output_folder=volume_est_cfg.CSV_DIR)
