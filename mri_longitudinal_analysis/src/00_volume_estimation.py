"""
Main script to initialize and run the VolumeEstimator.
    
This script initializes the VolumeEstimator with appropriate configuration settings,
processes provided segmentation files, visualizes the volume estimations, and 
exports the results to specified directories.
"""
import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from cfg.src import volume_est_cfg
from utils.helper_functions import (
    weighted_median,
    prefix_zeros_to_six_digit_ids,
    compute_95_ci,
)
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc
import ruptures as rpt


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

    def __init__(self, segmentations_path):
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
        self.moving_average_data = defaultdict(list)
        self.volume_change_rate = defaultdict(list)
        self.volume_change_pattern = defaultdict(list)
        self.volume_change_type = defaultdict(list)

        os.makedirs(volume_est_cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(volume_est_cfg.PLOTS_DIR, exist_ok=True)
        os.makedirs(volume_est_cfg.CSV_DIR, exist_ok=True)

        self.data_sources = {
            "raw": {},
            "filtered": {},
            "poly_smoothing": {},
            "kernel_smoothing": {},
            "window_smoothing": {},
            "moving_average": {},
        }

        self.segmentations_path = segmentations_path

        if not volume_est_cfg.TEST_DATA:
            # Get the ids from the directory to process
            identifiers_in_dir = set()
            for file_name in os.listdir(self.segmentations_path):
                if file_name.endswith("_mask.nii.gz"):
                    patient_id, _ = self.get_identifier(file_name)
                    identifiers_in_dir.add(patient_id)

            if volume_est_cfg.JOINT_DATA:
                # join both datasets by merging dataframes
                clinical_data_file_bch = volume_est_cfg.CLINICAL_DATA_FILE_BCH
                clinical_data_file_cbtn = volume_est_cfg.CLINICAL_DATA_FILE_CBTN
                clinical_data_bch = pd.read_csv(
                    clinical_data_file_bch, sep=",", encoding="utf-8"
                )
                clinical_data_cbtn = pd.read_csv(
                    clinical_data_file_cbtn, sep=",", encoding="utf-8"
                )

                # Process the .csv with clinical data
                clinical_data_bch["BCH MRN"] = clinical_data_bch["BCH MRN"].astype(str)
                clinical_data_bch["BCH MRN"] = clinical_data_bch["BCH MRN"].apply(
                    prefix_zeros_to_six_digit_ids
                )
                clinical_data_cbtn["CBTN Subject ID"] = clinical_data_cbtn[
                    "CBTN Subject ID"
                ].astype(str)

                # merge both dataframe by given columns
                final_clinical_data = pd.merge(
                    clinical_data_bch,
                    clinical_data_cbtn,
                    left_on="BCH MRN",
                    right_on="CBTN Subject ID",
                    how="outer",
                )

                final_clinical_data["Patient_ID"] = final_clinical_data[
                    "BCH MRN"
                ].fillna(final_clinical_data["CBTN Subject ID"])
                final_clinical_data = final_clinical_data.drop(
                    columns=["BCH MRN", "CBTN Subject ID"]
                )
                final_clinical_data = final_clinical_data[
                    final_clinical_data["Patient_ID"].isin(identifiers_in_dir)
                ]
                mismatch_ids_bch = len(clinical_data_bch) - len(final_clinical_data)
                mismatch_ids_cbtn = len(clinical_data_cbtn) - len(final_clinical_data)
                mismatch_ids = identifiers_in_dir - set(
                    final_clinical_data["Patient_ID"].values
                )
                print(
                    f"\tNumber of unique patient IDs in the original CSV: {len(clinical_data_bch)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the original CSV: {len(clinical_data_cbtn)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the directory: {len(identifiers_in_dir)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the final CSV: {len(final_clinical_data)}"
                )
                print(
                    f"\tNumber of reduced patient IDs: {mismatch_ids_bch + mismatch_ids_cbtn}"
                )
                print(f"\tNumber of mismatched patient IDs: {len(mismatch_ids)}")
                assert (
                    len(final_clinical_data)
                    == volume_est_cfg.NUMBER_TOTAL_JOINT_PATIENTS
                ), "Warning: The length of the filtered dataset is not 111. Check the csv again."

                self.clinical_data = final_clinical_data

            elif volume_est_cfg.CBTN_DATA:
                clinical_data_file = volume_est_cfg.CLINICAL_DATA_FILE_CBTN
                clinical_data = pd.read_csv(
                    clinical_data_file, sep=",", encoding="utf-8"
                )
                # Process the .csv with clinical data
                clinical_data["CBTN Subject ID"] = clinical_data[
                    "CBTN Subject ID"
                ].astype(str)
                final_clinical_data = clinical_data[
                    clinical_data["CBTN Subject ID"].isin(identifiers_in_dir)
                ]
                mismatch_ids = len(clinical_data) - len(final_clinical_data)
                print(
                    f"\tNumber of unique patient IDs in the original CSV: {len(clinical_data)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the directory: {len(identifiers_in_dir)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the final CSV: {len(final_clinical_data)}"
                )
                print(f"\tNumber of reduced patient IDs: {mismatch_ids}")

                assert (
                    len(final_clinical_data)
                    == volume_est_cfg.NUMBER_TOTAL_CBTN_PATIENTS
                ), "Warning: The length of the filtered dataset is not 115. Check the csv again."

                self.clinical_data = final_clinical_data

            elif volume_est_cfg.BCH_DATA:
                clinical_data_file = volume_est_cfg.CLINICAL_DATA_FILE_BCH
                clinical_data = pd.read_csv(
                    clinical_data_file, sep=",", encoding="utf-8"
                )
                clinical_data["BCH MRN"] = (
                    clinical_data["BCH MRN"]
                    .astype(str)
                    .apply(prefix_zeros_to_six_digit_ids)
                )
                final_clinical_data = clinical_data[
                    clinical_data["BCH MRN"].isin(identifiers_in_dir)
                ]
                mismatch_ids = len(clinical_data) - len(final_clinical_data)
                print(
                    f"\tNumber of unique patient IDs in the original CSV: {len(clinical_data)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the directory: {len(identifiers_in_dir)}"
                )
                print(
                    f"\tNumber of unique patient IDs in the final CSV: {len(final_clinical_data)}"
                )
                print(f"\tNumber of reduced patient IDs: {mismatch_ids}")

                assert (
                    len(final_clinical_data) == volume_est_cfg.NUMBER_TOTAL_BCH_PATIENTS
                ), "Warning: The length of the filtered dataset is not 61. Check the csv again."

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
        except ExceptionGroup as error:
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
        
        volume_change = current - previous
        percentage = (volume_change / previous) * 100 if previous != 0 else 0
        return volume_change, percentage

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

        with Pool(cpu_count()) as pool:
            volumes = pool.map(self.estimate_volume, file_paths)

        all_scans = defaultdict(list)
        zero_volume_scans = defaultdict(list)
        for file_path, volume in zip(file_paths, volumes):
            basename = os.path.basename(file_path)
            patient_id, scan_id = self.get_identifier(basename)
            if volume_est_cfg.BCH_DATA:
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
                self.filtered_data = dict(
                    list(self.filtered_data.items())[:max_patients]
                )
            print(f"\tSet the max number of patient to be loaded to: {max_patients}")
            print(f"\tThe length of the filtered dataset is: {len(self.filtered_data)}")
            total_scans = sum(len(scans) for scans in self.filtered_data.values())
            print("\tTotal number of scans in the filtered dataset:", total_scans)
            average_scans = np.mean(
                [len(scans) for scans in self.filtered_data.values()]
            )
            print("\tAverage scan number per patient:", average_scans)
            median_scans = np.median(
                [len(scans) for scans in self.filtered_data.values()]
            )
            print("\tMedian scan number per patient:", median_scans)

        # Additional logic to process and store other states of data
        # (poly-smoothed, kernel-smoothed, window-smoothed)
        if volume_est_cfg.POLY_SMOOTHING:
            self.poly_smoothing_data = self.apply_polysmoothing()
            self.data_sources["poly_smoothing"] = self.poly_smoothing_data
            print("\tAdded polynomial smoothing data!")
        if volume_est_cfg.KERNEL_SMOOTHING:
            self.kernel_smoothing_data = self.apply_kernel_smoothing()
            self.data_sources["kernel_smoothing"] = self.kernel_smoothing_data
            print("\tAdded kernel smoothing data!")
        if volume_est_cfg.WINDOW_SMOOTHING:
            (
                self.window_smoothing_data,
                self.moving_average_data,
            ) = self.apply_sliding_window_interpolation()
            self.data_sources["window_smoothing"] = self.window_smoothing_data
            self.data_sources["moving_average"] = self.moving_average_data
            print("\tAdded sliding window smoothing and moving average data!")

        # Additionally, process the volume rate data
        self.volume_change_rate = self.calculate_volume_change_rate(self.filtered_data)
        print("\tAdded volume rate data!")

    def process_scans(self, all_scans):
        """
        Processes scan information to calculate volumes and optionally age.

        Args:
            all_scans (dict): Dictionary containing scan information.

        Returns:
            defaultdict(list): Processed scan information, including date,
            volume, and optionally age.
        """
        scan_dict = defaultdict(list)
        for patient_id, scans in all_scans.items():
            # print(patient_id)
            # print(scans)

            if volume_est_cfg.BCH_DATA:
                # Handling for BCH data
                patient_id = prefix_zeros_to_six_digit_ids(patient_id)
                for _, volume, scan_id in scans:
                    date = datetime.strptime(scan_id, "%Y%m%d")
                    dob = self.clinical_data.loc[
                        self.clinical_data["BCH MRN"] == patient_id, "Date of Birth"
                    ].iloc[0]
                    dob = datetime.strptime(
                        dob, "%d/%m/%Y"
                    )  # Adjust format if necessary
                    age = (date - dob).days
                    scan_dict[patient_id].append((date, volume, age))

            elif volume_est_cfg.CBTN_DATA:
                # Handling for CBTN data
                for _, volume, scan_id in scans:
                    age = int(scan_id)
                    scan_dict[patient_id].append((volume, age))

            elif volume_est_cfg.JOINT_DATA:
                for _, volume, scan_id in scans:
                    # print(patient_id)
                    # print(scan_id)
                    if len(scan_id) == 8:
                        date = datetime.strptime(scan_id, "%Y%m%d")
                        # print(date)
                        # print(self.clinical_data.loc[self.clinical_data["Patient_ID"] == patient_id].head(10))
                        dob = self.clinical_data.loc[
                            self.clinical_data["Patient_ID"] == patient_id,
                            "Date of Birth",
                        ].iloc[0]
                        # print(dob)
                        dob = datetime.strptime(dob, "%d/%m/%Y")
                        age = (date - dob).days
                        scan_dict[patient_id].append((volume, age))
                    else:
                        age = int(scan_id)
                        scan_dict[patient_id].append((volume, age))

            else:
                # Default handling for other data formats
                for scan in scans:
                    scan_dict[patient_id].append(scan)

        return scan_dict

    def apply_filtering(self, all_scans, zero_volume_scans, minimum_days=365):
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
        total_zero_volume_scans = 0
        total_patients_with_short_follow_up = 0
        total_scans_few_scans = 0
        total_scans_removed = 0
        few_scans_file_path = volume_est_cfg.FEW_SCANS_FILE
        zero_volume_file_path = volume_est_cfg.ZERO_VOLUME_FILE
        high_volume_file_path = volume_est_cfg.HIGH_VOLUME_FILE

        with open(zero_volume_file_path, "w", encoding="utf-8") as file:
            for patient_id, scans in zero_volume_scans.items():
                total_zero_volume_scans += len(scans)  # Increment the scan counter
                total_scans_removed += len(scans)
                file.write(f"{patient_id}\n")
                for scan_id in scans:
                    file.write(f"---- {scan_id}\n")
            file.write(f"\nTotal zero volume scans:{total_zero_volume_scans}\n")

        for patient_id in all_scans:
            all_scans[patient_id] = [
                scan for scan in all_scans[patient_id] if scan[1] != 0
            ]

        with open(high_volume_file_path, "w", encoding="utf-8") as high_vol_file:
            for patient_id, scans in all_scans.items():
                patient_id_written = False
                for scan in scans:
                    _, volume, scan_id = scan
                    if volume > volume_est_cfg.VOLUME_THRESHOLD:
                        if not patient_id_written:
                            high_vol_file.write(f"{patient_id}\n")
                            patient_id_written = True
                        high_vol_file.write(f"---- {scan_id}: {volume}\n")

        with open(few_scans_file_path, "w", encoding="utf-8") as file:
            for patient_id, scans in all_scans.items():
                if len(scans) >= 3:
                    # Sort scans by age to calculate the age range
                    sorted_scans = sorted(scans, key=lambda x: int(x[-1]))
                    age_range = int(sorted_scans[-1][-1]) - int(sorted_scans[0][-1])
                    if age_range >= minimum_days:
                        filtered_data[patient_id] = scans
                    else:
                        total_patients_with_short_follow_up += 1
                        total_scans_removed += len(scans)
                        file.write(f"{patient_id} - Short Follow-up: {len(scans)}\n")
                        for _, _, scan_id in scans:
                            file.write(f"---- {scan_id}\n")
                else:
                    total_patients_with_few_scans += 1  # Increment the patient counter
                    total_scans_few_scans += len(scans)  # Increment the scan counter
                    total_scans_removed += len(scans)
                    file.write(f"{patient_id} - Too Few Scans: {len(scans)}\n")
                    for _, _, scan_id in scans:
                        file.write(f"---- {scan_id}\n")
            file.write(
                f"\nTotal patients with too few scans: {total_patients_with_few_scans}\n"
            )
            file.write(
                "Total patients with short follow-up period:"
                f" {total_patients_with_short_follow_up}\n"
            )
            file.write(f"Total scans with too few scans: {total_scans_few_scans}\n")
            file.write(f"Total scans removed: {total_scans_removed}\n")

        return filtered_data

    def apply_polysmoothing(
        self,
        max_poly_degree=7,
    ):
        """
        Applies polynomial smoothing to the volume data, dynamically selecting
        polynomial degree based on the number of scans for each patient.

        Returns:
            defaultdict(list): Data after polynomial smoothing.
        """

        polysmoothed_data = defaultdict(list)
        for patient_id, scans in self.filtered_data.items():
            n_scans = len(scans)
            poly_degree = min(max(1, n_scans - 1), max_poly_degree)
            num_points = 25  # Number of points for interpolation
            scans.sort(key=lambda x: x[-1])  # Sort by age

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                volumes, ages = zip(*scans)
            else:
                dates, volumes, ages = zip(*scans)

            age_numbers = [float(age) for age in ages]
            poly_coeff = np.polyfit(age_numbers, volumes, poly_degree)
            poly_interp = np.poly1d(poly_coeff)

            start = min(age_numbers)
            end = max(age_numbers)
            interpolated_ages = np.linspace(start, end, num_points)
            smoothed_volumes = poly_interp(interpolated_ages)
            smoothed_volumes = np.maximum(smoothed_volumes, 0)

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                polysmoothed_data[patient_id] = list(zip(smoothed_volumes, ages))
            else:
                polysmoothed_data[patient_id] = list(zip(dates, smoothed_volumes, ages))

        return polysmoothed_data

    def apply_kernel_smoothing(self, bandwidth_factor=0.1):
        """
        Applies kernel smoothing to the volume data.

        Args:
            bandwidth (int, optional): The bandwidth for the Gaussian kernel. Defaults to 30.

        Returns:
            defaultdict(list): Data after kernel smoothing.
        """
        kernelsmoothed_data = defaultdict(list)

        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[-1])  # Sort by age
            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                volumes, ages = zip(*scans)
            else:
                dates, volumes, ages = zip(*scans)

            age_numbers = np.array([float(age) for age in ages])
            smoothed_volumes = np.zeros(len(volumes))
            volume_array = np.array(volumes)
            volume_iqr = np.subtract(*np.percentile(volume_array, [75, 25]))
            bandwidth = max(volume_iqr * bandwidth_factor, 1)

            # Apply kernel smoothing to volumes based on age
            for i, age in enumerate(age_numbers):
                age_diff = age_numbers - age
                weights = np.exp(-0.5 * (age_diff / bandwidth) ** 2)
                weighted_sum = np.sum(weights * volume_array)
                weight_total = np.sum(weights)

                smoothed_volume = (
                    weighted_sum / weight_total
                    if weight_total != 0
                    else volume_array[i]
                )
                smoothed_volumes[i] = smoothed_volume

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                kernelsmoothed_data[patient_id] = list(zip(smoothed_volumes, ages))
            else:
                kernelsmoothed_data[patient_id] = list(
                    zip(dates, smoothed_volumes, ages)
                )

        return kernelsmoothed_data

    def apply_sliding_window_interpolation(self, window_size=None):
        """
        Apply sliding window interpolation to smooth volumetric data of patients.

        This method applies a Gaussian-weighted median-based smoothing technique
        on a time series of scan volumes. The weighted median is calculated for
        each scan based on neighboring scans within a defined window.

        Parameters:
            window_size (int): Size of the window used to calculate the weighted
                            median. Should be an odd number for balanced
                            calculation. Default is None.

        Returns:
            dict: A dictionary containing the interpolated scan data for each
                patient, sorted by scan date.
        """
        weighted_median_data = defaultdict(list)
        moving_average_data = defaultdict(list)

        for patient_id, scans in self.filtered_data.items():
            scans.sort(key=lambda x: x[-1])  # Sort by age

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                volumes, ages = zip(*scans)
            else:
                # BCH data: (date, volume, age)
                dates, volumes, ages = zip(*scans)

            num_scans = len(volumes)
            window_size = window_size if window_size else max(3, num_scans // 2)
            volumes_series = pd.Series(volumes)
            smoothed_volumes = volumes_series.rolling(
                window=window_size, min_periods=1, center=True
            ).mean()

            for i in range(len(volumes)):
                # Define the window boundaries
                left = max(0, i - window_size // 2)
                right = min(num_scans, i + window_size // 2 + 1)
                window_volumes = volumes[left:right]

                distances = np.abs(np.arange(left, right) - i)
                weights = 1 / (1 + distances)
                weighted_vol = weighted_median(window_volumes, weights)

                if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                    weighted_median_data[patient_id].append((weighted_vol, ages[i]))
                    moving_average_data[patient_id].append(
                        (smoothed_volumes[i], ages[i])
                    )
                else:
                    weighted_median_data[patient_id].append(
                        (dates[i], weighted_vol, ages[i])
                    )
                    moving_average_data[patient_id].append(
                        (dates[i], smoothed_volumes[i], ages[i])
                    )

        return weighted_median_data, moving_average_data

    #############################
    # Output related functions  #
    #############################

    def analyze_volume_changes(self):
        """
        Analyze volume changes and compute the 95% confidence interval.
        """
        scan_data = self.filtered_data

        # Calculate volume changes for each patient
        change_rate = []
        for _, scans in scan_data.items():
            for i in range(1, len(scans)):
                prev_vol = scans[i - 1][1]  # Previous volume
                curr_vol = scans[i][1]  # Current volume
                change_rate.append(curr_vol - prev_vol)

        # Compute the 95% confidence interval
        lower, upper = compute_95_ci(change_rate)
        print(f"\t95% CI for volume change: ({lower:.2f}, {upper:.2f})")

        volume_change_rates = [
            rate[1]
            for sublist in self.volume_change_rate.values()
            for rate in sublist
            if rate[1] is not None
        ]

        lower, upper = compute_95_ci(volume_change_rates)
        # Calculate mean and standard deviation
        print(f"\t95% CI for volume change rate: {lower:.2f}, {upper:.2f}")

    def calculate_volume_change_rate(self, scans):
        """
        Calculates the rate of volume change (normalized by time span) for each patient.

        Args:
            all_scans (dict): Dictionary containing scan information.

        Returns:
            defaultdict(list): Processed scan information with rate of volume change and date.
        """
        rate_of_change_dict = defaultdict(list)

        for patient_id, patient_scans in scans.items():
            if volume_est_cfg.BCH_DATA:
                patient_id = prefix_zeros_to_six_digit_ids(patient_id)

            sorted_scans = sorted(patient_scans, key=lambda x: x[-1])
            prev_volume = None
            prev_time_point = None

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                for volume, time_point in sorted_scans:
                    if prev_time_point is not None:
                        days_diff = float(time_point) - float(prev_time_point)

                        volume_rate_of_change = (
                            (volume - prev_volume) / days_diff if days_diff != 0 else 0
                        )
                    else:
                        volume_rate_of_change = None

                    prev_volume = volume
                    prev_time_point = time_point

                    rate_of_change_dict[patient_id].append(
                        (time_point, volume_rate_of_change)
                    )
            else:
                for _, volume, time_point in sorted_scans:
                    if prev_time_point is not None:
                        days_diff = float(time_point) - float(prev_time_point)

                        volume_rate_of_change = (
                            (volume - prev_volume) / days_diff if days_diff != 0 else 0
                        )
                    else:
                        volume_rate_of_change = None

                    prev_volume = volume
                    prev_time_point = time_point

                    rate_of_change_dict[patient_id].append(
                        (time_point, volume_rate_of_change)
                    )

        return rate_of_change_dict

    def generate_csv(self, output_folder):
        """
        Generate a CSV file for each patient containing their time series volume data using pandas.

        Args:
            output_folder (str): Path to the directory where CSV files should be saved.
        """

        os.makedirs(output_folder, exist_ok=True)
        ts_poly = os.path.join(output_folder, "polynomial")
        ts_kernel = os.path.join(output_folder, "kernel")
        ts_window = os.path.join(output_folder, "window")
        ts_moving_average = os.path.join(output_folder, "moving_average")
        ts_filtered = os.path.join(output_folder, "filtered")
        os.makedirs(ts_poly, exist_ok=True)
        os.makedirs(ts_kernel, exist_ok=True)
        os.makedirs(ts_window, exist_ok=True)
        os.makedirs(ts_moving_average, exist_ok=True)
        os.makedirs(ts_filtered, exist_ok=True)

        mapping = {
            "polynomial": (self.poly_smoothing_data, ts_poly),
            "kernel": (self.kernel_smoothing_data, ts_kernel),
            "window": (self.window_smoothing_data, ts_window),
            "moving_average": (self.moving_average_data, ts_moving_average),
            "filtered": (self.filtered_data, ts_filtered),
        }
        for method, (data, folder) in mapping.items():
            for patient_id, volume_data in data.items():
                csv_file_path = os.path.join(folder, f"{patient_id}_{method}.csv")

                # Creating DataFrame from volume data
                df_columns = (
                    ["Volume", "Age"]
                    if (volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA)
                    else ["Date", "Volume", "Age"]
                )
                df = pd.DataFrame(volume_data, columns=df_columns)

                # Calculate baseline volume
                initial_volume = df["Volume"].iloc[0] if not df.empty else None
                df["Baseline Volume"] = initial_volume

                # Sorting by Age or Date
                sort_column = "Age" if not volume_est_cfg.TEST_DATA else "Date"
                df.sort_values(by=sort_column)

                # calculate age stats
                df["Age Median"] = df["Age"].median()
                df["Age Avg"] = df["Age"].mean()
                df["Age Std"] = df["Age"].std() 

                # Calculate volume stats
                df["Volume Median"] = df["Volume"].median()
                df["Volume Avg"] = df["Volume"].mean()
                df["Volume Std"] = df["Volume"].std()

                # Calculate normalized volume
                df["Normalized Volume"] = (
                    df["Volume"] / initial_volume if initial_volume else 0
                )
                df["Normalized Volume Median"] = df["Normalized Volume"].median()
                df["Normalized Volume Avg"] = df["Normalized Volume"].mean()
                df["Normalized Volume Std"] = df["Normalized Volume"].std()

                # Calculate volume change
                df["Volume Change"] = df["Volume"].diff()
                df["Volume Change Median"] = df["Volume Change"].median()
                df["Volume Change Avg"] = df["Volume Change"].mean()
                df["Volume Change Std"] = df["Volume Change"].std()
                df['Volume Change Pct'] = df['Volume'].pct_change() * 100
                df['Volume Change Pct Median'] = df['Volume Change Pct'].median()
                df['Volume Change Pct Avg'] = df['Volume Change Pct'].mean()
                df['Volume Change Pct Std'] = df['Volume Change Pct'].std() 

                # Add volume change rate
                df["Volume Change Rate"] = df["Age"].map(
                    lambda age, pid=patient_id: next(
                        (x[1] for x in self.volume_change_rate[pid] if x[0] == age),
                        None,
                    )
                )
                df["Volume Change Rate Median"] = df["Volume Change Rate"].median()
                df["Volume Change Rate Avg"] = df["Volume Change Rate"].mean()
                df["Volume Change Rate Std"] = df["Volume Change Rate"].std()
                df["Volume Change Rate Pct"] = df["Volume Change Rate"].pct_change() * 100
                df["Volume Change Rate Pct Median"] = df["Volume Change Rate Pct"].median()
                df["Volume Change Rate Pct Avg"] = df["Volume Change Rate Pct"].mean()
                df["Volume Change Rate Pct Std"] = df["Volume Change Rate Pct"].std()

                # Analyze change variables
                df["Change Speed"] = self.calculate_change_speed(df)
                df["Change Type"] = self.calculate_change_type(df)
                df["Change Trend"] = self.calculate_change_trend(df)
                df["Change Acceleration"] = self.calculate_change_acceleration(df)
                
                # Other variables
                df["Relative Volume Change Pct"] = self.calculate_relative_volume_change_pct(df)
                df["Cumulative Volume Change Pct"] = self.calculate_cumulative_relative_volume_change_pct(df)
                df["AUC"] = self.calculate_auc(df)
                df["Coefficient of Variation"] = self.calculate_volume_coefficient_of_variation(df)
                df["Rolling Volume Change Average"] = self.calculate_rolling_average_volume_change(df)
                
                
                if not volume_est_cfg.TEST_DATA:
                    df["Days Between Scans"] = df["Age"].diff()
                    if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                        df["Date"] = "N/A"
                        df["Scan_ID"] = df[
                            "Age"
                        ]
                    else:
                        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d/%m/%Y")
                        df["Scan_ID"] = pd.to_datetime(
                            df["Date"], format="%d/%m/%Y"
                        ).dt.strftime("%Y%m%d")

                # Reordering columns based on data type
                columns_order = (
                    [
                        "Scan_ID",
                        "Date",
                        "Age",
                        "Days Between Scans",
                        "Volume",
                        "Volume Median",
                        "Volume Avg",
                        "Volume Std",
                        "Normalized Volume",
                        "Normalized Volume Median",
                        "Normalized Volume Avg",
                        "Normalized Volume Std",
                        "Baseline Volume",
                        "Volume Change",
                        "Volume Change Median",
                        "Volume Change Avg",
                        "Volume Change Std",
                        "Volume Change Pct",
                        "Volume Change Pct Median",
                        "Volume Change Pct Avg",
                        "Volume Change Pct Std",
                        "Volume Change Rate",
                        "Volume Change Rate Median",
                        "Volume Change Rate Avg",
                        "Volume Change Rate Std",
                        "Volume Change Rate Pct",
                        "Volume Change Rate Pct Median",
                        "Volume Change Rate Pct Avg",
                        "Volume Change Rate Pct Std",
                        "Change Speed",
                        "Change Type",
                        "Change Trend",
                        "Change Acceleration",
                        "Relative Volume Change Pct",
                        "Cumulative Volume Change Pct",
                        "AUC",
                        "Coefficient of Variation",
                        "Rolling Volume Change Average",
                        
                        
                    ]
                    if not volume_est_cfg.TEST_DATA
                    else [
                        "Scan_ID",
                        "Date",
                        "Volume",
                        "Volume Median",
                        "Volume Avg",
                        "Volume Std",
                        "Normalized Volume",
                        "Normalized Volume Median",
                        "Normalized Volume Avg",
                        "Normalized Volume Std",
                        "Baseline Volume",
                        "Volume Change",
                        "Volume Change Median",
                        "Volume Change Avg",
                        "Volume Change Std",
                        "Volume Change Pct",
                        "Volume Change Pct Median",
                        "Volume Change Pct Avg",
                        "Volume Change Pct Std",
                        "Volume Change Rate",
                        "Volume Change Rate Median",
                        "Volume Change Rate Avg",
                        "Volume Change Rate Std",
                        "Volume Change Rate Pct",
                        "Volume Change Rate Pct Median",
                        "Volume Change Rate Pct Avg",
                        "Volume Change Rate Pct Std",
                        "Change Speed",
                        "Change Type",
                        "Change Trend",
                        "Change Acceleration",
                        "Relative Volume Change Pct",
                        "Cumulative Volume Change Pct",
                        "AUC",
                        "Coefficient of Variation",
                        "Rolling Volume Change Average",
                    ]
                )
                df = df[columns_order]

                # Export to CSV
                df.to_csv(csv_file_path, index=False)

    @staticmethod
    def calculate_change_speed(df):
        """Classify the change speed based on the average volume change rate."""

        avg_change_rate = df["Volume Change Rate Pct Avg"].iloc[0]

        if avg_change_rate < np.percentile(df["Volume Change Rate Pct"], 25):
            return "slow"
        elif avg_change_rate < np.percentile(df["Volume Change Rate Pct"], 75):
            return "moderate"
        return "rapid"

    @staticmethod
    def calculate_change_type(df):
        """
        Calculate the change type for a given patient based on the variability and pattern of volume change rates.
        """
        df_clean = df.dropna(subset=["Age", "Volume"])

        x = df_clean["Age"].values
        y = df_clean["Volume"].values

        # linear regression
        _, _, r_value, _, _ = linregress(x, y)
        r_squared = r_value**2
        
        # Classify based on best fit
        if r_squared >= 0.8:
            return "Linear"
        else:
            return "Non-linear"
            
    @staticmethod
    def calculate_change_trend(df):
        """
        Classify the trend of the volume change rate as increasing, decreasing, or stable.
        """
        trend = np.sign(np.mean(df['Volume Change Rate Avg']))
        if trend > 0:
            return "Increasing"
        elif trend < 0:
            return "Decreasing"
        else:
            return "Stable"
         
    @staticmethod
    def calculate_change_acceleration(df):
        """
        Calcualte acceleration of the volume.
        """
        df_clean = df.dropna(subset=["Age", "Volume"])

        x = df_clean["Age"].values.reshape(-1, 1)
        y = df_clean["Volume"].values.reshape(-1, 1)
        
        poly_features = PolynomialFeatures(degree=2)
        x_poly = poly_features.fit_transform(x)
        model = LinearRegression()
        model.fit(x_poly, y)
        quadratic_coef = model.coef_[0][2]
        
        # Classify based on best fit
        if quadratic_coef > 0:
            return "Increasing"
        else:
            return "Decreasing"
    
    @staticmethod
    def calculate_relative_volume_change_pct(df):
        """
        Calculates relative volume change based on the baseline volume.
        """
        baseline_volume = df["Baseline Volume"].iloc[0]
        relative_change = (df["Volume"] - baseline_volume) / baseline_volume * 100
        return relative_change
    
    @staticmethod
    def calculate_auc(df):
        """
        Calculates the area under the curve (AUC) for the volume data.
        """
        df_clean = df.dropna(subset=["Age", "Normalized Volume"])
        x = df_clean["Age"].values
        y = df_clean["Normalized Volume"].values
        auc_value = auc(x, y)
        return auc_value
    
    @staticmethod
    def calculate_volume_coefficient_of_variation(df):
        """"
        Calculates the coefficient of variation for the volume data.
        """
        mean_volume = df["Volume Avg"]
        std_volume = df["Volume Std"]
        cv = std_volume / mean_volume
        return cv
    
    @staticmethod
    def calculate_rolling_average_volume_change(df, window_size=3):
        """
        Calculate the rolling average of the volume change based on a specified window size.
        """
        rolling_avg_change = df["Volume Change"].rolling(window=window_size).mean()
        return rolling_avg_change
    
    @staticmethod
    def calculate_cumulative_relative_volume_change_pct(df):
        """
        Calculate the cumulative average relative volume change percentage.
        """
        baseline_volume = df["Baseline Volume"].iloc[0]
        relative_change_pct = (df["Volume"] - baseline_volume) / baseline_volume * 100
        cumulative_relative_change_pct = relative_change_pct.cumsum()
        return cumulative_relative_change_pct
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
        a_x1.set_xlabel("Age (days)", size=15)
        if normalize:
            a_x1.set_ylabel("Normalized Volume (mm³)", size=15)
        else:
            a_x1.set_ylabel("Volume (mm³)", size=15)

        a_x1.tick_params(axis="y", size=13)
        return fig, a_x1

    def add_volume_change_to_plot(self, a_x, ages, volumes):
        """
        Add volume change annotations to the plot.

        Args:
            a_x (matplotlib.axes.Axes): Axis object of the plot.
            dates (list): List of dates.
            volumes (list): List of volumes.
        """
        if ages is None:
            ages = list(range(len(volumes)))

        _ , volume_change_pct = zip(*[
            self.calculate_volume_change(volumes[i - 1], vol)
            for i, vol in enumerate(volumes[1:], 1)
        ])
        volume_change_pct = list(volume_change_pct)
        volume_change_pct.insert(0, 0)

        for i, (age, volume, volume_change) in enumerate(
            zip(ages, volumes, volume_change_pct)
        ):
            a_x.text(
                age,
                volume,
                f"{volume_change:.2f}%",
                fontsize=8,
                va="bottom",
                ha="left",
            )

    def add_date_to_plot(self, a_x, dates, ages):
        """
        Add date annotations to the plot.
        """
        if dates is not None and ages:
            a_x2 = a_x.twiny()
            a_x2.xaxis.set_ticks_position("top")
            a_x2.xaxis.set_label_position("top")
            a_x2.set_xlabel("Dates")
            a_x2.set_xlim(a_x.get_xlim())
            a_x2.set_xticks(ages)
            date_labels = [
                date.strftime("%d/%m/%Y") if isinstance(date, datetime) else date
                for date in dates
            ]
            a_x2.set_xticklabels(date_labels, rotation=90)
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
                print(f"\tPlotted {data_type} data!")

    def plot_each_type(self, data, output_path, data_type):
        """
        Plots the volumes data for a specific data type.

        Args:
            data (dict): Data to plot.
            output_path (str): The directory where plots should be saved.
            data_type (str): The type of data ('raw', 'filtered', etc.)
        """
        type_output_path = os.path.join(output_path, data_type)
        os.makedirs(type_output_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        for patient_id, volumes_data in data.items():
            volumes_data.sort(key=lambda x: x[-1])  # sort by age

            if volume_est_cfg.CBTN_DATA or volume_est_cfg.JOINT_DATA:
                # CBTN data: (volume, age)
                volumes, ages = zip(*volumes_data)
                self.plot_data(
                    data_type,
                    type_output_path,
                    patient_id,
                    None,
                    volumes,
                    ages,
                    has_dates=False,  # No dates in CBTN data
                )
                self.plot_normalized_data(
                    data_type,
                    type_output_path,
                    patient_id,
                    None,
                    volumes,
                    ages,
                    has_dates=False,  # No dates in CBTN data
                )
            else:
                # BCH data: (date, volume, age)
                dates, volumes, ages = zip(*volumes_data)
                self.plot_data(
                    data_type,
                    type_output_path,
                    patient_id,
                    dates,
                    volumes,
                    ages,
                    has_dates=True,
                )
                self.plot_normalized_data(
                    data_type,
                    type_output_path,
                    patient_id,
                    dates,
                    volumes,
                    ages,
                    has_dates=True,
                )

    def plot_data(
        self,
        data_type,
        output_path,
        patient_id,
        dates,
        volumes,
        ages=None,
        has_dates=True,
    ):
        """
        Plots and saves the volume data for a single patient.

        Args:
            data_type (str): The type of data ('raw', 'filtered', etc.)
            output_path (str): The directory where plots should be saved.
            patient_id (str): The ID of the patient.
            dates (list): List of dates.
            volumes (list): List of volumes.
            ages (list): List of ages.
        """
        os.makedirs(output_path, exist_ok=True)
        fig, a_x1 = self.setup_plot_base(normalize=False)

        if has_dates:
            dates, volumes, ages = zip(
                *sorted(zip(dates, volumes, ages), key=lambda x: x[2])
            )
            a_x1.plot(ages, volumes, color="tab:blue", marker="o")
            self.add_volume_change_to_plot(a_x1, ages, volumes)
            self.add_date_to_plot(a_x1, dates, ages)

        else:
            volumes, ages = zip(*sorted(zip(volumes, ages), key=lambda x: x[1]))
            a_x1.plot(ages, volumes, color="tab:blue", marker="o")
            self.add_volume_change_to_plot(a_x1, ages, volumes)

        plt.title(f"Patient ID: {patient_id} - {data_type}")
        fig.set_tight_layout(True)

        age_range = (
            f"{min(ages)}_{max(ages)}"  # Assuming ages is a list of float values
        )
        plt.savefig(
            os.path.join(
                output_path, f"volume_{data_type}_{patient_id}_{age_range}.png"
            )
        )
        plt.close(fig)

    def plot_normalized_data(
        self,
        data_type,
        output_path,
        patient_id,
        dates,
        volumes,
        ages=None,
        has_dates=True,
        window_size=None,
    ):
        """
        Plots and saves the normalized volume data for a single patient, signifying 25% change and shrinkage.
        """
        fig, ax1 = self.setup_plot_base(normalize=True)

        initial_volume = volumes[0] if volumes[0] not in [0, np.nan] else 1
        normalized_volumes = [v / initial_volume for v in volumes]

        if has_dates:
            dates, normalized_volumes, ages = zip(
                *sorted(zip(dates, normalized_volumes, ages), key=lambda x: x[2])
            )

        else:
            normalized_volumes, ages = zip(
                *sorted(zip(normalized_volumes, ages), key=lambda x: x[1])
            )

        ax1.plot(ages, normalized_volumes, color="tab:blue", marker="o", linestyle="-")
        self.add_volume_change_to_plot(ax1, ages, normalized_volumes)

        num_scans = len(normalized_volumes)
        window_size = window_size if window_size else max(3, num_scans // 2)
        moving_average = (
            pd.Series(normalized_volumes)
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        ax1.plot(
            ages,
            moving_average,
            color="tab:orange",
            linestyle="-",
            label="Moving Average",
        )

        algo = rpt.detection.Pelt(model="rbf").fit(np.array(normalized_volumes))
        result = algo.predict(pen=10)

        # Plot detected change points
        for cp in result:
            if cp < len(ages):
                ax1.axvline(x=ages[cp], color="red", linestyle="--")

        if has_dates:
            self.add_date_to_plot(ax1, dates, ages)

        age_range = f"{min(ages)}_{max(ages)}"
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best")
        plt.title(f"Patient ID: {patient_id} - Normalized Volume {data_type}")
        fig.set_tight_layout(True)
        plt.savefig(
            os.path.join(
                output_path,
                f"normalized_volume_{data_type}_{patient_id}_{age_range}.png",
            )
        )
        plt.close(fig)

    def plot_comparison(self, output_path):
        """
        Plots a comparison of all the data types for each unique patient ID.

        Args:
            output_path (str): The directory where the comparison plots should be saved.
        """
        unique_patient_ids = set(self.data_sources["filtered"].keys())

        for patient_id in unique_patient_ids:
            fig, axs = plt.subplots(1, len(self.data_sources), figsize=(24, 8))

            for i, (key, data) in enumerate(self.data_sources.items()):
                a_x = axs[i]

                patient_data = data.get(patient_id, [])

                if not patient_data:
                    a_x.set_title(f"No Data: {key}")
                    continue

                if len(patient_data[0]) == 3:
                    _, volumes, ages = zip(*patient_data)
                    a_x.plot(ages, volumes, label=f"{key} data")
                    a_x.set_xlabel("Date")
                elif len(patient_data[0]) == 2:
                    volumes, ages = zip(*patient_data)
                    a_x.plot(ages, volumes, label=f"{key} data")
                    a_x.set_xlabel("Age")

                a_x.set_title(f"{key} Data for Patient {patient_id}")
                a_x.set_ylabel("Volume")
                a_x.legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_path, f"volume_comparison_{patient_id}.png")
            )
            plt.close(fig)


if __name__ == "__main__":
    print("Initializing Volume Estimator:")
    ve = VolumeEstimator(volume_est_cfg.SEG_DIR)
    print("\tAdjusted the clinical data.")
    print("Processing files:")
    if volume_est_cfg.BCH_DATA:
        ve.process_files(max_patients=volume_est_cfg.NUMBER_TOTAL_BCH_PATIENTS)
    elif volume_est_cfg.CBTN_DATA:
        ve.process_files(max_patients=volume_est_cfg.NUMBER_TOTAL_CBTN_PATIENTS)
    elif volume_est_cfg.JOINT_DATA:
        ve.process_files(max_patients=volume_est_cfg.NUMBER_TOTAL_JOINT_PATIENTS)
    else:
        ve.process_files()
    print("\tAll files processed.")
    print("Generating plots:")
    ve.plot_volumes(output_path=volume_est_cfg.PLOTS_DIR)
    print("\tSaved all plots.")

    if volume_est_cfg.PLOT_COMPARISON:
        print("Generating volume comparison:")
        ve.plot_comparison(output_path=volume_est_cfg.PLOTS_DIR)
        print("\tSaved comparison.")

    if volume_est_cfg.CONFIDENCE_INTERVAL:
        print("Analyzing volume changes:")
        ve.analyze_volume_changes()
        print("\tAnalyzed volume changes.")

    print("Generating time-series csv's.")
    ve.generate_csv(output_folder=volume_est_cfg.CSV_DIR)
    print("\tSaved all csv's.")
