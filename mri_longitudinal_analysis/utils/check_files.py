"""
This module contains the CheckFiles class that provides various utilities for
file operations and validations. It includes methods to:
- Extract IDs from a list of filenames
- Find .csv filenames in a given directory
- Find subfolders in a given directory
- Compare two lists and return their symmetric difference

This class is designed to be part of a larger system for file management and
data validation. Its main functions include checking for discrepancies
between original and inferred files on the hard drive, and discrepancies
between files with generated plots and segmentations.

Author: Juan Carlos Climent Pardo
Date: Sep 19th 2023
Version: 1.0
"""

import csv
import os
import re

from mri_longitudinal_analysis.cfg import check_files_cfg


class CheckFiles:
    """
    A class to handle file operations and validations.
    """

    @staticmethod
    def extract_ids(filenames) -> list:
        """
        Extract IDs from a list of filenames.

        Parameters:
        filenames (list): A list of filenames to extract IDs from.

        Returns:
        list: A list of extracted IDs.
        """
        id_list = []
        for filename in filenames:
            #Search for a sequence of digits between underscores and
            #at the beginning of the filename.
            match = re.search(r"_(\d+)_", filename) or re.match(
                r"^(\d+)(?=[._])", filename
            )
            # print(f"Filename: {filename}, Match: {match}")
            if match:
                # print(f"Match group 1: {match.group(1)}")
                bch_id = match.group(1)
                if len(bch_id) == 6:
                    bch_id = "0" + bch_id
                id_list.append(bch_id)
            else:
                id_list.append(None)
        return id_list

    @staticmethod
    def find_csv_filenames(path_to_dir, suffix=".csv") -> list:
        """
        Find all .csv filenames in a given directory.

        Parameters:
        path_to_dir (str): Path to the directory to search.
        suffix (str): File extension to look for.

        Returns:
        list: A list of filenames with the given suffix.
        """
        filenames = os.listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    @staticmethod
    def find_subfolders(path_to_dir) -> list:
        """
        Find all subfolders in a given directory.

        Parameters:
        path_to_dir (str): Path to the directory to search.

        Returns:
        list: A list of subfolder names.
        """
        contents = os.listdir(path_to_dir)
        subfolders = [
            item if len(item) != 6 else "0" + item
            for item in contents
            if os.path.isdir(os.path.join(path_to_dir, item))
        ]
        return subfolders

    @staticmethod
    def compare_ids(list1, list2) -> list:
        """
        Compare two lists and return their symmetric difference.

        Parameters:
        list1, list2 (list): Lists to compare.

        Returns:
        list: List of items that are not common in both lists.
        """
        return list(set(list1) ^ set(list2))

    def check_files_harddrive(self):
        """
        Check discrepancies between original and inferred files on the hard drive.
        """
        curated_no_ops = check_files_cfg.ORIGINAL_NO_OPS_DIR
        csv_files_an225 = self.find_csv_filenames(curated_no_ops)
        subfolders_an225 = self.find_subfolders(curated_no_ops)
        ids_an225 = self.extract_ids(csv_files_an225)

        inference_path = check_files_cfg.INFERRED_NO_OPS_DIR
        csv_files_inf = self.find_csv_filenames(inference_path)
        ids_inf = self.extract_ids(csv_files_inf)

        diff_csv_files = self.compare_ids(ids_an225, ids_inf)
        diff_folders_csv_files = self.compare_ids(ids_an225, subfolders_an225)

        print("Number of csv files in an225 folder", len(csv_files_an225))
        print("Number of subfolders in an225 folder", len(subfolders_an225))
        print("Mismatching folders / csv's", diff_folders_csv_files)

        print("Number of parsed files", len(csv_files_inf))
        print("Number of different files", len(diff_csv_files))
        print("Mismatching csv's (aka not processed ones)", diff_csv_files)

    def check_files_segmentation(self):
        """
        Check discrepancies between files with generated plots and segmentations.
        """
        files60 = [
            f
            for f in os.listdir(check_files_cfg.VOLUMES_60_DIR)
            if os.path.isfile(os.path.join(check_files_cfg.VOLUMES_60_DIR, f))
        ]
        files29 = [
            f
            for f in os.listdir(check_files_cfg.VOLUMES_29_DIR)
            if os.path.isfile(os.path.join(check_files_cfg.VOLUMES_29_DIR, f))
        ]

        all_files = files60 + files29
        print("Number of files with a plot: ", len(all_files))

        extracted_ids = self.extract_ids(all_files)
        print("Number of extracted IDs from plots: ", len(extracted_ids))
        # print("List of extracted IDs from plots:", extracted_ids)

        csv_ids = []
        csv_file_path = check_files_cfg.REDCAP_FILE
        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                bch_id = row["BCH MRN"]
                if len(bch_id) == 6:
                    bch_id = "0" + bch_id

                csv_ids.append(bch_id)

        print("Number of csv IDs:", len(csv_ids))
        # print("List of extracted IDs from csv:", csv_ids)

        difference = self.compare_ids(extracted_ids, csv_ids)
        print(f"{len(difference)} files not having plots. Exact files:", difference)


if __name__ == "__main__":
    cf = CheckFiles()
    print("-----------------------------------------------------------------")
    print("Checking mismatching files between hard drive and processed ones!")
    print("-----------------------------------------------------------------")
    cf.check_files_harddrive()

    print("-----------------------------------------------------------------")
    print("Checking mismatching files between gen. plots and segmentations! ")
    print("-----------------------------------------------------------------")
    cf.check_files_segmentation()
