import fnmatch
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm

sys.path.append(sys.path.append(str(Path(__file__).resolve().parent.parent)))

from cfg import review_cfg


class Review:
    def __init__(self):
        self.csv_files = {"60": review_cfg.CSV_FILE_60, "29": review_cfg.CSV_FILE_29}

    def move_files(df, condition, source_dir, destination_folder):
        # Define function to move files based on condition once review
        filenames = df[condition]["Image_Name"]
        for filename in tqdm(filenames, desc="Moving files"):
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_folder, filename)
            if os.path.isfile(source_path):
                shutil.copyfile(source_path, destination_path)
            else:
                print(f"File {source_path} does not exist.")

    def read_file(self, csv_file):
        # Define column names and load
        column_names = ["Image_Name", "Quality", "Comments"]
        df = pd.read_csv(csv_file, names=column_names)
        return df

    def create_review_dirs(self, suffix):
        dirs = {
            "reviewed_dir": review_cfg.__dict__[f"REVIEWED_FOLDER_{suffix}"],
            "dir1_no_comments": review_cfg.__dict__[f"DIR1_NO_COMMENTS_{suffix}"],
            "dir1_with_comments": review_cfg.__dict__[f"DIR1_WITH_COMMENTS_{suffix}"],
            "dir5": review_cfg.__dict__[f"DIR5_{suffix}"],
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        dirs["csv_file_after_review"] = dirs["dir1_with_comments"] / "annotations.csv"

        print("All directories created!")
        return dirs

    def detect_t2(self, data_folders, output_detection_folders):
        for data_folder, output_detection in zip(
            data_folders, output_detection_folders
        ):
            detected_path = os.path.join(data_folder, output_detection)
            os.makedirs(detected_path, exist_ok=True)

            # Define the list of sub-directories to scan
            sub_dirs = ["T1", "T1c", "FLAIR", "OTHER"]

            for sub_dir in sub_dirs:
                source_dir = os.path.join(data_folder, sub_dir)

                for filename in os.listdir(source_dir):
                    if "t2" in filename.lower() and "flair" not in filename.lower():
                        source_file = os.path.join(source_dir, filename)
                        dest_file = os.path.join(detected_path, filename)
                        shutil.copyfile(source_file, dest_file)

        print("Detection completed!")

    def move_valid_images(self, source_dir, dest_dir, csv_file):
        """
        Moves files from source_dir to dest_dir based on csv_file
        :param source_dir: source directory path
        :param dest_dir: destination directory path
        :param csv_file: csv file path
        """
        # Define column names and load
        column_names = ["Image_Name", "Quality", "Comments"]
        df = pd.read_csv(csv_file, names=column_names)

        # Filter rows where Quality is 1
        valid_images = df[(df["Quality"] == 1)]["Image_Name"]

        for filename in tqdm(valid_images, desc="Moving valid images"):
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(dest_dir, filename)
            if os.path.isfile(source_path):
                shutil.move(source_path, destination_path)
            else:
                if os.path.isfile(destination_path):
                    print(f"File {source_path} was already moved.")
                else:
                    print(
                        f"File {source_path} does not exist in the source directory and was not moved previously."
                    )

    def get_patient_scan_id(self, filename, delimiter="_"):
        """Extracts patient ID and scan ID from filename"""
        split_filename = filename.split(delimiter)
        patient_id = split_filename[0]
        scan_id = split_filename[1] if len(split_filename) > 1 else None
        return patient_id, scan_id

    def compare_ids_and_flag(self, t2_dir):
        """
        Compares multiple versions of the same scan for each patient, flags the one with the highest resolution,
        and moves the other versions to a 'lower resolution' directory.
        :param t2_dir: Path to the T2 directory.
        """
        # Gather all file paths in the T2 directory
        file_paths = list(t2_dir.glob("*.nii.gz"))

        # Group files by patient ID and scan ID
        patient_scan_files = {}
        for file_path in file_paths:
            patient_id, scan_id = self.get_patient_scan_id(
                file_path.stem.replace(".nii", "")
            )
            key = (patient_id, scan_id)
            if key not in patient_scan_files:
                patient_scan_files[key] = []
            patient_scan_files[key].append(file_path)

        # Create 'lower resolution' directory if it doesn't exist
        low_res_dir = t2_dir / "lower_resolution"
        os.makedirs(low_res_dir, exist_ok=True)

        # Compare resolution for each scan's versions and flag the highest resolution one
        for (patient_id, scan_id), file_paths in patient_scan_files.items():
            if len(file_paths) > 1:  # only consider scans with more than one version
                max_res = 0
                max_res_file = None
                for file_path in file_paths:
                    img = nib.load(str(file_path))
                    axial_res = img.header["pixdim"][
                        1:3
                    ].prod()  # Axial resolution is the product of the first two pixel dimensions
                    if axial_res > max_res:
                        max_res = axial_res
                        max_res_file = file_path
                print(
                    f"Highest resolution version for patient {patient_id}, scan {scan_id} is {max_res_file}"
                )

                # Move all other versions to the 'lower resolution' directory
                for file_path in file_paths:
                    if file_path != max_res_file:
                        shutil.move(str(file_path), str(low_res_dir))

    def remove_artifacts(self, directory):
        artifact_dir = directory / "Artifacts"
        os.makedirs(artifact_dir, exist_ok=True)

        patients_with_artifacts = review_cfg.PATIENTS_WITH_ARTIFACTS

        for root, dirnames, filenames in os.walk(directory):
            for patient in patients_with_artifacts:
                for filename in fnmatch.filter(filenames, f"{patient}*"):
                    source_path = os.path.join(root, filename)
                    destination_path = os.path.join(artifact_dir, filename)
                    if os.path.isfile(source_path):
                        shutil.move(source_path, destination_path)
                        print(
                            f"Moved artifact patient with file {filename} to {destination_path}"
                        )

    def add_zero_id(self, patient_id):
        """Add an initial 0 to the patient ID if it only contains 6 digits instead of 7"""
        if len(patient_id) == 6:
            return "0" + patient_id
        else:
            return patient_id
            pass

    def main(self):
        for suffix, csv_file in self.csv_files.items():
            df = self.read_file(csv_file=csv_file)
            dirs = self.create_review_dirs(suffix)

            # Detect files containing a t2 string in the modality folders
            detecting = review_cfg.DETECTING
            if detecting:
                data_folders = [review_cfg.DATA_FOLDER_60, review_cfg.DATA_FOLDER_29]
                output_detection_folders = [
                    review_cfg.OUTPUT_DETECTTION_60,
                    review_cfg.OUTPUT_DETECTTION_29,
                ]
                self.detect_t2(data_folders, output_detection_folders)

            # Moving files if needed for initial review
            moving2review = review_cfg.MOVING_2_REVIEW
            if moving2review:
                source_dir = dirs["reviewed_dir"]
                dir1_no_comments = dirs["dir1_no_comments"]
                dir1_with_comments = dirs["dir1_with_comments"]
                dir5 = dirs["dir5"]

                # Move the files
                try:
                    self.move_files(
                        df,
                        (df["Quality"] == 1) & (df["Comments"] == "Valid Images"),
                        source_dir,
                        dir1_no_comments,
                    )
                    self.move_files(
                        df,
                        (df["Quality"] == 1) & (df["Comments"] != "Valid Images"),
                        source_dir,
                        dir1_with_comments,
                    )
                    self.move_files(df, df["Quality"] == 5, source_dir, dir5)
                except IOError as e:
                    print(e)
                    pass

            # Moving files after radiologist review
            moving4dataset = review_cfg.MOVING_4_DATASET
            if moving4dataset:
                source_dir = dirs["dir1_with_comments"]
                dest_dir = dirs["dir1_no_comments"]
                csv_file_after_review = dirs["csv_file_after_review"]

                self.move_valid_images(source_dir, dest_dir, csv_file_after_review)
                print("Moved images for full dataset!")

        # Moving files containing massive artifacts
        delete_artifacts = review_cfg.DELETE_ARTIFACTS
        if delete_artifacts:
            print("Removing artifacts!")
            for t2_dir in [
                review_cfg.DIR1_NO_COMMENTS_60,
                review_cfg.DIR1_NO_COMMENTS_29,
            ]:
                self.remove_artifacts(t2_dir)

        compare_ids = review_cfg.COMPARE_IDS
        if compare_ids:
            print("Comparing IDs!")
            for t2_dir in [
                review_cfg.DIR1_NO_COMMENTS_60,
                review_cfg.DIR1_NO_COMMENTS_29,
            ]:
                self.compare_ids_and_flag(t2_dir)


if __name__ == "__main__":
    rw = Review()
    rw.main()
