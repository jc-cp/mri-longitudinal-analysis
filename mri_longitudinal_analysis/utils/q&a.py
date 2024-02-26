"""
Q&A script for the MRI Longitudinal Analysis pipeline.
"""
import csv
import os
import re
import shutil
from collections import defaultdict

import nibabel as nib
import numpy as np
from cfg.utils import qa_cfg
from tqdm import tqdm


def move_file(src, dest):
    """Move a file from src to dest if it exists and hasn't been moved already."""
    if os.path.exists(src) and not os.path.exists(dest):
        shutil.move(src, dest)


def rename_image_file(file_path):
    """Renames the image file by toggling _0000 before the
    .nii.gz extension."""
    if file_path.endswith("_0000.nii.gz"):
        new_file_path = file_path.replace("_0000.nii.gz", ".nii.gz")
    elif file_path.endswith(".nii.gz"):
        new_file_path = file_path.replace(".nii.gz", "_0000.nii.gz")
    else:
        return file_path

    os.rename(file_path, new_file_path)
    return new_file_path


def rename_files_in_directory(directory):
    """Renames all .nii.gz files in the specified directory."""
    for file_name in os.listdir(directory):
        full_path = os.path.join(directory, file_name)
        if os.path.isfile(full_path):
            new_file_path = rename_image_file(full_path)
            print(f"Renamed '{full_path}' to '{new_file_path}'")


def extract_ids(file_name, image=True, mask=False):
    """
    Extracts and returns the IDs from a given filename.

    This function is designed to extract either the image or mask IDs from a filename,
    assuming the filename follows a specific format with parts separated by underscores.
    """
    parts = file_name.split("_")

    if image and not mask:
        return parts[0], parts[1]
    if mask and not image:
        scanid = parts[1].split(".")[0]
        return parts[0], scanid


def new_mask_exists(img_name, directory):
    """
    Checks if a new mask file exists for a given image in the specified directory.

    This function searches for a mask file that matches a specific pattern, indicating
    it is a new mask for the provided image name. The pattern includes the base name of the
    image followed by '_mask_' and a datetime hash.
    """
    base_name = img_name.split(".")[0]  # Extract base name without extension
    pattern = re.compile(
        re.escape(base_name) + r"_mask_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\.nii\.gz"
    )
    for file_name in os.listdir(directory):
        if pattern.match(file_name):
            return file_name
    return False


def rename_mask_files(directory):
    """
    Renames mask files in the specified directory.
    Removes the datetime hash part from the filename if present and
    adds '_mask' before .nii.gz if it's not present.

    :param directory: Directory containing the files.
    """
    pattern_hash = re.compile(r"(.+_mask_)\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}(\.nii\.gz)")
    pattern_mask = re.compile(r"(.+)(\.nii\.gz)")

    for img_name in os.listdir(directory):
        old_full_path = os.path.join(directory, img_name)

        if os.path.isfile(old_full_path):
            # Remove the datetime hash from the filename if it's present
            new_filename = pattern_hash.sub(r"\1\2", img_name)

            # Add '_mask' before '.nii.gz' if it's not present
            if "_mask" not in new_filename:
                new_filename = pattern_mask.sub(r"\1_mask\2", new_filename)

            new_full_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_full_path, new_full_path)
            print(f"Renamed '{old_full_path}' to '{new_full_path}'")


def analyze_nifti_files(directory, csv_file):
    """
    Analyzes NIfTI files in a directory to determine if they are empty or contain tumors.
    Saves the analysis in a CSV file, including the volume of the tumor.

    :param directory: Directory containing NIfTI files.
    :param csv_file: Path to the CSV file to save the results.
    """
    count = 0
    with open(csv_file, mode="w", newline="", encoding="utf-8") as archive:
        writer = csv.writer(archive)
        writer.writerow(["Filename", "Status", "Volume"])  # Writing the header

        for file_name in os.listdir(directory):
            if file_name.endswith(".nii.gz"):
                full_path = os.path.join(directory, file_name)
                nifti_image = nib.load(full_path)
                data = nifti_image.get_fdata()

                if np.any(data):  # Check if there is any non-zero voxel in the image
                    stats = "Tumor present"
                    volume = np.count_nonzero(data)  # Count non-zero voxels for volume
                    count += 1
                else:
                    stats = "Empty"
                    volume = 0

                # Write the filename, status, and volume to the CSV file
                writer.writerow([file_name, stats, volume])

                print(f"{file_name} - {stats} - {volume}")
    print(f"Total files with useful masks: {count}")


def move_tumor_files(annotations_csv, masks_folder, images_folder, accepted_folder, voxel_count=10):
    """
    Reads annotations.csv and moves corresponding .nii.gz files from masks and images folders
    to the accepted folder if the 'Status' column is 'Tumor Present'.

    :param annotations_csv: Path to the annotations.csv file.
    :param masks_folder: Directory containing mask files.
    :param images_folder: Directory containing image files.
    :param accepted_folder: Directory where files should be moved if accepted.
    """
    with open(annotations_csv, mode="r", encoding="utf-8") as archive:
        read = csv.DictReader(archive)
        for row_line in read:
            if row_line["Status"] == "Tumor present" and float(row_line["Volume"]) > voxel_count:
                img_name = row_line["Filename"]
                msk_name = img_name.replace(".nii.gz", "_mask.nii.gz")

                try:
                    # Move the corresponding mask file
                    mask_file = os.path.join(masks_folder, msk_name)
                    new_mask_file = os.path.join(accepted_folder, msk_name)
                    move_file(mask_file, new_mask_file)

                    # Move the corresponding image file
                    image_file = os.path.join(images_folder, img_name)
                    new_image_file = os.path.join(accepted_folder, img_name)
                    move_file(image_file, new_image_file)

                    # print(f"Moved '{mask_file}' to '{accepted_folder}'")
                    # print(f"Moved '{image_file}' to '{accepted_folder}'")
                except ExceptionGroup as e:
                    print(f"Error moving files: {e}")


if qa_cfg.PART_1:
    os.makedirs(qa_cfg.TARGET_FOLDER, exist_ok=True)

    # Dictionary to store matched pairs
    matched_pairs = {}

    # Read and match images
    folder = qa_cfg.IMAGES_FOLDER
    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            patient_id, scan_id = extract_ids(filename, image=True, mask=False)
            matched_pairs[(patient_id, scan_id)] = [os.path.join(folder, filename), None]

    print(f"Total matched pairs found: {len(matched_pairs)}")

    # Match masks
    folder = qa_cfg.MASKS_FOLDER
    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            patient_id, scan_id = extract_ids(filename, image=False, mask=True)
            print(f"Mask file: {filename}, patient_id: {patient_id}, scan_id: {scan_id}")
            if (patient_id, scan_id) in matched_pairs:
                matched_pairs[(patient_id, scan_id)][1] = os.path.join(folder, filename)
            else:
                print(f"Unmatched mask file: {filename}")

    matched_count = sum(1 for _, v in matched_pairs.items() if v[1] is not None)
    print(f"Total matched pairs found: {matched_count}")

    # Copy and rename the files
    COUNTER = 1
    for (patient_id, scan_id), (image_path, mask_path) in tqdm(
        matched_pairs.items(), desc="Processing files"
    ):
        if (mask_path and image_path) is not None:
            image_target = os.path.join(
                qa_cfg.TARGET_FOLDER, f"image{COUNTER}_{patient_id}_{scan_id}.nii.gz"
            )
            mask_target = os.path.join(
                qa_cfg.TARGET_FOLDER, f"image{COUNTER}_{patient_id}_{scan_id}_mask.nii.gz"
            )
            shutil.copyfile(image_path, image_target)
            shutil.copyfile(mask_path, mask_target)
            COUNTER += 1
        else:
            print(
                f"Missing pair for patient_id={patient_id}, scan_id={scan_id}:"
                f" image_path={image_path}, mask_path={mask_path}"
            )

    print(f"Processed {COUNTER - 1} image-mask pairs.")

if qa_cfg.PART_2:
    image_directory = qa_cfg.TARGET_FOLDER

    # Read the CSV file
    accepted_images = []
    rejected_images = []
    special_case_images = []
    masks_edited = []
    new_masks_images = []
    patient_acceptance_count = defaultdict(int)

    with open(qa_cfg.ANNOTATIONS_CSV, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in tqdm(reader, desc="Processing CSV file"):
            image_name, status, mask_status = row[0], row[1], row[4]
            old_mask_name = row[3] if len(row) > 3 else None

            patient_id = image_name.split("_")[1]
            new_mask_name = new_mask_exists(image_name, image_directory)

            if status.startswith("Acceptable"):
                patient_acceptance_count[patient_id] += 1
                if new_mask_name:
                    new_masks_images.append((image_name, old_mask_name, new_mask_name))
                elif mask_status == "Mask edited":
                    masks_edited.append((image_name, old_mask_name))
                else:
                    accepted_images.append((image_name, old_mask_name))

            # Handling rejected images
            elif status.startswith("Unacceptable") or status == "Bad images":
                if new_mask_name:
                    new_masks_images.append((image_name, old_mask_name, new_mask_name))
                    patient_acceptance_count[patient_id] += 1
                elif mask_status == "Mask edited":
                    masks_edited.append((image_name, old_mask_name))
                else:
                    if status == "Bad images" and patient_acceptance_count[patient_id] >= 4:
                        special_case_images.append((image_name, old_mask_name))
                    else:
                        rejected_images.append((image_name, old_mask_name))

    # Output results
    print("Accepted Images:", len(accepted_images))
    print("Rejected Images:", len(rejected_images))
    print("Special Case Images:", len(special_case_images))
    print("New masks Images:", len(new_masks_images))
    print("Masks edited:", len(masks_edited))

    if qa_cfg.MOVING_FILES:
        os.makedirs(qa_cfg.REJECTED_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.ACCEPTED_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEWS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_IMAGES_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_OLD_MASKS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_NEW_MASKS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REJECTED_MASKS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_MASKS_EDITED_FOLDER, exist_ok=True)

        # Move new mask images to the accepted folder
        for image_name, old_mask_name, new_mask_name in tqdm(
            new_masks_images, desc="Moving new mask images"
        ):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, new_mask_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, new_mask_name),
            )
            move_file(
                os.path.join(image_directory, old_mask_name),
                os.path.join(qa_cfg.REJECTED_MASKS_FOLDER, old_mask_name),
            )
        # Move the rest of the rejected images to the rejected folder
        for image_name, mask_name in tqdm(rejected_images, desc="Moving rejected images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.REJECTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.REJECTED_FOLDER, mask_name),
            )
        # Move the rest of the accepted images to the accepted folder
        for image_name, mask_name in tqdm(accepted_images, desc="Moving accepted images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, mask_name),
            )

        for image_name, mask_name in tqdm(special_case_images, desc="Moving special case images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.REVIEW_IMAGES_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.REVIEW_OLD_MASKS_FOLDER, mask_name),
            )

        for image_name, mask_name in tqdm(masks_edited, desc="Moving masks edited images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.REVIEW_MASKS_EDITED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.REVIEW_MASKS_EDITED_FOLDER, mask_name),
            )

if qa_cfg.PART_3:
    review_directory = qa_cfg.REVIEW_IMAGES_FOLDER
    rename_files_in_directory(review_directory)

if qa_cfg.PART_4:
    analyze_nifti_files(qa_cfg.REVIEW_NEW_MASKS_FOLDER, qa_cfg.ANNOTATIONS_CSV_MASKS)

if qa_cfg.PART_5:
    rename_mask_files(qa_cfg.ACCEPTED_FOLDER)

if qa_cfg.PART_6:
    os.makedirs(qa_cfg.SECOND_REVIEW, exist_ok=True)
    
    move_tumor_files(
        qa_cfg.ANNOTATIONS_CSV_MASKS,
        qa_cfg.REVIEW_NEW_MASKS_FOLDER,
        qa_cfg.REVIEW_IMAGES_FOLDER,
        qa_cfg.SECOND_REVIEW,
        qa_cfg.VOXEL_COUNT,
    )

if qa_cfg.PART_7:  
    image_directory = qa_cfg.SEG_REVIEWS_FOLDER
    # Read the CSV file
    accepted_images = []
    rejected_images = []
    new_masks_images = []
    patient_acceptance_count = defaultdict(int)

    with open(qa_cfg.ANNOTATIONS_FINAL, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in tqdm(reader, desc="Processing CSV file"):
            image_name, status, mask_status = row[0], row[1], row[4]
            old_mask_name = row[3] if len(row) > 3 else None

            patient_id = image_name.split("_")[1]
            new_mask_name = new_mask_exists(image_name, image_directory)

            if status.startswith("Acceptable"):
                patient_acceptance_count[patient_id] += 1
                if new_mask_name:
                    new_masks_images.append((image_name, old_mask_name, new_mask_name))
                else:
                    accepted_images.append((image_name, old_mask_name))

            # Handling rejected images
            elif status.startswith("Unacceptable") or status == "Bad images":
                if new_mask_name:
                    new_masks_images.append((image_name, old_mask_name, new_mask_name))
                    patient_acceptance_count[patient_id] += 1
                else:
                    rejected_images.append((image_name, old_mask_name))

    # Output results
    print("Accepted Images:", len(accepted_images))
    print("Rejected Images:", len(rejected_images))
    print("New masks Images:", len(new_masks_images))

    if qa_cfg.FINAL_MOVE:
        for image_name, old_mask_name, new_mask_name in tqdm(
            new_masks_images, desc="Moving new mask images"
        ):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, new_mask_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, new_mask_name),
            )
            move_file(
                os.path.join(image_directory, old_mask_name),
                os.path.join(qa_cfg.REJECTED_MASKS_FOLDER, old_mask_name),
            )
        # Move the rest of the rejected images to the rejected folder
        for image_name, mask_name in tqdm(rejected_images, desc="Moving rejected images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.REJECTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.REJECTED_FOLDER, mask_name),
            )
        # Move the rest of the accepted images to the accepted folder
        for image_name, mask_name in tqdm(accepted_images, desc="Moving accepted images"):
            move_file(
                os.path.join(image_directory, image_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, image_name),
            )
            move_file(
                os.path.join(image_directory, mask_name),
                os.path.join(qa_cfg.ACCEPTED_FOLDER, mask_name),
            )
