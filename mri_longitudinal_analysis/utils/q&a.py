import os
import shutil
from tqdm import tqdm
from cfg import qa_cfg
from collections import defaultdict
import csv
import re


def move_file(src, dest):
    """Move a file from src to dest if it exists and hasn't been moved already."""
    if os.path.exists(src) and not os.path.exists(dest):
        shutil.move(src, dest)


def rename_image_file(file_path):
    """Renames the image file by adding _0000 before the .nii.gz extension."""
    if file_path.endswith(".nii.gz"):
        new_file_path = file_path.replace(".nii.gz", "_0000.nii.gz")
        os.rename(file_path, new_file_path)
        return new_file_path
    else:
        return file_path


def rename_files_in_directory(directory):
    """Renames all .nii.gz files in the specified directory."""
    for file_name in os.listdir(directory):
        full_path = os.path.join(directory, file_name)
        if os.path.isfile(full_path):
            new_file_path = rename_image_file(full_path)
            print(f"Renamed '{full_path}' to '{new_file_path}'")


# Function to extract patientID and scanID from a filename
def extract_ids(file_name, image=True, mask=False):
    parts = file_name.split("_")

    if image and not mask:
        return parts[0], parts[1]
    if mask and not image:
        scan_id = parts[1].split(".")[0]
        return parts[0], scan_id


# Function to check if a new mask exists for a given image
def new_mask_exists(image_name, directory):
    base_name = image_name.split(".")[0]  # Extract base name without extension
    pattern = re.compile(
        re.escape(base_name) + r"_mask_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\.nii\.gz"
    )
    for file_name in os.listdir(directory):
        if pattern.match(file_name):
            return file_name
    return False


if qa_cfg.PART_1:
    os.makedirs(qa_cfg.TARGET_FOLDER, exist_ok=True)

    # Dictionary to store matched pairs
    matched_pairs = {}

    # Read and match images from both image folders
    for folder in [qa_cfg.IMAGES_FOLDER_1, qa_cfg.IMAGES_FOLDER_2]:
        for filename in os.listdir(folder):
            if filename.endswith(".nii.gz"):
                patient_id, scan_id = extract_ids(filename, image=True, mask=False)
                matched_pairs[(patient_id, scan_id)] = [os.path.join(folder, filename), None]

    print(f"Total matched pairs found: {len(matched_pairs)}")

    # Match masks from both mask folders
    for folder in [qa_cfg.MASKS_FOLDER_1, qa_cfg.MASKS_FOLDER_2]:
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
    counter = 1
    for (patient_id, scan_id), (image_path, mask_path) in tqdm(
        matched_pairs.items(), desc="Processing files"
    ):
        if (mask_path and image_path) is not None:
            image_target = os.path.join(
                qa_cfg.TARGET_FOLDER, f"image{counter}_{patient_id}_{scan_id}.nii.gz"
            )
            mask_target = os.path.join(
                qa_cfg.TARGET_FOLDER, f"image{counter}_{patient_id}_{scan_id}_mask.nii.gz"
            )
            shutil.copyfile(image_path, image_target)
            shutil.copyfile(mask_path, mask_target)
            counter += 1
        else:
            print(
                f"Missing pair for patient_id={patient_id}, scan_id={scan_id}:"
                f" image_path={image_path}, mask_path={mask_path}"
            )

    print(f"Processed {counter - 1} image-mask pairs.")

if qa_cfg.PART_2:
    image_directory = qa_cfg.TARGET_FOLDER

    # Read the CSV file
    accepted_images = []
    rejected_images = []
    special_case_images = []
    new_masks_images = []
    patient_acceptance_count = defaultdict(int)

    with open(qa_cfg.ANNOTATIONS_CSV, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in tqdm(reader, desc="Processing CSV file"):
            image_name, status = row[0], row[1]
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
                    if status == "Bad images" and patient_acceptance_count[patient_id] >= 4:
                        special_case_images.append((image_name, old_mask_name))
                    else:
                        rejected_images.append((image_name, old_mask_name))

    # Output results
    print("Accepted Images:", len(accepted_images))
    print("Rejected Images:", len(rejected_images))
    print("Special Case Images:", len(special_case_images))
    print("New masks Images:", len(new_masks_images))

    if qa_cfg.MOVING_FILES:
        os.makedirs(qa_cfg.REJECTED_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.ACCEPTED_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEWS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_IMAGES_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REVIEW_MASKS_FOLDER, exist_ok=True)
        os.makedirs(qa_cfg.REJECTED_MASKS_FOLDER, exist_ok=True)

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
                os.path.join(qa_cfg.REVIEW_MASKS_FOLDER, mask_name),
            )

if qa_cfg.PART_3:
    review_directory = qa_cfg.REVIEW_IMAGES_FOLDER
    rename_files_in_directory(review_directory)
