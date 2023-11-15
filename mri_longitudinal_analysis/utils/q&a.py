import os
import shutil
from tqdm import tqdm

# Directories containing the images and masks
images_folder1 = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/T2W_brain_extraction"
images_folder2 = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/T2W_brain_extraction"
masks_folder1 = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/seg_predictions"
masks_folder2 = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/seg_predictions"

# Target directory for the consolidated files
target_folder = (
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_longitudinal_dataset_new"
)

os.makedirs(target_folder, exist_ok=True)


# Function to extract patientID and scanID from a filename
def extract_ids(file_name, image=True, mask=False):
    parts = file_name.split("_")

    if image and not mask:
        return parts[0], parts[1]
    if mask and not image:
        scan_id = parts[1].split(".")[0]
        return parts[0], scan_id


# Dictionary to store matched pairs
matched_pairs = {}

# Read and match images from both image folders
for folder in [images_folder1, images_folder2]:
    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            patient_id, scan_id = extract_ids(filename, image=True, mask=False)
            matched_pairs[(patient_id, scan_id)] = [os.path.join(folder, filename), None]

print(f"Total matched pairs found: {len(matched_pairs)}")

# Match masks from both mask folders
for folder in [masks_folder1, masks_folder2]:
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
        image_target = os.path.join(target_folder, f"image{counter}_{patient_id}_{scan_id}.nii.gz")
        mask_target = os.path.join(
            target_folder, f"image{counter}_{patient_id}_{scan_id}_mask.nii.gz"
        )
        shutil.copyfile(image_path, image_target)
        shutil.copyfile(mask_path, mask_target)
        counter += 1
    else:
        print(
            f"Missing pair for patient_id={patient_id}, scan_id={scan_id}: image_path={image_path},"
            f" mask_path={mask_path}"
        )


print(f"Processed {counter - 1} image-mask pairs.")
