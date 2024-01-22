"""
Creates a basic script that outputs the necessary csv-file for the BRAF inference.
"""
import os
import csv

# Directory containing the files
DIR = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_longitudinal_dataset_new/accepted"

# Prepare the CSV file
CSV_NAME = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/cbtn_patients_scans.csv"
header = ["pat_id", "scandate", "label"]

# Open the CSV file for writing
with open(CSV_NAME, "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)

    # Iterate over files in the directory
    for filename in os.listdir(DIR):
        if filename.endswith(".nii.gz") and "_mask" not in filename:
            # Extract patient ID and scan ID from the filename
            parts = filename.split("_")
            patient_id = parts[1]
            scan_id = parts[2].split(".")[0]

            # Write the information to the CSV file
            csvwriter.writerow([patient_id, scan_id, "3"])

print(f"Data written to {CSV_NAME}")
