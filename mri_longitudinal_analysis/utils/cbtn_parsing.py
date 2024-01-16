"""
Parser script to extract data from the CBTN dataset.
"""
import os
import shutil
from collections import Counter
import re

import nibabel as nib
import pandas as pd
from cfg.utils import cbtn_parsing_cfg
from tqdm import tqdm


def read_csv(path_cbtn_csv, path_cbtn_img) -> (list, list, list):
    """
    Read the csv file and return the ages and the path to the MR images.
    It includes assertations to make sure that the csv is already preprocessed.

    Args:
    - path_cbtn_csv: path to the csv file
    - path_cbtn_img: path to the images folder

    Returns:
    - ages: list of ages of the patients at the time of the event
    - path_mr_cbtn: list of paths to the folders of the MR images of patients
    """
    df_cbtn = pd.read_csv(path_cbtn_csv)

    # Add here filtering and assertation criteria
    if cbtn_parsing_cfg.ASSERTATIONS:
        assert (df_cbtn["Event Type"] == "Initial CNS Tumor").all(), "Event Type assertion failed"

        assert (
            df_cbtn["Flywheel Imaging Available"] == "Yes"
        ).all(), "Flywheel Imaging Available assertion failed"

        assert (
            df_cbtn["Diagnoses"]
            .isin(
                [
                    "Low-grade glioma/astrocytoma (WHO grade I/II)"
                    or "Ganglioglioma"
                    or "Glial-neuronal tumor NOS"
                    or "Low-grade glioma/astrocytoma (WHO grade I/II)"
                    or "Ganglioglioma; Low-grade glioma/astrocytoma (WHO grade I/II)"
                ]
            )
            .any()
        ), "Diagnoses assertion failed"

        assert (
            df_cbtn["Follow Up Event"] == "Initial Diagnosis"
        ).all(), "Follow Up Event assertion failed"

        assert df_cbtn["Age at Event Days"].notna().all(), "Age at Event Days assertion failed"

        assert (
            df_cbtn["Specimen Collection Origin"]
            .isin(["Initial CNS Tumor Surgery" or "Not Applicable"])
            .any()
        ), "Specimen Collection Origin assertion failed"

        assert (
            len(df_cbtn) == cbtn_parsing_cfg.LENGTH_DATA
        ), (  # this is fix from previous filtering in the csv file
            "DataFrame length assertation failed"
        )

    ids = sorted(df_cbtn["CBTN Subject ID"].tolist())
    id_counts = Counter(ids)

    duplicates = [id for id, count in id_counts.items() if count > 1]
    if duplicates:
        print("Duplicate IDs:", duplicates)
    else:
        print("No duplicate IDs in the clinical csv!")

    print("Total number of patients in clinical data:", len(ids))

    ages = []
    surgery_status = []
    for patient_id in ids:
        for i, patient_item in enumerate(df_cbtn["CBTN Subject ID"]):
            if patient_id == patient_item:
                ages.append(df_cbtn.loc[i, "Age at Event Days"])
                surgery = df_cbtn.loc[i, "Surgery"]
                surgery_status.append("Yes" if surgery == "Yes" else "No")

    assert len(ages) == len(ids) and len(surgery_status) == len(ids), "Length assertion failed"

    path_mr_cbtn = []
    for patient_id in ids:
        for patient_item in os.listdir(path_cbtn_img):
            if patient_id == patient_item:
                path_mr_cbtn.append(os.path.join(path_cbtn_img, patient_item))

    assert len(path_mr_cbtn) == len(ids), "Path and IDs length assertion failed"

    return ages, surgery_status, path_mr_cbtn


def extract_session(s):
    """
    Function to extract the number before 'd' in the string.
    """
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def extract_patient_id(path):
    """
    Extract the patient ID from the path.
    """
    match = re.search(r"C\d{5,7}", path)
    return match.group() if match else None


def get_sessions_and_patients(path_mr_cbtn, ages, surgery_status) -> (list, list):
    """
    Create DataFrames with session path, age at session, and patient ID.

    Returns:
    - df_pre_event: dataframe of pre-event session / patients
    - df_post_event: dataframe of post-event session / patients
    """
    sessions = []

    for path, age, surgery in zip(path_mr_cbtn, ages, surgery_status):
        patient_id = extract_patient_id(path)
        if patient_id is None:
            print("Patient ID not found!")
            continue

        for folder in os.listdir(path):
            session_age = extract_session(folder)
            if session_age is not None:
                if surgery == "Yes" and session_age <= int(age):
                    sessions.append(
                        [patient_id, session_age, os.path.join(path, folder), age, "pre"]
                    )
                elif surgery == "No":
                    sessions.append(
                        [patient_id, session_age, os.path.join(path, folder), age, "pre"]
                    )
                else:
                    sessions.append(
                        [patient_id, session_age, os.path.join(path, folder), age, "post"]
                    )

    dtype_mapping = {
        "Patient_ID": "string",
        "Session Age": "int",
        "Session Path": "string",
        "Age at Event Days": "int",
        "Event Type": "string",
    }

    df_sessions = pd.DataFrame(
        sessions,
        columns=["Patient_ID", "Session Age", "Session Path", "Age at Event Days", "Event Type"],
    )

    for column, dtype in dtype_mapping.items():
        df_sessions[column] = df_sessions[column].astype(dtype)

    df_sessions = df_sessions.sort_values(by="Patient_ID")

    df_pre_event = df_sessions[df_sessions["Event Type"] == "pre"]
    df_post_event = df_sessions[df_sessions["Event Type"] == "post"]

    return df_pre_event, df_post_event


def align_dataframes(df_event, meta_data, suffix):
    """
    Function to align two dataframes.
    """
    metadata_df = pd.read_csv(meta_data).sort_values(by=["subject_label", "session_label"])
    metadata_df = metadata_df.sort_values(by=["subject_label", "age_at_imaging_in_days"])
    df_event = df_event.sort_values(by=["Patient_ID", "Session Age"])

    metadata_ids = metadata_df["subject_label"].unique()
    print("Total number of patients in imaging data: ", len(metadata_ids))
    df_event_ids = df_event["Patient_ID"].unique()
    print(f"Number of patients in {suffix}-event clinical data:", len(df_event_ids))
    print("Starting dataframe alignement...")

    if len(metadata_ids) > len(df_event_ids):
        mismatch = set(metadata_ids) - set(df_event_ids)
        print("\tIDs in metadata but not in in the clinical data.")
        print(f"\tThere is a mismatch of {len(mismatch)} ids.")
        # Filter metadata_df to include only rows where 'subject_label' is in df_event_ids
        metadata_df = metadata_df[metadata_df["subject_label"].isin(df_event_ids)]
        metadata_ids = metadata_df["subject_label"].unique()
        df_event_ids = df_event["Patient_ID"].unique()

    if len(metadata_ids) < len(df_event_ids):
        mismatch = set(df_event_ids) - set(metadata_ids)
        print("\tIDs in clinical subset but no imagining available according to the metadata.")
        print(f"\tThere is a mismatch of {len(mismatch)} ids.")
        # Filter out these IDs from df_event
        df_event = df_event[~df_event["Patient_ID"].isin(mismatch)]
        metadata_ids = metadata_df["subject_label"].unique()
        df_event_ids = df_event["Patient_ID"].unique()

    assert len(df_event_ids) == len(metadata_ids), "IDs length assertion failed"
    assert set(df_event_ids) == set(metadata_ids), "IDs equality assertion failed"
    print(f"\tNumber of total available patients in clinical AND imaging data: {len(metadata_ids)}")

    return df_event, metadata_df


def get_t2_sequences(df_event_aligned, df_metadata_aligned):
    """
    Gets the T2 sequences from the dataframes and creates
    the correct paths upon inspection of quality.
    """
    print("Starting T2 sequence extraction...")
    matched_files = []

    for _, row in df_event_aligned.iterrows():
        patient_id = row["Patient_ID"]
        session_age = row["Session Age"]
        session_path = row["Session Path"]
        last_part_session_path = os.path.basename(session_path)

        condition = (
            (df_metadata_aligned["subject_label"] == patient_id)
            & (df_metadata_aligned["age_at_imaging_in_days"] == session_age)
            & (
                df_metadata_aligned["fw_class"].str.contains("Structural")
                & df_metadata_aligned["fw_class"].str.contains("T2")
            )
            & (df_metadata_aligned["session_label"] == last_part_session_path)
        )

        filtered_df = df_metadata_aligned[condition].sort_values(
            by=["subject_label", "session_label", "acquisition_label"]
        )

        selected_entry = {}
        for _, x in filtered_df.iterrows():
            acquisition_label = x["acquisition_label"]
            full_path = os.path.join(session_path, acquisition_label)

            if os.path.exists(full_path) and (
                os.path.isdir(full_path) or os.path.isfile(full_path)
            ):
                if "3D" in x["fw_class"]:
                    resolution_product = get_resolution(full_path)
                    if (
                        selected_entry is None
                        or resolution_product > selected_entry["Resolution_Product"]
                    ):
                        selected_entry = x
                        selected_entry["Resolution_Product"] = resolution_product
                elif selected_entry is None:
                    selected_entry = x
                    selected_entry["Resolution_Product"] = get_resolution(full_path)

        if selected_entry is not None:
            acquisition_label = selected_entry["acquisition_label"]
            acq_path = os.path.join(session_path, acquisition_label)
            matched_files.append((patient_id, session_age, acquisition_label, acq_path))
            print(
                f"Patient {patient_id} at session age {session_age} has an image"
                f" {acquisition_label}."
            )
        else:
            print(
                f"No valid T2 entries found for patient {patient_id} at session age {session_age}"
            )

    return pd.DataFrame(
        matched_files,
        columns=["Patient_ID", "Session Age", "Acquisition Label", "Acquisition Path"],
    )


def get_resolution(folder_path):
    """
    Get the resolution of an MRI image using nibabel.

    Parameters:
    file_path (str): Path to the MRI image file.

    Returns:
    tuple: Resolution of the image.
    """

    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        if os.path.isfile(file_path):
            try:
                img = nib.load(file_path)
                header = img.header
                # The voxel dimensions are usually in the header under 'pixdim'
                resolution = header.get_zooms()[
                    :3
                ]  # Get the first three dimensions (for 3D images)

                if resolution:
                    return resolution
            except FileNotFoundError as e:
                print(f"Error loading file {file_path}: {e}")
                return None
        else:
            print(f"Error loading file {file_path}: Not a file.")
            return None
    else:
        print(f"Error loading folder {folder_path}: Not a directory.")
        return None


def copy_images(csv_file_path, new_path_cbtn):
    """
    Copies files from the dataframe to a new folder.
    """
    df = pd.read_csv(csv_file_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        acquisition_folder = row["Acquisition Path"]

        if os.path.isdir(acquisition_folder):
            files_in_folder = os.listdir(acquisition_folder)

            # Check if there is exactly one file in the folder
            if len(files_in_folder) == 1:
                file_to_copy = files_in_folder[0]
                source_path = os.path.join(acquisition_folder, file_to_copy)
                dest_file_name = f"{row['Patient_ID']}_{row['Session Age']}.nii.gz"
                dest_path = os.path.join(new_path_cbtn, dest_file_name)

                # Copy the file
                shutil.copyfile(source_path, dest_path)
            elif len(files_in_folder) == 0:
                print(f"\tWarning: No files found in folder: {acquisition_folder}")
            else:
                print(
                    f"\tWarning: Multiple files found in folder {acquisition_folder}, skipping"
                    " copy."
                )
        else:
            print(f"\t Warning: Folder not found: {acquisition_folder}")


def main():
    """
    Main function for parsing the cbtn dataset.
    """
    pd.set_option("display.max_colwidth", None)
    path_cbtn_csv = cbtn_parsing_cfg.PATH_CLINICAL_CSV
    metadata_path = cbtn_parsing_cfg.PATH_METADATA_CSV
    path_cbtn_img = cbtn_parsing_cfg.PATH_IMAGES

    if cbtn_parsing_cfg.PARSING:
        # get list of ages from patients and a list of all paths
        # corresponding to the relevant patients throught the ids
        ages, surgery_status, path_mr_cbtn = read_csv(path_cbtn_csv, path_cbtn_img)

        # get the individual sessions of patients based on event age
        df_pre_event, df_post_event = get_sessions_and_patients(path_mr_cbtn, ages, surgery_status)

        # remove the missing patients from the metadata and clinical data
        df_event_aligned_pre, df_metadata_aligned_pre = align_dataframes(
            df_pre_event, metadata_path, suffix="pre"
        )

        df_event_aligned_post, df_metadata_aligned_post = align_dataframes(
            df_post_event, metadata_path, suffix="post"
        )

        df_pre_event_matched = get_t2_sequences(df_event_aligned_pre, df_metadata_aligned_pre)
        df_pre_event_matched.to_csv(cbtn_parsing_cfg.OUTPUT_CSV_PRE_EVENT)

        df_post_event_matched = get_t2_sequences(df_event_aligned_post, df_metadata_aligned_post)
        df_post_event_matched.to_csv(cbtn_parsing_cfg.OUTPUT_CSV_POST_EVENT)

    if cbtn_parsing_cfg.MOVING:
        csv_path_pre = cbtn_parsing_cfg.OUTPUT_CSV_PRE_EVENT
        csv_path_post = cbtn_parsing_cfg.OUTPUT_CSV_POST_EVENT
        new_path_cbtn_pre = cbtn_parsing_cfg.NEW_PATH_IMAGES_PRE
        new_path_cbtn_post = cbtn_parsing_cfg.NEW_PATH_IMAGES_POST
        os.makedirs(new_path_cbtn_pre, exist_ok=True)
        os.makedirs(new_path_cbtn_post, exist_ok=True)

        print("Copying pre-event images...")
        copy_images(csv_path_pre, new_path_cbtn_pre)
        print("Copying post-event images...")
        copy_images(csv_path_post, new_path_cbtn_post)


if __name__ == "__main__":
    main()
