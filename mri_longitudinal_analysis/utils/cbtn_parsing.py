"""
Parser script to extract data from the CBTN dataset.
"""
import os
import re
import shutil
from collections import Counter
import pandas as pd
import nibabel as nib
from cfg.utils import cbtn_parsing_cfg


def read_csv(path_cbtn_csv, path_cbtn_img) -> (list, list):
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
        print("No duplicate IDs!")

    print("Total number of patients:", len(ids))

    ages = []
    for patient_id in ids:
        for i, patient_item in enumerate(df_cbtn["CBTN Subject ID"]):
            if patient_id == patient_item:
                ages.append(df_cbtn.loc[i, "Age at Event Days"])

    # assert len(ages) == len(ids), "Ages and IDs length assertion failed"

    path_mr_cbtn = []
    for patient_id in ids:
        for patient_item in os.listdir(path_cbtn_img):
            if patient_id == patient_item:
                path_mr_cbtn.append(os.path.join(path_cbtn_img, patient_item))

    # assert len(path_mr_cbtn) == len(ids), "Path and IDs length assertion failed"

    return ages, path_mr_cbtn


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


def get_sessions_and_patients(path_mr_cbtn, ages) -> (list, list):
    """
    Create DataFrames with session path, age at session, and patient ID.

    Returns:
    - df_pre_event: dataframe of pre-event session / patients
    - df_post_event: dataframe of post-event session / patients
    """
    pre_event_sessions = []
    post_event_sessions = []

    for path, age in zip(path_mr_cbtn, ages):
        patient_id = extract_patient_id(path)
        if patient_id is None:
            print("Patient ID not found!")
            continue

        for folder in os.listdir(path):
            if "brain" in folder:
                session_age = extract_session(folder)
                if session_age is not None:
                    if session_age <= int(age):
                        pre_event_sessions.append(
                            [patient_id, session_age, os.path.join(path, folder), age]
                        )
                    else:
                        post_event_sessions.append(
                            [patient_id, session_age, os.path.join(path, folder), age]
                        )

    dtype_mapping = {
        "Patient_ID": "string",
        "Session Age": "int",
        "Session Path": "string",
        "Age at Event Days": "int",
    }

    df_pre_event = pd.DataFrame(
        pre_event_sessions,
        columns=["Patient_ID", "Session Age", "Session Path", "Age at Event Days"],
    )
    df_post_event = pd.DataFrame(
        post_event_sessions,
        columns=["Patient_ID", "Session Age", "Session Path", "Age at Event Days"],
    )

    for column, dtype in dtype_mapping.items():
        df_pre_event[column] = df_pre_event[column].astype(dtype)
        df_post_event[column] = df_post_event[column].astype(dtype)

    df_pre_event = df_pre_event.sort_values(by="Patient_ID")
    df_post_event = df_post_event.sort_values(by="Patient_ID")

    return df_pre_event, df_post_event


def get_t2_sequences(df_event, meta_data):
    """
    Function to get the T2 sequences from the metadata file.
    """
    metadata_df = pd.read_csv(meta_data)
    matched_files = []

    for _, row in df_event.iterrows():
        patient_id = row["Patient_ID"]
        session_age = row["Session Age"]

        condition = (
            (metadata_df["subject_label"] == patient_id)
            & (metadata_df["age_at_imaging_in_days"] == session_age)
            & (
                metadata_df["fw_class"].str.contains("Structural")
                & metadata_df["fw_class"].str.contains("T2")
            )
        )

        filtered_df = metadata_df[condition]

        three_d_entries = filtered_df[filtered_df["fw_class"].str.contains("3D")]
        if not three_d_entries.empty:
            three_d_entries["Resolution"] = three_d_entries.apply(
                lambda x: get_resolution(os.path.join(row["Session Path"], x["acquisition_label"])),
                axis=1,
            )
            three_d_entries_sorted = three_d_entries.sort_values(
                by=["Resolution", "acquisition_label"], ascending=[False, True]
            )
            selected_entry = (
                three_d_entries_sorted.iloc[0]
                if three_d_entries_sorted.shape[0] == 1
                else three_d_entries_sorted.iloc[1]
            )

        else:
            # Fallback criteria for non-3D entries
            filtered_df["Resolution"] = filtered_df.apply(
                lambda x: get_resolution(os.path.join(row["Session Path"], x["acquisition_label"])),
                axis=1,
            )
            sorted_df = filtered_df.sort_values(
                by=["Resolution", "acquisition_label"], ascending=[False, True]
            )
            selected_entry = (
                sorted_df.iloc[0]
                if "axial" in sorted_df.iloc[0]["acquisition_label"].lower()
                else sorted_df.iloc[1]
            )

        acquisition_label = selected_entry["acquisition_label"]
        acq_path = os.path.join(row["Session Path"], acquisition_label)
        matched_files.append((patient_id, session_age, acquisition_label, acq_path))

    return pd.DataFrame(
        matched_files,
        columns=["Patient_ID", "Session Age", "Acquisition Label", "Acquisition Path"],
    )


def get_resolution(file_path):
    """
    Get the resolution of an MRI image using nibabel.

    Parameters:
    file_path (str): Path to the MRI image file.

    Returns:
    tuple: Resolution of the image.
    """
    try:
        img = nib.load(file_path)
        header = img.header
        # The voxel dimensions are usually in the header under 'pixdim'
        resolution = header.get_zooms()[:3]  # Get the first three dimensions (for 3D images)
        return resolution
    except FileNotFoundError as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def count_nan(lst):
    count = 0
    for item in lst:
        if isinstance(item, list):  # Check if it is a list
            count += count_nan(item)  # Recursively count in the sublist
        elif item == "nan":
            count += 1
    return count


def copy_images(path_mr_cbtn, session_label, seqs, new_path_cbtn):
    nan_count = count_nan(seqs)
    print(nan_count)

    for i, item in enumerate(path_mr_cbtn):
        if len(seqs[i]) != 0:
            print(item, session_label[i], seqs[i][0])
            image = os.listdir(os.path.join(item, session_label[i], seqs[i][0]))
            shutil.copy(
                os.path.join(item, session_label[i], seqs[i][0], image[0]),
                os.path.join(new_path_cbtn, f"{item[27:]}.nii.gz"),
            )
        else:
            pass


def main():
    """
    Main function for parsing the cbtn dataset.
    """
    path_cbtn_csv = cbtn_parsing_cfg.PATH_CSV
    path_cbtn_img = cbtn_parsing_cfg.PATH_IMAGES
    ages, path_mr_cbtn = read_csv(path_cbtn_csv, path_cbtn_img)
    print(len(ages), len(path_mr_cbtn))

    df_pre_event, df_post_event = get_sessions_and_patients(path_mr_cbtn, ages)
    print(df_pre_event.dtypes)

    meta_path = cbtn_parsing_cfg.PATH_METADATA

    new_path_cbtn = cbtn_parsing_cfg.NEW_PATH_IMAGES


if __name__ == "__main__":
    main()
