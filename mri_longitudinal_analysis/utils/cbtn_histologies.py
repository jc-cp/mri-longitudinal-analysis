"""This script is used to filter the clinical / histology data from the CBTN cohort."""
import os
import pandas as pd

def get_identifier(file_name):
    """Extracts patient ID and scan ID from the file name."""
    # file format are:
    # imageXYZ_patientID_scanID_mask.nii.gz
    # imageXYZ_patientID_scanID.nii.gz
    parts = file_name.split("_")
    return parts[1], parts[2]

def get_patient_ids(df_clinical, df_histologies, seg_dir) -> pd.DataFrame:
    """
    Get the patient IDs from the clinical data and 
    the histologies data that are present in the segmentation directory.
    """
    identifiers_in_dir = set()
    for file_name in os.listdir(seg_dir):
        if file_name.endswith("_mask.nii.gz"):
            patient_id, _ = get_identifier(file_name)
            identifiers_in_dir.add(patient_id)
    df_clinical["CBTN Subject ID"] = df_clinical["CBTN Subject ID"].astype(str)
    df_histologies["cohort_participant_id"] = df_histologies["cohort_participant_id"].astype(str)
    df_cl = df_clinical[df_clinical["CBTN Subject ID"].isin(identifiers_in_dir)]
    df_his = df_histologies[df_histologies["cohort_participant_id"].isin(identifiers_in_dir)].drop_duplicates("cohort_participant_id", keep='first')

    mismatch_ids_cl = len(df_clinical) - len(df_cl)
    mismatch_ids_his = len(df_histologies) - len(df_his)    
    print(f"\tNumber of unique patient IDs in the original clinical CSV: {len(df_clinical)}")
    print(f"\tNumber of unique patient IDs in the directory: {len(identifiers_in_dir)}")
    print(
        f"\tNumber of unique patient IDs in the final CSV: {len(df_cl)}"
    )
    print(f"\tNumber of reduced patient IDs: {mismatch_ids_cl}")
    print(f"\tNumber of unique patient IDs in the original histologies CSV: {len(df_histologies)}")
    print(f"\tNumber of unique patient IDs in the final histologies CSV: {len(df_his)}")
    print(f"\tNumber of reduced patient IDs: {mismatch_ids_his}")    
    assert len(identifiers_in_dir) == 115
    assert len(df_cl) == len(identifiers_in_dir)
    #assert len(df_his) == len(identifiers_in_dir)
    return df_cl, df_his, identifiers_in_dir

def export_ids_and_diagnosis_to_csv(df, filepath):
    """
    Exports patient IDs and pathology free text diagnoses to a new CSV file.
    
    Parameters:
    - df: pandas DataFrame containing patient IDs and 'pathology_free_text_diagnosis'.
    - filepath: String representing the path to the output CSV file.
    """
    columns_to_export = ['cohort_participant_id', 'pathology_free_text_diagnosis']
    if not set(columns_to_export).issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the following columns: {columns_to_export}")
    df_to_export = df[columns_to_export]
    df_to_export.to_csv(filepath, index=False)
    print(f"Data exported successfully to {filepath}")

def main():
    """Main function."""
    clinical_data = pd.read_csv('/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_filtered_and_pruned_513.csv')    
    histologies = pd.read_csv('/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_histologies.csv')
    seg_dir = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/qa"
    output_dir = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/clinical_data"
    
    _, histolog, _  = get_patient_ids(clinical_data, histologies, seg_dir)
    export_ids_and_diagnosis_to_csv(histolog, os.path.join(output_dir, "cbtn_histologies_data.csv"))
    

if __name__ == '__main__':
    main()