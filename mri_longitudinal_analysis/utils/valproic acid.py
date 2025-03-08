import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/jc053/GIT/mri_longitudinal_analysis/data/output/02_trajectories/JOINT_trajectories_cohort_data_features.csv')

# Filter for DF_BCH dataset and get unique Patient_IDs with their classifications
filtered_df = df[df['Dataset'] == 'DF_BCH'].groupby('Patient_ID').first().reset_index()

# Verify we have 56 entries
assert len(filtered_df) == 56, f"Expected 56 entries, but got {len(filtered_df)}"

# Select only the columns we need
result_df = filtered_df[['Patient_ID', 'Dataset', 'Patient Classification Volumetric']]

print(result_df.head(15))

# Export to new CSV
result_df.to_csv('/home/jc053/GIT/mri_longitudinal_analysis/data/output/02_trajectories/bch_patient_classifications.csv', index=False)
