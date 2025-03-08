"""
Data curation script to create a joint cohort of patients. It takes care of data loading, processing, and merging.
"""

import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from cfg.src import cohort_creation_cfg
from cfg.utils.helper_functions_cfg import NORD_PALETTE
from utils.helper_functions import zero_fill, categorize_age_group, categorize_time_since_first_diagnosis, save_dataframe, check_assumptions

class CohortCreation:
    
    def __init__(self, data_paths_, output_dir, cohort):
        """
        Initialize the TumorAnalysis class.

        Parameters:
            clinical_data_file (str): Path to the clinical data CSV file.
            volumes_data_file (str): Path to the tumor volumes data CSV file.
        """
        pd.options.display.float_format = "{:.3f}".format
        self.merged_data = pd.DataFrame()
        self.clinical_data_reduced = pd.DataFrame()
        self.sample_size_plots = cohort_creation_cfg.SAMPLE_SIZE
        self.cohort = cohort
        print("Step 0: Initializing CohortCreation class...")


        patient_ids_volumes = self.load_volumes_data(data_paths_["volumes_data_paths"])
        if self.cohort == "JOINT":
            self.validate_files(list(data_paths_["clinical_data_paths"].values()), data_paths_["volumes_data_paths"])
            self.load_clinical_data(
                data_paths_["clinical_data_paths"], patient_ids_volumes
            )
        else:
            self.validate_files(data_paths_["clinical_data_paths"], data_paths_["volumes_data_paths"])
            if self.cohort == "DF_BCH":
                _ = self.load_clinical_data_bch(
                    data_paths_["clinical_data_paths"][0], patient_ids_volumes
                )
            elif self.cohort == "CBTN":
                _ = self.load_clinical_data_cbtn(
                    data_paths_["clinical_data_paths"][0], patient_ids_volumes
                )
        self.merge_data()
        self.check_data_consistency()
        self.merged_data = self.sort_df(self.merged_data)
        if self.merged_data.isnull().values.any():
            self.check_dtypes_and_nan()
            self.merged_data.replace(np.nan, np.inf, inplace=True)

        save_dataframe(df=self.merged_data, cohort=f"{self.cohort.lower()}", output_dir=output_dir)

    def validate_files(self, clinical_data_paths, volumes_data_paths):
        """
        Check if the specified clinical data and volume data files exist.

        Parameters:
        - clinical_data_paths (dict): Dictionary containing paths to clinical data files for each cohort.
        - volumes_data_paths (list): List containing paths to volume data directories for each cohort.

        Raises:
        - FileNotFoundError: If any of the files specified do not exist.

        Prints a validation message if all files exist.
        """
        missing_files = []

        for clinical_data_path in clinical_data_paths:
            if not os.path.exists(clinical_data_path):
                missing_files.append(clinical_data_path)

        for volumes_data_path in volumes_data_paths:
            if not os.path.exists(volumes_data_path):
                missing_files.append(volumes_data_path)

        if missing_files:
            raise FileNotFoundError(
                f"The following files could not be found: {missing_files}"
            )
        print("\tValidated files.")

    def map_dictionary(self, dictionary, column, map_type):
        """
        Maps given instance according to predefined dictionary.
        """

        def map_value(cell):
            for keyword, value in dictionary.items():
                if keyword.lower() in str(cell).casefold():
                    return value
                
            if map_type == "location":
                return "Other"
            if map_type == "symptoms":
                return "Asymptomatic (Incidentally Found)"
            if map_type == "histology":
                return "Other"
            else:
                return None
                        
        return column.apply(map_value)

    def load_clinical_data_bch(self, clinical_data_path, patient_ids_volumes):
        """
        Load clinical data from a CSV file, parse the clinical data to
        categorize diagnosesand other relevant fields to reduce the data for analysis.
        Updates the `self.clinical_data_reduced` attribute.

        Parameters:
        - clinical_data_path (str): Path to the clinical data file.

        The function updates the `self.clinical_data` attribute with processed data.
        """
        self.clinical_data = pd.read_csv(clinical_data_path)
        print(f"\tOriginal clinical data has length {len(self.clinical_data)}.")

        self.clinical_data["BCH MRN"] = zero_fill(self.clinical_data["BCH MRN"], 7)

        self.clinical_data["Location"] = self.map_dictionary(
            cohort_creation_cfg.BCH_LOCATION,
            self.clinical_data["Location of Tumor"],
            map_type="location",
        )

        self.clinical_data["Symptoms"] = self.map_dictionary(
            cohort_creation_cfg.BCH_SYMPTOMS,
            self.clinical_data["Symptoms at diagnosis"],
            map_type="symptoms",
        )

        self.clinical_data["Histology"] = self.map_dictionary(
            cohort_creation_cfg.BCH_GLIOMA_TYPES,
            self.clinical_data["Pathologic diagnosis"],
            map_type="histology",
        )

        self.clinical_data["Sex"] = self.clinical_data["Sex"].apply(
            lambda x: "Female" if x == "Female" else "Male"
        )

        self.clinical_data["BRAF Status"] = self.clinical_data.apply(
            lambda row: "V600E"
            if row["BRAF V600E mutation"] == "Yes"
            else ("Fusion" if row["BRAF fusion"] == "Yes" else "Wildtype"),
            axis=1,
        )

        self.clinical_data["Date of Birth"] = pd.to_datetime(
            self.clinical_data["Date of Birth"], dayfirst=True
        )

        self.clinical_data["Date First Diagnosis"] = pd.to_datetime(
            self.clinical_data["MRI Date"], dayfirst=True
        )

        self.clinical_data["Age at First Diagnosis"] = pd.to_numeric((
            self.clinical_data["Date First Diagnosis"]
            - self.clinical_data["Date of Birth"]
        ).dt.days, errors='coerce')

        self.clinical_data["Date of last clinical follow-up"] = pd.to_datetime(
            self.clinical_data["Date of last clinical follow-up"], dayfirst=True
        )

        self.clinical_data["Age at Last Clinical Follow-Up"] = pd.to_numeric((self.clinical_data["Date of last clinical follow-up"]
                - self.clinical_data["Date of Birth"]).dt.days, errors='coerce')

        self.clinical_data["Date First Treatment"] = pd.to_datetime(
            self.clinical_data["First Treatment"], dayfirst=True
        )

        self.clinical_data["Age at First Treatment"] = pd.to_numeric((self.clinical_data["Date First Treatment"]
            - self.clinical_data["Date of Birth"]).dt.days, errors='coerce')
            
        self.clinical_data["Received Treatment"] = (
            self.clinical_data["Age at First Treatment"]
            .notna()
            .map({True: "Yes", False: "No"})
        )

        self.clinical_data["Treatment Type"] = self.extract_treatment_types_bch()

        self.clinical_data["Follow-Up Time"] = np.where(
            self.clinical_data["Follow-Up"].notna(),
            self.clinical_data["Follow-Up"],
            (self.clinical_data["Age at Last Clinical Follow-Up"]
            - self.clinical_data["Age at First Diagnosis"]),
        )

        # Apply the type conversions according to the dictionary
        for column, dtype in cohort_creation_cfg.BCH_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        all_relevant_columns = (
            list(cohort_creation_cfg.BCH_DTYPE_MAPPING.keys())
            + cohort_creation_cfg.BCH_DATETIME_COLUMNS
        )
        self.clinical_data_reduced = self.clinical_data[all_relevant_columns].copy()
        self.clinical_data_reduced["BCH MRN"] = (
            self.clinical_data_reduced["BCH MRN"].astype(str).str.zfill(7)
        )
        self.clinical_data_reduced = self.clinical_data_reduced[
            self.clinical_data_reduced["BCH MRN"].isin(patient_ids_volumes)
        ]
        print(f"\tFiltered clinical data has length {len(self.clinical_data_reduced)}.")

        self.clinical_data_reduced["Dataset"] = "DF_BCH"
        print("\tParsed clinical data.")
        clinical_data_bch = self.clinical_data_reduced
        return clinical_data_bch

    def load_clinical_data_cbtn(self, clinical_data_path, patient_ids_volumes):
        """
        Load clinical data from a CBTN CSV file, parse the clinical data to
        categorize diagnoses and other relevant fields to reduce the data for analysis.
        Updates the `self.clinical_data_reduced` attribute for CBTN data.

        Parameters:
        - clinical_data_path (str): Path to the clinical data file.
        """
        self.clinical_data = pd.read_csv(clinical_data_path)
        print(f"\tOriginal CBTN clinical data has length {len(self.clinical_data)}.")

        # Map CBTN columns to BCH equivalent
        self.clinical_data["CBTN Subject ID"] = self.clinical_data[
            "CBTN Subject ID"
        ].astype(str)

        # Map Location
        self.clinical_data["Location"] = self.map_dictionary(
            cohort_creation_cfg.CBTN_LOCATION,  # Define a suitable mapping dictionary
            self.clinical_data["Tumor Locations"],
            map_type="location",
        )

        # Map symptoms
        self.clinical_data["Symptoms"] = self.map_dictionary(
            cohort_creation_cfg.CBTN_SYMPTOMS,
            self.clinical_data["Medical Conditions Present at Event"],
            map_type="symptoms",
        )

        # Map Sex
        self.clinical_data["Sex"] = self.clinical_data["Legal Sex"].apply(
            lambda x: "Female" if x == "Female" else "Male"
        )

        # Map Histology
        self.clinical_data["Histology"] = self.map_dictionary(
            cohort_creation_cfg.CBTN_GLIOMA_TYPES,
            self.clinical_data["Diagnoses"],
            map_type="histology",
        )

        # Age Last Clinical Follow Up
        self.clinical_data["Age at Last Clinical Follow-Up"] = pd.to_numeric(self.clinical_data[
            "Age at Last Known Clinical Status"], errors="coerce")

        # BRAF Status
        self.clinical_data["BRAF Status"] = self.clinical_data[
            "OpenPedCan Molecular Subtype"
        ].apply(
            lambda x: "V600E"
            if "V600E" in str(x)
            else ("Fusion" if "1549" in str(x) else "Wildtype")
        )

        # Treatment Type, Received Treatment, Time to treatment and Age at First Treatment
        self.extract_treatment_info_cbtn()

        # Apply type conversions - Define CBTN_DTYPE_MAPPING as per CBTN data structure
        for column, dtype in cohort_creation_cfg.CBTN_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        # Select relevant columns for reduced data
        all_relevant_columns = (
            list(cohort_creation_cfg.CBTN_DTYPE_MAPPING.keys())
            + cohort_creation_cfg.CBTN_DATETIME_COLUMNS
        )

        self.clinical_data_reduced = self.clinical_data[all_relevant_columns].copy()

        self.clinical_data_reduced = self.clinical_data_reduced[
            self.clinical_data_reduced["CBTN Subject ID"].isin(patient_ids_volumes)
        ]
        print(
            f"\tFiltered CBTN clinical data has length {len(self.clinical_data_reduced)}."
        )

        self.clinical_data_reduced["Dataset"] = "CBTN"
        print("\tParsed CBTN clinical data.")
        clinical_data_cbtn = self.clinical_data_reduced
        return clinical_data_cbtn

    def load_clinical_data(self, clinical_data_paths, patient_ids_volumes):
        """
        Load and process clinical data from both BCH and CBTN files.
        """
        bch_clinical_data = pd.DataFrame()
        cbtn_clinical_data = pd.DataFrame()
        if "df_bch" in clinical_data_paths:
            bch_clinical_data = self.load_clinical_data_bch(
                clinical_data_paths["df_bch"], patient_ids_volumes
            )
        if "cbtn" in clinical_data_paths:
            cbtn_clinical_data = self.load_clinical_data_cbtn(
                clinical_data_paths["cbtn"], patient_ids_volumes
            )

        if self.cohort == "JOINT":
            # Concatenate BCH and CBTN clinical data
            print(
                f"\tJoining BCH and CBTN clinical data with lengths {len(bch_clinical_data)} and {len(cbtn_clinical_data)}."
            )
            self.clinical_data_reduced = pd.concat(
                [bch_clinical_data, cbtn_clinical_data]
            )

            # Process the combined clinical data
            self.clinical_data_reduced["Patient_ID"] = self.clinical_data_reduced.apply(
                lambda row: str(row.get("BCH MRN", "")).zfill(7)
                if pd.notna(row.get("BCH MRN"))
                else str(row.get("CBTN Subject ID", "")),
                axis=1,
            )
        else:
            # Handle single dataset cases
            if self.cohort == "DF_BCH":
                self.clinical_data_reduced = bch_clinical_data
            elif self.cohort == "CBTN":
                self.clinical_data_reduced = cbtn_clinical_data
            else:
                raise ValueError(f"Invalid cohort: {self.cohort}")

        print(
            f"\tFinal clinical {self.cohort} data has length {len(self.clinical_data_reduced)}."
        )

    def load_volumes_data(self, volumes_data_paths):
        """
        Load volumes data from specified paths. Each path contains CSV files for different patients.
        The data from each file is loaded, processed, and concatenated into a single DataFrame.
        Parameters:
        - volumes_data_paths (list): List containing paths to directories of volume data CSV files.
        The function updates the `self.volumes_data` attribute with the concatenated DataFrame.
        """
        data_frames = []
        total_files = 0
        age_at_last_scan = {}
        age_at_first_scan = {}
        for volumes_data_path in volumes_data_paths:
            all_files = [f for f in os.listdir(volumes_data_path) if f.endswith(".csv")]
            print(f"\tVolume data found: {len(all_files)}.")
            total_files += len(all_files)
            for file in all_files:
                patient_id = file.split("_")[0]
                patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
                # Adjust patient id
                patient_df["Patient_ID"] = patient_id
                if "bch" in str(volumes_data_path).lower():
                    patient_df["Patient_ID"] = (
                        patient_df["Patient_ID"]
                        .astype(str)
                        .str.zfill(7)
                        .astype("string")
                    )
                    patient_df["Date"] = pd.to_datetime(
                        patient_df["Date"], format="%d/%m/%Y", errors='coerce'
                    )
                age_at_last_scan[patient_id] = patient_df["Age"].max()
                age_at_first_scan[patient_id] = patient_df["Age"].min()
        
                data_frames.append(patient_df)
        print(f"\tTotal volume data files found: {total_files}.")
        
        # Ensure all DataFrames have the same columns before concatenation
        all_columns = set().union(*(df.columns for df in data_frames))
        data_frames = [df.reindex(columns=all_columns) for df in data_frames]
        
        # Concatenate DataFrames
        self.volumes_data = pd.concat(data_frames, ignore_index=True)
        
        # Convert 'Age' column to numeric, coercing errors to NaN
        self.volumes_data['Age'] = pd.to_numeric(self.volumes_data['Age'], errors='coerce')
        
        # Fill NaN values appropriately based on column type
        fill_values = {}
        for column in self.volumes_data.columns:
            if pd.api.types.is_numeric_dtype(self.volumes_data[column]):
                fill_values[column] = 0.0
            elif pd.api.types.is_datetime64_any_dtype(self.volumes_data[column]):
                # For datetime columns, we'll fill with NaT (Not a Time)
                fill_values[column] = pd.NaT
            else:
                fill_values[column] = ''
        
        # Apply the fill operation to the entire DataFrame at once
        self.volumes_data = self.volumes_data.fillna(fill_values)
        
        self.age_at_last_scan = age_at_last_scan
        self.age_at_first_scan = age_at_first_scan
        
        if self.cohort in ["CBTN", "JOINT"]:
            follow_up_times = {}
            cbtn_patient_ids = [
                patient_id
                for patient_id in self.volumes_data["Patient_ID"].unique()
                if not patient_id.isdigit()
            ]
            for patient_id in cbtn_patient_ids:
                patient_data = self.volumes_data[
                    self.volumes_data["Patient_ID"] == patient_id
                ]
                min_date = patient_data["Age"].min()
                max_date = patient_data["Age"].max()
                follow_up = max_date - min_date
                follow_up_times[patient_id] = follow_up
            self.volumes_data["Follow-Up Time"] = self.volumes_data["Patient_ID"].map(
                follow_up_times
            )
        
        # Patient IDs in volumes data
        patient_ids_volumes = set(self.volumes_data["Patient_ID"].unique())
        
        return patient_ids_volumes
    
    def extract_treatment_types_bch(self):
        """
        Extract treatment types from the clinical data based on whether surgical resection,
        systemic therapy, or radiation was part of the initial treatment.

        Returns:
        - treatment_list (list): A list of treatment types derived from the clinical data.

        This function is called within the `load_clinical_data` method.
        """
        treatment_list = []
        for _, row in self.clinical_data.iterrows():
            treatments = []

            if row["Surgical Resection"] == "Yes":
                treatments.append("Surgery")

            if row["Systemic therapy before radiation"] == "Yes":
                treatments.append("Chemotherapy")

            if row["Radiation as part of initial treatment"] == "Yes":
                treatments.append("Radiation")

            treatment_type = ", ".join(treatments) if treatments else "No Treatment"
            if treatment_type == "Surgery, Chemotherapy, Radiation":
                treatment_type = "All Treatments"
            treatment_list.append(treatment_type)

        return treatment_list

    def extract_treatment_info_cbtn(self):
        """
        Extracts treatment information for CBTN data.

        Updates self.clinical_data with the columns:
        [Age at First Diagnosis, Treatment Type,
        Received Treatment, Age at First Treatment]
        """

        # Extract the first age recorded in volumes data for each patient
        min_ages = self.volumes_data.groupby('Patient_ID')['Age'].min().reset_index()
        min_ages.columns = ['Patient_ID', 'First Age']
        
        # Initialize new columns in clinical_data
        new_columns = {
            "Age at First Diagnosis": pd.Series(dtype='float64'),
            "Treatment Type": pd.Series(dtype='object'),
            "Received Treatment": pd.Series(dtype='object'),
            "Age at First Treatment": pd.Series(dtype='float64')
        }
        self.clinical_data = self.clinical_data.assign(**new_columns)
        # Prepare a dictionary to store updates
        updates = {col: [] for col in new_columns.keys()}
        update_indices = []
        
        # Loop through each patient in clinical_data
        for idx, row in self.clinical_data.iterrows():
            patient_id = str(row["CBTN Subject ID"])
            surgery = row["Surgery"]
            chemotherapy = row["Chemotherapy"]
            radiation = row["Radiation"]
            age_at_first_treatment = row["Age at Treatment"]
            age_at_first_diagnosis = row["Age at Diagnosis"]
            

            if patient_id in min_ages['Patient_ID'].values:
                first_age = min_ages[min_ages["Patient_ID"] == patient_id]["First Age"].values[0]
            else:
                continue
            
            # Age at First Diagnosis
            if np.isnan(age_at_first_diagnosis) or age_at_first_diagnosis > first_age:
                age_at_first_diagnosis = int(first_age)
                
            updates["Age at First Diagnosis"].append(age_at_first_diagnosis)
    

            # Age at First Treatment
            if np.isnan(age_at_first_treatment):
                age_at_first_treatment = np.nan
            updates["Age at First Treatment"].append(age_at_first_treatment)            
            
            # Treatment Type
            treatments = []
            if surgery == "Yes":
                treatments.append("Surgery")
            if chemotherapy == "Yes":
                treatments.append("Chemotherapy")
            if radiation == "Yes":
                treatments.append("Radiation")
            treatment_type = ", ".join(treatments) if treatments else "No Treatment"
            if treatment_type == "Surgery, Chemotherapy, Radiation":
                treatment_type = "All Treatments"
            updates["Treatment Type"].append(treatment_type)

            # Received Treatment
            received_treatment = "Yes" if treatments else "No"
            updates["Received Treatment"].append(received_treatment)
                    
            update_indices.append(idx)
                
        # Apply all updates at once
        for col, values in updates.items():
            self.clinical_data.loc[update_indices, col] = values

        # Convert columns to appropriate data types
        self.clinical_data["Age at First Diagnosis"] = pd.to_numeric(
            self.clinical_data["Age at First Diagnosis"], errors="coerce"
        )
        self.clinical_data["Age at First Treatment"] = pd.to_numeric(
            self.clinical_data["Age at First Treatment"], errors="coerce"
        )
        
        # Fill NA values in Age at First Treatment
        mask = self.clinical_data["Age at First Treatment"].isna()
        self.clinical_data.loc[mask, "Age at First Treatment"] = self.clinical_data.loc[mask, "Age at Last Clinical Follow-Up"]

        # Update Age at Last Clinical Follow-Up
        self.clinical_data["Age at Last Clinical Follow-Up"] = pd.to_numeric(
            self.clinical_data["Age at Last Clinical Follow-Up"], errors="coerce"
        )
        self.clinical_data["Age at Last Clinical Follow-Up"] = np.minimum(
            self.clinical_data["Age at Last Clinical Follow-Up"],
            self.clinical_data["Age at First Treatment"]
        )

    def merge_data(self):
        """
        Merge reduced clinical data with the volumes data based on patient ID,
        excluding redundant columns.

        This function updates the `self.merged_data` attribute with the merged DataFrame.
        """
        if "Follow-Up Time" in self.volumes_data.columns:
            self.volumes_data["Volumes Follow-Up Time"] = self.volumes_data[
                "Follow-Up Time"
            ]
            self.volumes_data = self.volumes_data.drop(columns=["Follow-Up Time"])

        if self.cohort == "JOINT":
            self.merged_data = pd.merge(
                self.clinical_data_reduced,
                self.volumes_data,
                on=["Patient_ID"],
                how="right",
            )
            self.merged_data["Follow-Up Time"] = self.merged_data.apply(
                lambda row: row["Follow-Up Time"]
                if row["Dataset"] == "DF_BCH"
                else row["Volumes Follow-Up Time"],
                axis=1,
            )

            self.merged_data = self.merged_data.drop(
                columns=["CBTN Subject ID", "BCH MRN", "Volumes Follow-Up Time"]
            )

        else:
            if self.cohort == "DF_BCH":
                column = "BCH MRN"
            else:
                column = "CBTN Subject ID"

            self.merged_data = pd.merge(
                self.clinical_data_reduced,
                self.volumes_data,
                left_on=[column],
                right_on=["Patient_ID"],
                how="right",
            )
            self.merged_data = self.merged_data.drop(columns=[column])
            if "Volumes Follow-Up Time" in self.merged_data.columns:
                self.merged_data["Follow-Up Time"] = self.merged_data["Volumes Follow-Up Time"]
                self.merged_data = self.merged_data.drop(columns=["Volumes Follow-Up Time"])
        
        self.merged_data["Age Group"] = self.merged_data.apply( lambda x: categorize_age_group(x, column="Age"), axis=1).astype("category")
        self.merged_data["Age Group at Diagnosis"] = self.merged_data.apply(lambda x: categorize_age_group(x, column="Age at First Diagnosis"), axis=1).astype("category")
        self.merged_data['Age at First Diagnosis (Years)'] = self.merged_data['Age at First Diagnosis'] / 365.25
        self.merged_data["Time Period Since Diagnosis"] = self.merged_data.apply(categorize_time_since_first_diagnosis, axis=1).astype("category")
        self.merged_data["Baseline Volume cm3"] = self.merged_data["Baseline Volume"] / 1000
        self.merged_data["Change Speed"] = self.merged_data["Change Speed"].astype(
            "category"
        )
        self.merged_data["Change Type"] = self.merged_data["Change Type"].astype(
            "category"
        )
        self.merged_data["Change Trend"] = self.merged_data["Change Trend"].astype("category")
        self.merged_data["Change Acceleration"] = self.merged_data["Change Acceleration"].astype("category")
        self.merged_data["Dataset"] = self.merged_data["Dataset"].astype("category")
        self.merged_data["Histology"] = self.merged_data["Histology"].astype("category")
        self.merged_data["Patient_ID"] = self.merged_data["Patient_ID"].astype("string")
        self.merged_data["Treatment Type"] = self.merged_data["Treatment Type"].astype(
            "category"
        )
        self.merged_data['Age Median'] = self.merged_data.groupby('Patient_ID')['Age'].transform('median')
        self.merged_data['Follow-Up Time Median'] = self.merged_data.groupby('Patient_ID')['Follow-Up Time'].transform('median')
        self.merged_data['Days Between Scans Median'] = self.merged_data.groupby('Patient_ID')['Days Between Scans'].transform('median')
        self.merged_data.reset_index(drop=True, inplace=True)
        print(
            f"\tMerged clinical and volume data. Final data has length {len(self.merged_data)} and unique patients {self.merged_data['Patient_ID'].nunique()}."
        )

    def check_data_consistency(self, debug=False):
        """
        Check the consistency of the data after processing and merging.
        """
        columns_to_check = ["Age at First Diagnosis", "Age at Last Clinical Follow-Up"]
        for column in columns_to_check:
            assert not self.merged_data[column].isnull().any(), f"Column '{column}' contains NaN values."
        if debug:
            print("\tAge columns have no NaN values.")
        
        for patient_id, age_at_last_scan in self.age_at_last_scan.items():
            mask_patient = self.merged_data[self.merged_data["Patient_ID"] == patient_id]
            age_at_last_clinical_follow_up = mask_patient["Age at Last Clinical Follow-Up"].values[0]
            if not age_at_last_clinical_follow_up == age_at_last_scan:
                #print(f"For patient {patient_id}, 'Age at Last Clinical Follow-Up' ({age_at_last_clinical_follow_up}) should be the same as 'Age at Last Scan' ({age_at_last_scan}).")
                self.merged_data.loc[self.merged_data["Patient_ID"] == patient_id, "Age at Last Clinical Follow-Up"] = age_at_last_scan
                #print(f"Updated 'Age at Last Clinical Follow-Up' to {age_at_last_clinical_follow_up}.")
        if debug:
            print("\tData consistency for 'Age at Last Clinical Follow-Up' == 'Age at Last Scan' passed.")

        for patient_id, age_at_first_scan in self.age_at_first_scan.items():
            mask_patient = self.merged_data[self.merged_data["Patient_ID"] == patient_id]
            age_at_first_diagnosis = mask_patient["Age at First Diagnosis"].values[0]
            if not age_at_first_diagnosis == age_at_first_scan:
                #print(f"For patient {patient_id}, 'Age at First Diagnosis' ({age_at_first_diagnosis}) should be the same as 'Age at First Scan' ({age_at_first_scan}).")
                self.merged_data.loc[self.merged_data["Patient_ID"] == patient_id, "Age at First Diagnosis"] = age_at_first_scan
                #print(f"Updated 'Age at Last Clinical Follow-Up' to {age_at_last_clinical_follow_up}.")
        if debug:  
            print("\tData consistency for 'Age at First Diagnosis' == 'Age at First Scan' passed.")
        
        # Check if "Age at Last Clinical Follow-Up" is the same as "Age at Treatment" if "Received Treatment" is True with an assertion
        mask_treatment = self.merged_data["Received Treatment"] == 'Yes'
        age_last_follow_up = self.merged_data.loc[mask_treatment, "Age at Last Clinical Follow-Up"]
        age_first_treatment = self.merged_data.loc[mask_treatment, "Age at First Treatment"]
        mismatch_mask = age_last_follow_up >= age_first_treatment
        assert mismatch_mask.sum() == 0, f"'Age at Last Clinical Follow-Up' should be the same as 'Age at First Treatment'. Mismatches found:\n{self.merged_data.loc[mask_treatment & mismatch_mask, ['Patient_ID', 'Age at Last Clinical Follow-Up', 'Age at First Treatment']]}"

        if debug:
            print("\tAssertion for 'Age at First Treatment' == 'Age at Last Clinical Follow-Up' passed.")
    
        
        print("\tData consistency check passed.")

    @staticmethod
    def sort_df(df):
        """
        Method to sort the merged data by Patient_ID and Age.
        """
        first_columns = ["Patient_ID", "Age"]
        all_columns = list(df.columns)
        columns_to_sort = [col for col in all_columns if col not in first_columns]
        sorted_columns = sorted(columns_to_sort)
        final_column_order = first_columns + sorted_columns
        if "Date" in final_column_order:
            final_column_order.remove("Date")
        
        return df[final_column_order]
   
    def check_dtypes_and_nan(self):
        """
        Check data types and NaN values in the cohort data.
        """
        print("\tChecking data types and NaN values...")
        print(self.merged_data.info())
        print(self.merged_data.isnull().sum())
        print("\tData types and NaN values checked.")


class CohortStatistics:
    def __init__(self, cohort_data):
        self.cohort_data = cohort_data
        print("Step 1: Initializing CohortStatistics class...")

    def printout_stats(self, file_path):
        """
        Descriptive statistics written to a file.

        Parameters:
        - output_file_path (str): Path to the output path.
        - prefix (str): Prefix used for naming the output file.
        """
        with open(file_path, "w", encoding="utf-8") as file:

            def write_stat(statement):
                file.write(statement + "\n")

            # Age
            median_age = self.cohort_data["Age"].median()
            max_age = self.cohort_data["Age"].max()
            min_age = self.cohort_data["Age"].min()
            write_stat(f"\t\tMedian Age: {median_age / 365.25} years")
            write_stat(f"\t\tMaximum Age: {max_age / 365.25} years")
            write_stat(f"\t\tMinimum Age: {min_age / 365.25} years")
            median_age_diagnosis = self.cohort_data["Age at First Diagnosis (Years)"].median()
            max_age_diagnosis = self.cohort_data["Age at First Diagnosis (Years)"].max()
            min_age_diagnosis = self.cohort_data["Age at First Diagnosis (Years)"].min()
            write_stat(f"\t\tMedian Age at Diagnosis: {median_age_diagnosis} years")
            write_stat(f"\t\tMaximum Age at Diagnosis: {max_age_diagnosis} years")
            write_stat(f"\t\tMinimum Age at Diagnosis: {min_age_diagnosis} years")

            # Days Between Scans
            median_days_between_scans = self.cohort_data["Days Between Scans"].median()
            max_days_between_scans = self.cohort_data["Days Between Scans"].max()
            unique_days = self.cohort_data["Days Between Scans"].unique()
            next_smallest = min(day for day in unique_days if day > 0)

            write_stat(f"\t\tMedian Days Between Scans: {median_days_between_scans / 30.4} months")
            write_stat(f"\t\tMaximum Days Between Scans: {max_days_between_scans / 30.4} months")
            write_stat(f"\t\tMinimum Days Between Scans: {next_smallest /30.4} months")

            # Sex, Received Treatment, Symptoms, Location,
            # Patient Classification, Treatment Type
            copy_df = self.cohort_data.copy()
            unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
            counts_braf = unique_pat["BRAF Status"].value_counts()
            counts_sex = unique_pat["Sex"].value_counts()
            counts_received_treatment = unique_pat["Received Treatment"].value_counts()
            counts_symptoms = unique_pat["Symptoms"].value_counts()
            counts_histology = unique_pat["Histology"].value_counts()
            counts_location = unique_pat["Location"].value_counts()
            counts_treatment_type = unique_pat["Treatment Type"].value_counts()
            counts_age_group = unique_pat["Age Group at Diagnosis"].value_counts()
            
            write_stat(f"\t\tAge Group at Diagnosis: {counts_age_group}")
            write_stat(f"\t\tReceived Treatment: {counts_received_treatment}")
            write_stat(f"\t\tSymptoms: {counts_symptoms}")
            write_stat(f"\t\tHistology: {counts_histology}")
            write_stat(f"\t\tLocation: {counts_location}")
            write_stat(f"\t\tSex: {counts_sex}")
            write_stat(f"\t\tTreatment Type: {counts_treatment_type}")
            write_stat(f"\t\tBRAF Status: {counts_braf}")

            # Volume Change
            mm3_to_cm3 = 1000
            filtered_data = self.cohort_data[self.cohort_data["Volume Change"] != 0]
            median_volume_change = filtered_data["Volume Change"].median()
            max_volume_change = filtered_data["Volume Change"].max()
            min_volume_change = filtered_data["Volume Change"].min()
            write_stat(f"\t\tMedian Volume Change: {median_volume_change / mm3_to_cm3} cm3")
            write_stat(f"\t\tMaximum Volume Change: {max_volume_change / mm3_to_cm3} cm3")
            write_stat(f"\t\tMinimum Volume Change: {min_volume_change / mm3_to_cm3} cm3")

            filtered_data = self.cohort_data[self.cohort_data["Volume Change Pct"] != 0]
            median_volume_change_pct = filtered_data["Volume Change Pct"].median()
            max_volume_change_pct = filtered_data["Volume Change Pct"].max()
            min_volume_change_pct = filtered_data["Volume Change Pct"].min()
            write_stat(f"\t\tMedian Volume Change Pct: {median_volume_change_pct} %")
            write_stat(f"\t\tMaximum Volume Change Pct: {max_volume_change_pct} %")
            write_stat(f"\t\tMinimum Volume Change Pct: {min_volume_change_pct} %")

            # Volume Change Rate
            filtered_data = self.cohort_data[
                self.cohort_data["Volume Change Rate"] != 0
            ]
            # Convert from per day to per month (multiply by average days in month)
            days_per_month = 30.44  # average days in a month
            median_volume_change_rate = filtered_data["Volume Change Rate"].median() * days_per_month
            max_volume_change_rate = filtered_data["Volume Change Rate"].max() * days_per_month
            min_volume_change_rate = filtered_data["Volume Change Rate"].min() * days_per_month
            write_stat(
                f"\t\tMedian Volume Change Rate: {median_volume_change_rate/mm3_to_cm3:.3f} cm3/month"
            )
            write_stat(
                f"\t\tMaximum Volume Change Rate: {max_volume_change_rate/mm3_to_cm3:.3f} cm3/month"
            )
            write_stat(
                f"\t\tMinimum Volume Change Rate: {min_volume_change_rate/mm3_to_cm3:.3f} cm3/month"
            )
            filtered_data = self.cohort_data[self.cohort_data["Volume Change Rate Pct"] != 0]
            median_volume_change_rate_pct = filtered_data["Volume Change Rate Pct"].median() * days_per_month
            max_volume_change_rate_pct = filtered_data["Volume Change Rate Pct"].max() * days_per_month
            min_volume_change_rate_pct = filtered_data["Volume Change Rate Pct"].min() * days_per_month
            write_stat("\t\tMedian Volume Change Rate Pct: {:.2f} %/month".format(median_volume_change_rate_pct))
            write_stat("\t\tMaximum Volume Change Rate Pct: {:.2f} %/month".format(max_volume_change_rate_pct))
            write_stat("\t\tMinimum Volume Change Rate Pct: {:.2f} %/month".format(min_volume_change_rate_pct))

            # Normalized Volume
            median_normalized_volume = self.cohort_data["Normalized Volume"].median()
            max_normalized_volume = self.cohort_data["Normalized Volume"].max()
            min_normalized_volume = self.cohort_data["Normalized Volume"].min()
            write_stat(f"\t\tMedian Normalized Volume: {median_normalized_volume} mm^3")
            write_stat(f"\t\tMaximum Normalized Volume: {max_normalized_volume} mm^3")
            write_stat(f"\t\tMinimum Normalized Volume: {min_normalized_volume} mm^3")

            # Volume
            median_volume = self.cohort_data["Volume"].median()
            max_volume = self.cohort_data["Volume"].max()
            min_volume = self.cohort_data["Volume"].min()
            write_stat(f"\t\tMedian Volume: {median_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMaximum Volume: {max_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMinimum Volume: {min_volume / mm3_to_cm3} cm^3")

            # Baseline volume
            median_baseline_volume = self.cohort_data["Baseline Volume"].median()
            max_baseline_volume = self.cohort_data["Baseline Volume"].max()
            min_baseline_volume = self.cohort_data["Baseline Volume"].min()
            write_stat(
                f"\t\tMedian Baseline Volume: {median_baseline_volume / mm3_to_cm3} cm^3"
            )
            write_stat(
                f"\t\tMaximum Baseline Volume: {max_baseline_volume / mm3_to_cm3} cm^3"
            )
            write_stat(
                f"\t\tMinimum Baseline Volume: {min_baseline_volume / mm3_to_cm3} cm^3"
            )

            # Follow-Up time
            median_follow_up = self.cohort_data["Follow-Up Time"].median()
            max_follow_up = self.cohort_data["Follow-Up Time"].max()
            min_follow_up = self.cohort_data["Follow-Up Time"].min()
            median_follow_up_y = median_follow_up / 365.25
            max_follow_up_y = max_follow_up / 365.25
            min_follow_up_y = min_follow_up / 365.25
            write_stat(f"\t\tMedian Follow-Up Time: {median_follow_up_y:.2f} years")
            write_stat(f"\t\tMaximum Follow-Up Time: {max_follow_up_y:.2f} years")
            write_stat(f"\t\tMinimum Follow-Up Time: {min_follow_up_y:.2f} years")

            # get the ids of the patients with the three highest normalized volumes that do not repeat
            top_normalized_volumes = self.cohort_data.nlargest(3, "Normalized Volume")
            top_volumes = self.cohort_data.nlargest(3, "Volume")
            self.cohort_data["Absolute Volume Change"] = self.cohort_data[
                "Volume Change"
            ].abs()
            self.cohort_data["Absolute Volume Change Rate"] = self.cohort_data[
                "Volume Change Rate"
            ].abs()
            top_volume_changes = self.cohort_data.nlargest(3, "Absolute Volume Change")
            top_volume_change_rates = self.cohort_data.nlargest(
                3, "Absolute Volume Change Rate"
            )
            patient_ids_highest_norm_volumes = top_normalized_volumes[
                "Patient_ID"
            ].tolist()
            patient_ids_highest_volumes = top_volumes["Patient_ID"].tolist()
            patient_ids_highest_volume_changes = top_volume_changes[
                "Patient_ID"
            ].tolist()
            patient_ids_highest_volume_change_rates = top_volume_change_rates[
                "Patient_ID"
            ].tolist()
            write_stat(
                f"\t\tPatients with highest normalized volumes: {patient_ids_highest_norm_volumes}"
            )
            write_stat(
                f"\t\tPatients with highest volumes: {patient_ids_highest_volumes}"
            )
            write_stat(
                f"\t\tPatients with highest volume changes: {patient_ids_highest_volume_changes}"
            )
            write_stat(
                f"\t\tPatients with highest volume change rates: {patient_ids_highest_volume_change_rates}"
            )

        print(f"\tSaved summary statistics to {file_path}.")

    def generate_distribution_plots(self, output_dir):
        """
        Violin plots.
        """
        data = self.cohort_data.copy()
        palette = NORD_PALETTE[:3]
        sns.set_palette(palette)
        plt.rc('xtick', labelsize=15) 
        plt.rc('ytick', labelsize=15)
        
        # Create a figure with subplots in a single column
        _, axs = plt.subplots(3, 1, figsize=(10, 20))
        
        # Violin plot for "Follow-Up Time" distribution per dataset
        data["Follow-Up Time (Years)"] = data["Follow-Up Time"] / 365.25
        sns.boxplot(x="Dataset", y="Follow-Up Time (Years)", data=data, ax=axs[0], palette=palette)
        axs[0].set_title("Distribution of Follow-Up Time", fontsize=20)
        axs[0].set_xlabel("Dataset", fontsize=15)
        axs[0].set_ylabel("Follow-Up Time [years]", fontsize=15)
        
        # Violin plot for number of scans per patient per dataset
        scans_per_patient = (
            self.cohort_data.groupby(["Dataset", "Patient_ID"])
            .size()
            .reset_index(name="Number of Scans")
        )
        # Filter out patients with less than 3 scans
        scans_per_patient = scans_per_patient[scans_per_patient["Number of Scans"] >= 3]
        sns.violinplot(
            x="Dataset", y="Number of Scans", data=scans_per_patient, ax=axs[1], palette=palette
        )
        sns.stripplot(
            x="Dataset",
            y="Number of Scans",
            data=scans_per_patient,
            ax=axs[1],
            color="black",
            size=3,
            alpha=0.5,
        )
        axs[1].set_title("Distribution of Number of Scans per Patient", fontsize=20)
        axs[1].set_xlabel("Dataset", fontsize=15)
        axs[1].set_ylabel("Number of Scans", fontsize=15)
        axs[1].set_ylim(bottom=2.5, top=32)
        
        # Violin plot for follow-up interval distribution per dataset
        # Convert days to months (using average month length of 30.44 days)
        data["Months Between Scans"] = data["Days Between Scans"] / 30.44
        sns.violinplot(y="Dataset", x="Months Between Scans", data=data, ax=axs[2], palette=palette)
        axs[2].set_title("Distribution of Follow-Up Intervals", fontsize=20)
        axs[2].set_ylabel("Dataset", fontsize=15)
        axs[2].set_xlabel("Time Between Scans [months]", fontsize=15)
        
        plt.tight_layout()
        # Display the plot
        file_name = os.path.join(output_dir, "dataset_comparison.png")
        plt.savefig(file_name, dpi=300)    
        
        print(f"\tSaved distribution plots to {file_name}.")
    
    def create_cohort_table(self, categorical_vars, continuous_vars, output_file_path):
        """
        Create a table comparing the two cohorts based on the variables of interest.
        """
        cohort_var = "Dataset"
        data = self.cohort_data.copy()
        cohort_table = pd.DataFrame(
            columns=["Variable", "Cohort 1", "Cohort 2", "P-value"]
        )
        aggregated_data = data.groupby(["Patient_ID", cohort_var]).last().reset_index()
        aggregated_data = aggregated_data.dropna(subset=continuous_vars)
        for var in categorical_vars + continuous_vars:
            cohort1_data = aggregated_data[
                aggregated_data[cohort_var] == aggregated_data[cohort_var].unique()[0]
            ]
            cohort2_data = aggregated_data[
                aggregated_data[cohort_var] == aggregated_data[cohort_var].unique()[1]
            ]

            if var in categorical_vars:
                contingency_table = pd.crosstab(
                    aggregated_data[cohort_var], aggregated_data[var]
                )
                if cohort1_data[var].nunique() == 2 and cohort2_data[var].nunique() == 2:
                    _, p_val = fisher_exact(contingency_table)
                else:
                    _, p_val, _, _ = chi2_contingency(contingency_table)

                cohort1_value = contingency_table.loc[
                    aggregated_data[cohort_var].unique()[0]
                ].to_dict()
                cohort2_value = contingency_table.loc[
                    aggregated_data[cohort_var].unique()[1]
                ].to_dict()
            else:
                if len(cohort1_data) >= 30 and len(cohort2_data) >= 30:
                    if check_assumptions(var, var, cohort1_data, "t-test") and check_assumptions(var, var, cohort2_data, "t-test"):
                        _, p_val = ttest_ind(cohort1_data[var], cohort2_data[var])
                    else:
                        _, p_val = mannwhitneyu(cohort1_data[var], cohort2_data[var])
                else:
                    _, p_val = mannwhitneyu(cohort1_data[var], cohort2_data[var])

                cohort1_value = (
                    f"{cohort1_data[var].mean():.2f} ± {cohort1_data[var].std():.2f}"
                )
                cohort2_value = (
                    f"{cohort2_data[var].mean():.2f} ± {cohort2_data[var].std():.2f}"
                )

            new_row = pd.DataFrame(
                {
                    "Variable": [var],
                    "Cohort 1": [cohort1_value],
                    "Cohort 2": [cohort2_value],
                    "P-value": [f"{p_val:.3f}"],
                }
            )
            cohort_table = pd.concat([cohort_table, new_row], ignore_index=True)

        # Ensure the output directory exists
        file_type = str(output_file_path).split('.')[-1]
        if file_type.lower() == 'csv':
            cohort_table.to_csv(output_file_path, index=False)
            print(f"\tCohort table saved as CSV: {output_file_path}")
        elif file_type.lower() == 'txt':
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cohort_table.to_string(index=False))
            print(f"\tCohort table saved as TXT: {output_file_path}")
        else:
            raise ValueError("Invalid file_type. Choose either 'csv' or 'txt'.")

        return cohort_table


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cohort_name = cohort_creation_cfg.COHORT
    if cohort_name == "JOINT":
        data_paths = {
            "clinical_data_paths": cohort_creation_cfg.CLINICAL_CSV_PATHS,
            "volumes_data_paths": [
                cohort_creation_cfg.VOLUMES_DATA_PATHS["joint"]
            ],
        }
    else:
        data_paths = {
            "clinical_data_paths": [cohort_creation_cfg.CLINICAL_CSV_PATHS[
                cohort_name.lower()
            ]],
            "volumes_data_paths": [
                cohort_creation_cfg.VOLUMES_DATA_PATHS[cohort_name.lower()]
            ],
        }

    output_dir = cohort_creation_cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    cohort_creation = CohortCreation(data_paths, output_dir, cohort=cohort_name)
    
    # Once data is merged, create the basic stats plot out of this
    cohort_stats_file = cohort_creation_cfg.OUTPUT_STATS_FILE
    cohort_table_file = cohort_creation_cfg.COHORT_TABLE_FILE
    categorical_vars = cohort_creation_cfg.CATEGORICAL_VARS
    numerical_vars = cohort_creation_cfg.NUMERICAL_VARS
    cohort_data = cohort_creation.merged_data
    cohort_statistics = CohortStatistics(cohort_data)
    cohort_statistics.printout_stats(cohort_stats_file)
    cohort_statistics.generate_distribution_plots(output_dir)
    cohort_statistics.create_cohort_table(categorical_vars, numerical_vars, cohort_table_file)
