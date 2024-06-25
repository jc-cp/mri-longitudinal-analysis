# pylint: disable=too-many-lines
"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations, stability and trend analysis.
"""
import os
import warnings
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro, ttest_ind, chi2_contingency, mannwhitneyu, fisher_exact, levene
from cfg.src import correlation_cfg
from cfg.utils.helper_functions_cfg import NORD_PALETTE
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.plotting import add_at_risk_counts 
from lifelines.utils import concordance_index
from utils.helper_functions import (
    bonferroni_correction,
    chi_squared_test,
    f_one,
    point_bi_serial,
    perform_propensity_score_matching,
    calculate_propensity_scores,
    sensitivity_analysis,
    spearman_correlation,
    pearson_correlation,
    ttest,
    zero_fill,
    check_balance,
    visualize_smds,
    visualize_p_value_bonferroni_corrections,
    fdr_correction,
    visualize_fdr_correction,
    save_for_deep_learning,
    categorize_age_group,
    # calculate_group_norms_and_stability,
    classify_patient,
    plot_trend_trajectories,
    plot_individual_trajectories,
    #calculate_percentage_change,
    #visualize_tumor_stability,
    #consistency_check,
    kruskal_wallis_test,
    fisher_exact_test,
    logistic_regression_analysis,
    # calculate_vif,
    calculate_stability_index,
    visualize_stability_index,
    visualize_individual_indexes,
    roc_curve_and_auc, 
    visualize_ind_indexes_distrib,
    visualize_volume_change, 
    #grid_search_weights,
    categorize_time_since_first_diagnosis,
    plot_histo_distributions
)


class TumorAnalysis:
    """
    A class to perform tumor analysis using clinical and volumetric data.
    """

    ##################################
    # DATA LOADING AND PREPROCESSING #
    ##################################

    def __init__(self, data_paths_, cohort):
        """
        Initialize the TumorAnalysis class.

        Parameters:
            clinical_data_file (str): Path to the clinical data CSV file.
            volumes_data_file (str): Path to the tumor volumes data CSV file.
        """
        pd.options.display.float_format = "{:.3f}".format
        self.merged_data = pd.DataFrame()
        self.clinical_data_reduced = pd.DataFrame()
        self.post_treatment_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.results = {}
        self.progression_threshold = correlation_cfg.PROGRESSION_THRESHOLD
        self.volume_change_threshold = correlation_cfg.CHANGE_THRESHOLD
        self.caliper = correlation_cfg.CALIPER
        self.sample_size_plots = correlation_cfg.SAMPLE_SIZE
        self.cohort = cohort
        self.reference_categories = {}
        print("Step 0: Initializing TumorAnalysis class...")


        patient_ids_volumes = self.load_volumes_data(data_paths_["volumes_data_paths"])
        if self.cohort == "JOINT":
            self.validate_files(list(data_paths_["clinical_data_paths"].values()), data_paths_["volumes_data_paths"])
            self.load_clinical_data(
                data_paths_["clinical_data_paths"], patient_ids_volumes
            )
        else:
            self.validate_files(data_paths_["clinical_data_paths"], data_paths_["volumes_data_paths"])
            if self.cohort == "BCH":
                _ = self.load_clinical_data_bch(
                    data_paths_["clinical_data_paths"][0], patient_ids_volumes
                )
            elif self.cohort == "CBTN":
                _ = self.load_clinical_data_cbtn(
                    data_paths_["clinical_data_paths"][0], patient_ids_volumes
                )
        self.merge_data()
        self.check_data_consistency()

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
                return "No symptoms (incident finding)"
            if map_type == "histology":
                return "Other"

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
            correlation_cfg.BCH_LOCATION,
            self.clinical_data["Location of Tumor"],
            map_type="location",
        )

        self.clinical_data["Symptoms"] = self.map_dictionary(
            correlation_cfg.BCH_SYMPTOMS,
            self.clinical_data["Symptoms at diagnosis"],
            map_type="symptoms",
        )

        self.clinical_data["Histology"] = self.map_dictionary(
            correlation_cfg.BCH_GLIOMA_TYPES,
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
        for column, dtype in correlation_cfg.BCH_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        all_relevant_columns = (
            list(correlation_cfg.BCH_DTYPE_MAPPING.keys())
            + correlation_cfg.BCH_DATETIME_COLUMNS
        )
        self.clinical_data_reduced = self.clinical_data[all_relevant_columns].copy()
        self.clinical_data_reduced["BCH MRN"] = (
            self.clinical_data_reduced["BCH MRN"].astype(str).str.zfill(7)
        )
        self.clinical_data_reduced = self.clinical_data_reduced[
            self.clinical_data_reduced["BCH MRN"].isin(patient_ids_volumes)
        ]
        print(f"\tFiltered clinical data has length {len(self.clinical_data_reduced)}.")

        self.clinical_data_reduced["Dataset"] = "BCH"
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
            correlation_cfg.CBTN_LOCATION,  # Define a suitable mapping dictionary
            self.clinical_data["Tumor Locations"],
            map_type="location",
        )

        # Map symptoms
        self.clinical_data["Symptoms"] = self.map_dictionary(
            correlation_cfg.CBTN_SYMPTOMS,
            self.clinical_data["Medical Conditions Present at Event"],
            map_type="symptoms",
        )

        # Map Sex
        self.clinical_data["Sex"] = self.clinical_data["Legal Sex"].apply(
            lambda x: "Female" if x == "Female" else "Male"
        )

        # Map Histology
        self.clinical_data["Histology"] = self.map_dictionary(
            correlation_cfg.CBTN_GLIOMA_TYPES,
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
        for column, dtype in correlation_cfg.CBTN_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        # Select relevant columns for reduced data
        all_relevant_columns = (
            list(correlation_cfg.CBTN_DTYPE_MAPPING.keys())
            + correlation_cfg.CBTN_DATETIME_COLUMNS
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
        if "bch" in clinical_data_paths:
            bch_clinical_data = self.load_clinical_data_bch(
                clinical_data_paths["bch"], patient_ids_volumes
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
            if self.cohort == "BCH":
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
                        patient_df["Date"], format="%d/%m/%Y"
                    )
                age_at_last_scan[patient_id] = patient_df["Age"].max()
                age_at_first_scan[patient_id] = patient_df["Age"].min()
            
                data_frames.append(patient_df)

        print(f"\tTotal volume data files found: {total_files}.")

        self.volumes_data = pd.concat(data_frames, ignore_index=True)
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
    	
        self.volumes_data['Age'] = pd.to_numeric(self.volumes_data['Age'], errors='coerce')
        
        if self.volumes_data.isna().any().any():
            self.volumes_data.fillna(0.0, inplace=True)
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
        self.clinical_data["Age at First Diagnosis"] = None
        self.clinical_data["Treatment Type"] = None
        self.clinical_data["Received Treatment"] = None
        self.clinical_data["Age at First Treatment"] = None
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
                
            self.clinical_data.at[
                idx, "Age at First Diagnosis"
            ] = age_at_first_diagnosis

            # Age at First Treatment
            if np.isnan(age_at_first_treatment):
                age_at_first_treatment = self.clinical_data.at[idx, "Age at Last Clinical Follow-Up"]
            self.clinical_data.at[idx, "Age at First Treatment"] = age_at_first_treatment
            
            # Treatment Type
            treatments = []
            if surgery == "Yes":
                treatments.append("Surgery")
            if chemotherapy == "Yes":
                treatments.append("Chemotherapy")
            if radiation == "Yes":
                treatments.append("Radiation")
            treatment_type = ", ".join(treatments) if treatments else "No Treatment"

            if treatment_type != "Surgery, Chemotherapy, Radiation":
                self.clinical_data.at[idx, "Treatment Type"] = treatment_type
            else:
                self.clinical_data.at[idx, "Treatment Type"] = "All Treatments"

            # Received Treatment
            received_treatment = "Yes" if treatments else "No"
            self.clinical_data.at[idx, "Received Treatment"] = received_treatment

                
        self.clinical_data["Age at First Diagnosis"] = pd.to_numeric(
            self.clinical_data["Age at First Diagnosis"], errors="coerce"
        )
        self.clinical_data["Age at First Treatment"].fillna(self.clinical_data["Age at Last Clinical Follow-Up"], inplace=True)               
        self.clinical_data["Age at Last Clinical Follow-Up"] = np.minimum(
            pd.to_numeric(
                self.clinical_data["Age at Last Clinical Follow-Up"], errors="coerce"
            ),
            self.clinical_data["Age at First Treatment"],
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
                if row["Dataset"] == "BCH"
                else row["Volumes Follow-Up Time"],
                axis=1,
            )

            self.merged_data = self.merged_data.drop(
                columns=["CBTN Subject ID", "BCH MRN", "Volumes Follow-Up Time"]
            )

        else:
            if self.cohort == "BCH":
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
        # self.merged_data["Patient_ID"] = self.merged_data["Patient_ID"].astype("string")
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

    def check_data_consistency(self):
        """
        Check the consistency of the data after processing and merging.
        """
        columns_to_check = ["Age at First Diagnosis", "Age at Last Clinical Follow-Up"]
        for column in columns_to_check:
            assert not self.merged_data[column].isnull().any(), f"Column '{column}' contains NaN values."
        print("\tAge columns have no NaN values.")
        
        for patient_id, age_at_last_scan in self.age_at_last_scan.items():
            mask_patient = self.merged_data[self.merged_data["Patient_ID"] == patient_id]
            age_at_last_clinical_follow_up = mask_patient["Age at Last Clinical Follow-Up"].values[0]
            if not age_at_last_clinical_follow_up == age_at_last_scan:
                #print(f"For patient {patient_id}, 'Age at Last Clinical Follow-Up' ({age_at_last_clinical_follow_up}) should be the same as 'Age at Last Scan' ({age_at_last_scan}).")
                self.merged_data.loc[self.merged_data["Patient_ID"] == patient_id, "Age at Last Clinical Follow-Up"] = age_at_last_scan
                #print(f"Updated 'Age at Last Clinical Follow-Up' to {age_at_last_clinical_follow_up}.")
        print("\tData consistency for 'Age at Last Clinical Follow-Up' == 'Age at Last Scan' passed.")

        for patient_id, age_at_first_scan in self.age_at_first_scan.items():
            mask_patient = self.merged_data[self.merged_data["Patient_ID"] == patient_id]
            age_at_first_diagnosis = mask_patient["Age at First Diagnosis"].values[0]
            if not age_at_first_diagnosis == age_at_first_scan:
                #print(f"For patient {patient_id}, 'Age at First Diagnosis' ({age_at_first_diagnosis}) should be the same as 'Age at First Scan' ({age_at_first_scan}).")
                self.merged_data.loc[self.merged_data["Patient_ID"] == patient_id, "Age at First Diagnosis"] = age_at_first_scan
                #print(f"Updated 'Age at Last Clinical Follow-Up' to {age_at_last_clinical_follow_up}.")
        print("\tData consistency for 'Age at First Diagnosis' == 'Age at First Scan' passed.")
        
        # Check if "Age at Last Clinical Follow-Up" is the same as "Age at Treatment" if "Received Treatment" is True with an assertion
        mask_treatment = self.merged_data["Received Treatment"] == 'Yes'
        age_last_follow_up = self.merged_data.loc[mask_treatment, "Age at Last Clinical Follow-Up"]
        age_first_treatment = self.merged_data.loc[mask_treatment, "Age at First Treatment"]
        mismatch_mask = age_last_follow_up >= age_first_treatment
        assert mismatch_mask.sum() == 0, f"'Age at Last Clinical Follow-Up' should be the same as 'Age at First Treatment'. Mismatches found:\n{self.merged_data.loc[mask_treatment & mismatch_mask, ['Patient_ID', 'Age at Last Clinical Follow-Up', 'Age at First Treatment']]}"

        print("\tAssertion for 'Age at First Treatment' == 'Age at Last Clinical Follow-Up' passed.")
    
        
        print("\tData consistency check passed.")

    ###################################
    # DATA ANALYSIS AND VISUALIZATION #
    ###################################
    def analyze_pre_treatment(self, prefix, output_dir):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, etc.
        """
        print("\tPre-treatment Correlations:")
        #print(self.merged_data.info(verbose=True))
        # variable types
        categorical_vars = [
            "Location",
            "Symptoms",
            "Histology",
            "Treatment Type",
            # "Age Group at Progression",
            "Age Group at Diagnosis",
            "BRAF Status",
            "Sex",
            #"Tumor Classification",
            "Received Treatment",
            #"Change Speed",
            "Change Type",
            "Change Trend",
            "Change Acceleration",
        ]
        numerical_vars = [
            "Age",
            "Age at First Diagnosis",
            "Age at Last Clinical Follow-Up",
            "Days Between Scans",
            "Volume",
            "Normalized Volume",
            "Volume Change",
            "Volume Change Rate",
            "Baseline Volume",
            "Follow-Up Time",
            "Age Median",
            "Volume Median",
            "Volume Change Median",
            "Volume Change Rate Median",
            "Follow-Up Time Median",
            "Days Between Scans Median",
        ]
        outcome_var = "Patient Classification Binary"
        # for full blown out comparison uncomment the following lines
        correlation_dir = os.path.join(output_dir, "correlations")
        os.makedirs(correlation_dir, exist_ok=True)
        # for num_var in numerical_vars:
        #     for cat_var in categorical_vars:
        #         if self.merged_data[cat_var].nunique() == 2:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type="t-test",
        #             )
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type="point-biserial",
        #             )
        #         else:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type=None,
        #             )
            
        #     filtered_vars = [
        #         var
        #         for var in numerical_vars
        #         if not var.startswith(("Volume Change ", "Volume ", "Normalized"))
        #     ]
        #     for other_num_var in filtered_vars:
        #         if other_num_var != num_var:
        #             self.analyze_correlation(
        #                 num_var,
        #                 other_num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type="Spearman",
        #             )
        #             self.analyze_correlation(
        #                 num_var,
        #                 other_num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type="Pearson",
        #             )
        
        # aggregated_data = (
        #     self.merged_data.sort_values("Age").groupby("Patient_ID", as_index=False).last()
        # )
        # for cat_var in categorical_vars:
        #     for other_cat_var in categorical_vars:
        #         if cat_var != other_cat_var:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 other_cat_var,
        #                 aggregated_data,
        #                 prefix,
        #                 correlation_dir,
        #                 test_type=None
        #             )

        ##############################################
        ##### Cohort Table with basic statistics #####
        ##############################################
        if self.cohort == "JOINT":
            print("\t\tCreating cohort table:")
            cohort_table = self.create_cohort_table(
                categorical_vars=categorical_vars, continuous_vars=numerical_vars
            )
            print(cohort_table)

        ####################################################################
        ##### Univariate analysis, logistic regression and forest plot #####
        ####################################################################
        patient_constant_vars = [
            "Location",
            "Symptoms",
            "Histology",
            "BRAF Status",
            "Sex",
            "Received Treatment",
            "Baseline Volume cm3",
            #"Treatment Type",
            #"Age Median",
            #"Age Group at Progression",
            "Age Group at Diagnosis",
            #"Change Speed",
            "Coefficient of Variation",
            "Relative Volume Change Pct",
            #"Cumulative Volume Change Pct",
            "Change Type",
            "Change Trend",
            "Change Acceleration",
        ]
        pooled_results_uni = pooled_results_multi = pd.DataFrame(
            columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
        )
        print("\t\tUnivariate Analysis:")
        for variable in patient_constant_vars:
            print(f"\t\tAnalyzing {variable}...")
            pooled_results_uni = self.univariate_analysis(
                variable,
                outcome_var,
                pooled_results_uni,
                categorical_vars,
            )
        self.plot_forest_plot(pooled_results_uni, output_dir, categorical_vars)
        print("\t\tUnivariate Analysis done! Forest Plot saved.")

        #############################################
        ##### Multi-variate logistic regression #####
        #############################################
        print("\t\tMultivariate Analysis:")
        variable_combinations = [
            [
                "Location",
                "Symptoms",
                "Histology",
                "BRAF Status",
                "Sex",
                "Received Treatment",
                "Age Group at Diagnosis",
                "Baseline Volume cm3",
                "Coefficient of Variation",
            ],
            [   
                "Volume Change Rate",
                "Volume Change",
                "Volume",
                "Baseline Volume cm3",
                "Coefficient of Variation",
                "Relative Volume Change Pct",
                "Cumulative Volume Change Pct",
                "Change Type",
                "Change Trend",
                "Change Acceleration",
            ],  # Categorical variables

        ]
        pooled_results_multi = pd.DataFrame(
            columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
        )
        for combo in variable_combinations:
            pooled_results_multi = self.multivariate_analysis(
                combo, outcome_var, pooled_results_multi, categorical_vars
            )
            self.plot_forest_plot(
                pooled_results_multi,
                output_dir,
                categorical_vars,
                analysis_type="Multivariate",
                combo=combo,
            )
            pooled_results_multi.drop(pooled_results_multi.index, inplace=True) # enable to clear up th epooled results and not have a cumulative forest plot
        print("\t\tMulti-variate Analysis done! Forest Plots saved.")

    def analyze_correlation(
        self, x_val, y_val, data, prefix, output_dir, test_type
    ):
        """
        Perform and print the results of a statistical test to analyze the correlation
        between two variables.

        Parameters:
        - x_val (str): The name of the first variable.
        - y_val (str): The name of the second variable.
        - data (DataFrame): The data containing the variables.
        - prefix (str): The prefix to be used for naming visualizations.
        - test_type (str): The statistical method to be used.

        Updates the class attributes with the results of the test and prints the outcome.
        """
        test_result, coef, p_val = None, None, None  # Initialize test_result
        x_dtype = data[x_val].dtype
        y_dtype = data[y_val].dtype

        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(
            y_dtype
        ):
            if test_type == "Spearman":
                coef, p_val = spearman_correlation(data[x_val], data[y_val])
                test_result = (coef, p_val)
            elif test_type == "Pearson":
                coef, p_val = pearson_correlation(data[x_val], data[y_val])
                test_result = (coef, p_val)
        elif pd.api.types.is_categorical_dtype(
            x_dtype
        ) and pd.api.types.is_numeric_dtype(y_dtype):
            categories = data[x_val].nunique()
            if categories == 2:
                if test_type == "t-test":
                    if self.check_assumptions(x_val, y_val, data, "t-test"):
                        t_stat, p_val = ttest(data, x_val, y_val)
                        test_result = (t_stat, p_val)
                    else:
                        print(
                            f"\t\tCould not perform analysis on {x_val} and {y_val} due to unmet assumptions checks."
                        )
                if test_type == "point-biserial":
                    coef, p_val = point_bi_serial(data, x_val, y_val)
                    test_result = (coef, p_val)
            else:
                #ANOVA
                if self.check_assumptions(x_val, y_val, data, "ANOVA"):
                    f_stat, p_val = f_one(data, x_val, y_val)
                    test_result = (f_stat, p_val)
                    test_type = "ANOVA"
                #Kruskal-Wallis
                else:
                    test_stat, p_val = kruskal_wallis_test(data, x_val, y_val)
                    test_result = (test_stat, p_val)
                    test_type = "Kruskal-Wallis"

        elif pd.api.types.is_categorical_dtype(
            x_dtype
        ) and pd.api.types.is_categorical_dtype(y_dtype):
            # Fisher's Exact Test
            if data[x_val].nunique() == 2 and data[y_val].nunique() == 2:
                odds_ratio, p_val = fisher_exact_test(data, x_val, y_val)
                test_result = (odds_ratio, p_val)
                test_type = "Fisher's Exact"
            # Chi-squared Test
            else:
                chi2, p_val, _, _ = chi_squared_test(data, x_val, y_val)
                test_result = (chi2, p_val)
                test_type = "Chi-squared"

        if test_result and test_result[1] < 0.05:
            print(
               f"\t\t{x_val} and {y_val} - {test_type.title()} Test: Statistic={test_result[0]},"
               f" P-value={test_result[1]}"
            )
            self.visualize_statistical_test(
                x_val,
                y_val,
                data,
                test_result,
                prefix,
                output_dir,
                test_type,
            )
        elif test_result and not test_result[1] < 0.05:
            print(
                f"\t\t{x_val} and {y_val} - {test_type.title()} -> (Not significant)"
            )
        else:
            print(
                f"\t\tCould not perform analysis on {x_val} and {y_val} due to incompatible data"
                " types."
            )

        # save all of the p-values and coefficients in a dictionary call results
        self.results[(x_val, y_val)] = test_result
        
    def visualize_statistical_test(
        self,
        x_val,
        y_val,
        data,
        test_result,
        prefix,
        output_dir,
        test_type,
    ):
        """
        Visualize the result of a statistical test, including correlation heatmaps.

        Parameters:
            x_val, y_val (str): Column names for the variables analyzed.
            data (DataFrame): The dataframe containing the data.
            test_result (tuple): The result of the statistical test (e.g., (statistic, p_value)).
            prefix (str): The prefix to be used for naming visualizations.
            test_type (str): The type of statistical test ('correlation', 't-test',
            'anova', 'chi-squared').
        """
        stat, p_val = test_result
        title = f"{x_val} vs {y_val} ({test_type.capitalize()}) \n"
        num_patients = data["Patient_ID"].nunique()

        units = {
            "Age": "days",
            "Age Group": "days",
            "Date": "date",
            "Volume": "mm³",
            "Baseline Volume": "mm³",
        }

        x_unit = units.get(x_val, "")
        y_unit = units.get(y_val, "")

        # Plot based on test type
        if test_type in ["ANOVA", "Kruskal-Wallis"]:
            sns.violinplot(x=x_val, y=y_val, data=data)
            title += f"Statistic: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type in ["point-biserial", "t-test"]:
            sns.boxplot(x=x_val, y=y_val, data=data)
            title += f"Correlation Coefficient: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type in ["Spearman", "Pearson"]:
            sns.scatterplot(x=x_val, y=y_val, data=data)
            sns.regplot(x=x_val, y=y_val, data=data, scatter=False, color="blue")
            title += (
                f"{test_type} correlation coefficient: {stat:.2f}, P-value:"
                f" {p_val:.3e} (N={num_patients})"
            )
        elif test_type in ["Chi-squared", "Fisher's Exact"]:
            contingency_table = pd.crosstab(data[y_val], data[x_val])
            sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="g")
            if test_type == "Chi-squared":
                title += f"Chi2: {stat:.2f}, P-value: {p_val:.3e}, (N={num_patients})"
            else:
                title += f"Odds Ratio: {stat:.2f}, P-value: {p_val:.3e}, (N={num_patients})"
        
        plt.title(title)
        if x_unit:
            plt.xlabel(f"{x_val} [{x_unit}]")
        else:
            plt.xlabel(x_val)
        if y_unit:
            plt.ylabel(f"{y_val} [{y_unit}]")
        else:
            plt.ylabel(y_val)
        plt.tight_layout()

        save_file = os.path.join(
            output_dir, f"{prefix}_{x_val}_vs_{y_val}_{test_type}.png"
        )
        plt.savefig(save_file)
        plt.close()

    def check_assumptions(self, x_val, y_val, data, test_type):
        """
        Function to check the assumptions of the statistical tests.
        """
        if test_type in ["t-test", "ANOVA"]:
            categories = data[x_val].nunique()
            if categories == 2:
                group1 = data[data[x_val] == data[x_val].unique()[0]][y_val]
                group2 = data[data[x_val] == data[x_val].unique()[1]][y_val]
                if len(group1) < 3 or len(group2) < 3:
                    return False
                _, p_norm1 = shapiro(group1)
                _, p_norm2 = shapiro(group2)
                _, p_equal_var = levene(group1, group2)
                return p_norm1 > 0.05 and p_norm2 > 0.05 and p_equal_var > 0.05
            else:
                groups = [data[data[x_val] == category][y_val] for category in data[x_val].unique()]
                if any(len(group) < 3 for group in groups):
                    return False
                normality_tests = [shapiro(group)[1] for group in groups]
                equal_variance_test = levene(*groups)[1] > 0.05
                return all(p > 0.05 for p in normality_tests) and equal_variance_test
        elif test_type in ["Spearman", "Pearson"]:
            if len(data[x_val]) < 3 or len(data[y_val]) < 3:
                return False
            _, p_norm_x = shapiro(data[x_val])
            _, p_norm_y = shapiro(data[y_val])
            return p_norm_x > 0.05 and p_norm_y > 0.05
        else:
            return True

    def univariate_analysis(self, variable, outcome_var, pooled_results_uni, cat_vars):
        """
        Perform univariate logistic regression analysis for a given variable.
        """
        X, y = self.prepare_data_for_analysis(variable, outcome_var, cat_vars)

        if X is not None and not X.empty:
            try:
                # calculate_vif(X)
                result = logistic_regression_analysis(y, X)
                # print(result.summary2())
                pooled_results_uni = self.pool_results(
                    result, variable, pooled_results_uni, cat_vars
                )
                print(f"\t\t\tModel fitted successfully with {variable}.")
            except ExceptionGroup as e:
                print(f"\t\tError fitting model with {variable}: {e}")
        else:
            print(f"\t\tNo data available for {variable}.")

        return pooled_results_uni

    def prepare_data_for_analysis(self, variables, outcome_var, cat_vars):
        """
        Prepare the data for univariate logistic regression analysis.
        This function handles patient-constant and time-varying variables differently.
        """
        if isinstance(
            variables, str
        ):  # For univariate case where a single variable string is passed
            variables = [variables]

        if len(variables) == 1:
            # Univariate case
            variable = variables[0]
            data_agg = (
                self.merged_data.groupby("Patient_ID")
                .agg({variable: "first", outcome_var: "first"})
                .reset_index()
            )
        else:
            # Multivariate case
            data_agg = self.merged_data[
                ["Patient_ID"] + variables + [outcome_var]
            ].copy()

            # Separate categorical and numerical variables
            cat_vars_subset = [var for var in variables if var in cat_vars]
            num_vars_subset = [var for var in variables if var not in cat_vars]

            # Aggregate categorical variables by taking the mode, numerical the man and outcome the first value per patient
            if cat_vars_subset:
                cat_agg = data_agg.groupby("Patient_ID")[cat_vars_subset].agg(
                    lambda x: x.value_counts().index[0]
                )
                data_agg = data_agg.drop(columns=cat_vars_subset).merge(
                    cat_agg, on="Patient_ID"
                )
            if num_vars_subset:
                num_agg = data_agg.groupby("Patient_ID")[num_vars_subset].mean()
                data_agg = data_agg.drop(columns=num_vars_subset).merge(
                    num_agg, on="Patient_ID"
                )

            outcome_agg = data_agg.groupby("Patient_ID")[outcome_var].first()
            data_agg = data_agg.drop(columns=[outcome_var]).merge(
                outcome_agg, on="Patient_ID"
            )

        for variable in variables:
            # For categorical variables, convert them to dummy variables
            if variable in cat_vars:
                reference_category = data_agg[variable].mode()[0]
                ref_count = (data_agg[variable] == reference_category).sum()
                print("\t\t\tReference category: ", reference_category)
                self.reference_categories[variable] = (reference_category, ref_count)
                data_agg[variable] = data_agg[variable].astype(str)
                dummies = pd.get_dummies(
                    data_agg[variable], prefix=variable, drop_first=False
                )
                if f"{variable}_{reference_category}" in dummies.columns:
                    dummies.drop(
                        columns=[f"{variable}_{reference_category}"], inplace=True
                    )
                data_agg = pd.concat(
                    [data_agg.drop(columns=[variable]), dummies], axis=1
                )
                for col in dummies.columns:
                    data_agg[col] = data_agg[col].astype(int)
            else:
                data_agg[variable] = pd.to_numeric(data_agg[variable], errors="coerce")
                if (data_agg[variable] <= 0).any():
                    # Handle zeros or negative values if necessary, e.g., by adding a small constant
                    # data_agg[variable] += 1
                    data_agg[variable] = data_agg[variable].replace(0, 0.1)
                    data_agg[variable] = data_agg[variable].clip(lower=0.1)
                # Apply log transformation
                data_agg[variable] = np.log(data_agg[variable])

        # Ensure outcome_var is binary numeric, reduce to relevant columns, check for missing values
        data_agg[outcome_var] = (
            pd.to_numeric(data_agg[outcome_var], errors="coerce").fillna(0).astype(int)
        )

        # drop patient ID and assign constant for regression
        data_agg = data_agg.drop(columns=["Patient_ID"], errors="ignore")
        data_agg = data_agg[
            [outcome_var] + [col for col in data_agg.columns if col != outcome_var]
        ]
        data_agg.dropna(inplace=True)
        if "const" not in data_agg.columns:
            data_agg = sm.add_constant(data_agg)

        if data_agg.empty:
            print("\t\tWarning: No data available. Recheck code and data structure.")
            return None, None
        else:
            y = data_agg[outcome_var]
            X = data_agg.drop(columns=[outcome_var], errors="ignore")
            return X, y

    def plot_forest_plot(
        self,
        pooled_results,
        output_dir,
        cat_vars,
        analysis_type="Univariate",
        combo=None,
    ):
        """
        Create a forest plot from the pooled results of univariate analyses.

        Args:
            pooled_results: DataFrame with columns 'Variable', 'OR', 'Lower', 'Upper', and 'p'.
            output_file: File path to save the forest plot image.
        """
        # print(pooled_results)
        expected_columns = {"MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"}
        if not expected_columns.issubset(pooled_results.columns):
            missing_cols = expected_columns - set(pooled_results.columns)
            raise ValueError(
                f"The DataFrame is missing the following required columns: {missing_cols}"
            )
        # Exclude 'Reference' entries from calculations
        reference_mask = pooled_results["Subcategory"].str.contains("Reference")
        references = pooled_results[reference_mask]
        filtered_results = pooled_results[~reference_mask]

        # sort pooled results alphabetically, then clear out non-positive and infinite values
        filtered_results = filtered_results[
            (filtered_results["OR"] > 0)
            & (filtered_results["Lower"] > 0)
            & (filtered_results["Upper"] > 0)
        ]
        filtered_results.replace([np.inf, -np.inf], np.nan, inplace=True)
        filtered_results.dropna(subset=["OR", "Lower", "Upper", "p"], inplace=True)
        if not filtered_results.empty:
            max_hr = 100  # remove outliers
            filtered_results = filtered_results[filtered_results["Upper"] <= max_hr]

        # Include 'Reference' entries for plotting without affecting calculations
        final_results = pd.concat([filtered_results, references], ignore_index=True)
        final_results.sort_values(
            by=["MainCategory", "Subcategory"], ascending=[False, False], inplace=True
        )
        final_results.reset_index(drop=True, inplace=True)

        # General plot settings + x parameters
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.3, right=0.7)
        ax.set_xscale("log")
        ax.set_xlim(left=0.01, right=100)
        ax.set_xlabel("<-- Lower Risk of Progression | Higher Risk of Progression -->")
        ax.axvline(x=1, linestyle="--", color="blue", lw=1)

        # Categories handling and colors
        unique_main_categories = final_results["MainCategory"].unique()
        colormap = plt.get_cmap("tab20")
        colors = [colormap(i) for i in range(len(unique_main_categories))]
        category_colors = {
            cat: color for cat, color in zip(unique_main_categories, colors)
        }

        # annotations on the right
        ax.margins(x=1)
        fig.canvas.draw()  # Need to draw the canvas to update axes positions

        # Get the bounds of the axes in figure space
        ax_bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Calculate the figure and axes widths in inches
        fig_width_inches = fig.get_size_inches()[0]
        axes_width_inches = ax_bounds.width
        annotation_x_position = ax_bounds.x1 + 0.01 * fig_width_inches

        # Annotations on the left
        copy_df = self.merged_data.copy()
        unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
        y_labels = []
        for i, row in enumerate(final_results.itertuples()):
            main_category = row.MainCategory
            subcategory = row.Subcategory
            if "(Reference)" in subcategory:
                _, count = self.reference_categories.get(main_category, (None, 0))
            else:
                if main_category in cat_vars:
                    count = unique_pat[main_category].value_counts().get(subcategory, 0)
                else:
                    count = len(unique_pat)
            label = f"{main_category} - {subcategory} - {count}"
            y_labels.append(label)

            # plotting
            if "(Reference)" not in subcategory:
                ax.errorbar(
                    row.OR,
                    i,
                    xerr=[[row.OR - row.Lower], [row.Upper - row.OR]],
                    fmt="o",
                    color=category_colors[main_category],
                    ecolor=category_colors[main_category],
                    elinewidth=1,
                    capsize=3,
                )
                ax.text(
                    annotation_x_position + (40 * axes_width_inches),
                    i,
                    f"{row.OR:.2f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
                ax.text(
                    annotation_x_position + (100 * axes_width_inches),
                    i,
                    f"({row.Lower:.2f}-{row.Upper:.2f})",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
                ax.text(
                    annotation_x_position + (600 * axes_width_inches),
                    i,
                    f"{row.p:.3f}" if row.p >= 0.01 else "<0.01",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
            else:
                ax.errorbar(
                    1.0,
                    i,
                    fmt="^",
                    color=category_colors[main_category],
                    capsize=3,
                )
                ax.text(
                    annotation_x_position + (40 * axes_width_inches),
                    i,
                    "Reference",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, ha="right")

        # titles on the plot
        ax.text(
            -0.35,
            1.01,
            "Variables and \n Subgroups",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            -0.2,
            1.01,
            "Count (n)",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.05,
            1.01,
            "OR",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.15,
            1.01,
            "95% CI",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.35,
            1.01,
            "P-val",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Add title, grid, and layout
        ax.set_title(f"{analysis_type} Analysis Forest Plot")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout(rect=[0, 0, 1, 0])

        if analysis_type == "Multivariate":
            combo_str = "_".join(combo)
            combo_len = len(combo_str)
            output_file = os.path.join(
                output_dir, f"{analysis_type}_{combo_len}_forest_plot.png"
            )
        else:
            output_file = os.path.join(output_dir, f"{analysis_type}_forest_plot.png")
        plt.savefig(output_file, dpi=300)
        plt.close()

    def pool_results(self, result, variables, pooled_results, cat_vars):
        """
        Pool the results of univariate analysis to create a forest plot.

        Args:
            result: Result object from univariate analysis.
            pooled_results: DataFrame to store pooled results.
        """
        if result is None:
            raise ValueError("No result object provided for pooling.")

        if not isinstance(pooled_results, pd.DataFrame) or pooled_results is None:
            pooled_results = pd.DataFrame(
                columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
            )

        if not isinstance(variables, list):
            variables = [variables]

        for variable in variables:
            if variable in cat_vars:
                reference_category = self.reference_categories.get(variable, None)
                if reference_category is None:
                    raise ValueError(f"No reference category set for {variable}")
                ref_row = pd.DataFrame(
                    {
                        "MainCategory": variable,
                        "Subcategory": f"{reference_category} (Reference)",
                        "OR": 1.0,
                        "Lower": np.nan,
                        "Upper": np.nan,
                        "p": np.nan,
                    },
                    index=[0],
                )
                pooled_results = pd.concat([pooled_results, ref_row], ignore_index=True)
        for idx in result.params.index:
            if idx != "const":
                parts = idx.split("_")
                main_category = parts[0]
                subcategory = " ".join(parts[1:]) if len(parts) > 1 else "Continuous"

                coef = result.params[idx]
                conf = result.conf_int().loc[idx].values
                p_val = result.pvalues[idx]

                new_row = pd.DataFrame(
                    {
                        "MainCategory": main_category,
                        "Subcategory": subcategory,
                        "OR": np.exp(coef),
                        "Lower": np.exp(conf[0]),
                        "Upper": np.exp(conf[1]),
                        "p": p_val,
                    },
                    index=[0],
                )
                pooled_results = pd.concat([pooled_results, new_row], ignore_index=True)
                print(f"\t\t\tPooled results updated with {idx}.")

        return pooled_results

    def multivariate_analysis(
        self, variables, outcome_var, pooled_results_multi, cat_vars
    ):
        """
        Perform multivariate logistic regression analysis for a given set of variables.
        """
        X, y = self.prepare_data_for_analysis(variables, outcome_var, cat_vars)

        if X is not None and not X.empty:
            try:
                result = logistic_regression_analysis(y, X)
                # print(result.summary2())
                print(f"\t\t\tModel fitted successfully with {variables}.")
                pooled_results_multi = self.pool_results(
                    result, variables, pooled_results_multi, cat_vars
                )

            except ExceptionGroup as e:
                print(f"\t\tError fitting model with {variables}: {e}")
        else:
            print(f"\t\tNo data available for {variables}.")

        return pooled_results_multi

    #####################################
    # CURVE PLOTTING AND TREND ANALYSIS #
    #####################################

    def trajectories(self, prefix, output_dir):
        """
        Plot trajectories of patients.

        Parameters:
        - prefix (str): Prefix used for naming the output files.
        - output_dir (str): Directory to save the output plots.
        """
        print("\tModeling growth trajectories:")
        # Data preparation for modeling
        pre_treatment_data = self.merged_data.copy()
        pre_treatment_data.sort_values(by=["Patient_ID", "Age"], inplace=True)
        pre_treatment_data["Time since First Scan"] = pre_treatment_data.groupby(
            "Patient_ID"
        )["Age"].transform(lambda x: (x - x.iloc[0]))
        pre_treatment_data.reset_index(drop=True, inplace=True)

        self.merged_data.sort_values(by=["Patient_ID", "Age"], inplace=True)
        self.merged_data["Time since First Scan"] = pre_treatment_data[
            "Time since First Scan"
        ]
        self.merged_data.reset_index(drop=True, inplace=True)

        # Error handling for sample size
        sample_size = self.sample_size_plots
        if sample_size:
            # Sample a subset of patients if sample_size is provided
            unique_patient_count = pre_treatment_data["Patient_ID"].nunique()
            if sample_size > unique_patient_count:
                print(
                    f"\t\tSample size {sample_size} is greater than the number of unique patients"
                    f" {unique_patient_count}. Using {unique_patient_count} instead."
                )
                sample_size = unique_patient_count

            sample_ids = (
                pre_treatment_data["Patient_ID"].drop_duplicates().sample(n=sample_size)
            )
            pre_treatment_data = pre_treatment_data[
                pre_treatment_data["Patient_ID"].isin(sample_ids)
            ]

        # Plot the overlaying curves plots
        volume_change_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_change_trajectories_plot.png"
        )
        plot_individual_trajectories(
            volume_change_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume Change",
            unit="mm^3",
        )
        volume_change_pct_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_change_pct_trajectories_plot.png")
        plot_individual_trajectories(
            volume_change_pct_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume Change Pct",
            unit="%",)
        volume_change_rate_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_change_rate_trajectories_plot.png"
        )
        plot_individual_trajectories(
            volume_change_rate_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume Change Rate",
            unit="mm^3 / day",
        )
        volume_change_rate_pct_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_change_rate_pct_trajectories_plot.png")
        plot_individual_trajectories(
            volume_change_rate_pct_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume Change Rate Pct",
            unit="% / day")
        normalized_volume_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_normalized_volume_trajectories_plot.png"
        )
        plot_individual_trajectories(
            normalized_volume_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Normalized Volume",
            unit="mm^3",
        )
        volume_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_trajectories_plot.png"
        )
        plot_individual_trajectories(
            volume_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume",
            unit="mm^3",
        )

        category_list = [
            "Sex",
            "BRAF Status",
            "Received Treatment",
            "Location",
            "Treatment Type",
            "Symptoms",
            "Histology",
        ]

        category_out = os.path.join(output_dir, "category_plots")
        os.makedirs(category_out, exist_ok=True)

        for cat in category_list:
            cat_volume_change_name = os.path.join(
                category_out, f"{prefix}_{cat}_volume_change_trajectories_plot.png"
            )
            plot_individual_trajectories(
                cat_volume_change_name,
                plot_data=pre_treatment_data,
                column="Volume Change",
                category_column=cat,
                unit="%",
            )
            cat_normalized_volume_name = os.path.join(
                category_out, f"{prefix}_{cat}_normalized_volume_trajectories_plot.png"
            )
            plot_individual_trajectories(
                cat_normalized_volume_name,
                plot_data=pre_treatment_data,
                column="Normalized Volume",
                category_column=cat,
                unit="mm^3",
            )
            cat_volume_change_rate_trajectories_plot = os.path.join(
                category_out, f"{prefix}_{cat}_volume_change_rate_trajectories_plot.png"
            )
            plot_individual_trajectories(
                cat_volume_change_rate_trajectories_plot,
                plot_data=pre_treatment_data,
                column="Volume Change Rate",
                category_column=cat,
                unit="% / day",
            )

        # Trend analysis and classifciation of patients
        self.trend_analysis(pre_treatment_data, output_dir, prefix)

    def trend_analysis(self, data, output_dir, prefix):
        """
        Classify patients based on their tumor growth trajectories
        into progressors, stable or regressors.
        """
        patients_ids = data["Patient_ID"].unique()
        # Edit this to have other plots
        # column_name = "Volume Change"
        column_name = "Normalized Volume"

        print("\tStarting Trend Analysis:")
        patient_classifications = {
            patient_id: classify_patient(
                data,
                patient_id,
                column_name,
                self.progression_threshold,
            )
            for patient_id in patients_ids
        }
        data["Classification"] = data["Patient_ID"].map(patient_classifications)
        data["Patient Classification Binary"] = (
            pd.to_numeric(
                data["Classification"].apply(lambda x: 1 if x == "Progressor" else 0),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        # Save to original dataframe
        classifications_series = pd.Series(patient_classifications)
        self.merged_data["Patient Classification"] = (
            self.merged_data["Patient_ID"]
            .map(classifications_series)
            .astype("category")
        )
        self.merged_data["Patient Classification Binary"] = data[
            "Patient Classification Binary"
        ]

        # Plots
        output_filename = os.path.join(output_dir, f"{prefix}_trend_analysis.png")
        plot_trend_trajectories(data, output_filename, column_name, unit="mm^3")
        print("\t\tSaved trend analysis plot.")

    def analyze_tumor_stability(
        self,
        data,
        output_dir,
        #volume_weight=0.5,
        #growth_weight=0.5,
        #change_threshold=20,
    ):
        """
        Analyze the stability of tumors based on their growth rates and volume changes.

        Parameters:
        - data (DataFrame): Data containing tumor growth and volume information.
        - output_dir (str): Directory to save the output plots.
        - volume_weight (float): Clinical significance weight for tumor volume stability.
        - growth_weight (float): Clinical significance weight for tumor growth stability.

        Returns:
        - data (DataFrame): Data with added Stability Index and Tumor Classification.
        """
        print("\tAnalyzing tumor stability:")
        data = data.copy()
        # volume_column = "Normalized Volume"
        # volume_change_column = "Volume Change"
        # data = calculate_group_norms_and_stability(
        #     data, volume_column, volume_change_column
        # )
        # # Calculate the overall volume change for each patient
        # data["Overall Volume Change"] = data["Patient_ID"].apply(
        #     lambda x: calculate_percentage_change(data, x, volume_column)
        # )
        # # Calculate the Stability Index using weighted scores
        # data["Stability Index"] = (
        #     volume_weight * data["Volume Stability Score"]
        #     + growth_weight * data["Change Stability Score"]
        # )

        # # Normalize the Stability Index to have a mean of 1
        # data["Stability Index"] /= np.mean(data["Stability Index"])

        # significant_volume_change = (
        #     abs(data["Overall Volume Change"]) >= change_threshold
        # )
        # stable_subset = data.loc[~significant_volume_change, "Stability Index"]
        # mean_stability_index = stable_subset.mean()
        # std_stability_index = stable_subset.std()
        # num_std_dev = 2
        # stability_threshold = mean_stability_index + (num_std_dev * std_stability_index)

        # data["Tumor Classification"] = data.apply(
        #     lambda row: "Unstable"
        #     if abs(row["Overall Volume Change"]) >= change_threshold
        #     or row["Stability Index"] > stability_threshold
        #     else "Stable",
        #     axis=1,
        # ).astype("category")

        tumor_stability_out = os.path.join(output_dir, "tumor_stability_plots")
        os.makedirs(tumor_stability_out, exist_ok=True)
        data_n = calculate_stability_index(data)
        visualize_individual_indexes(data_n, tumor_stability_out)
        visualize_stability_index(data_n, tumor_stability_out)
        visualize_volume_change(data_n, tumor_stability_out)
        visualize_ind_indexes_distrib(data_n, tumor_stability_out)
        roc_curve_and_auc(data_n, tumor_stability_out)
        #grid_search_weights(data_n)
        # Map the 'Stability Index' and 'Tumor Classification' to the
        # self.merged_data using the maps
        # m_data = pd.merge(
        #     self.merged_data,
        #     data[
        #         [
        #             "Patient_ID",
        #             "Age",
        #             "Stability Index",
        #             "Tumor Classification",
        #             "Overall Volume Change",
        #         ]
        #     ],
        #     on=["Patient_ID", "Age"],
        #     how="left",
        # )

        # self.merged_data = m_data
        # self.merged_data.reset_index(drop=True, inplace=True)
        # visualize_tumor_stability(
        #     data, output_dir, stability_threshold, change_threshold
        # )
        print("\t\tSaved tumor stability plots.")

    def printout_stats(self, output_file_path, prefix):
        """
        Descriptive statistics written to a file.

        Parameters:
        - output_file_path (str): Path to the output path.
        - prefix (str): Prefix used for naming the output file.
        """
        filename = f"{prefix}_summary_statistics.txt"
        file_path = os.path.join(output_file_path, filename)
        with open(file_path, "w", encoding="utf-8") as file:

            def write_stat(statement):
                file.write(statement + "\n")

            # Age
            median_age = self.merged_data["Age"].median()
            max_age = self.merged_data["Age"].max()
            min_age = self.merged_data["Age"].min()
            write_stat(f"\t\tMedian Age: {median_age} days")
            write_stat(f"\t\tMaximum Age: {max_age} days")
            write_stat(f"\t\tMinimum Age: {min_age} days")

            # Days Between Scans
            median_days_between_scans = self.merged_data["Days Between Scans"].median()
            max_days_between_scans = self.merged_data["Days Between Scans"].max()
            unique_days = self.merged_data["Days Between Scans"].unique()
            next_smallest = min(day for day in unique_days if day > 0)

            write_stat(f"\t\tMedian Days Between Scans: {median_days_between_scans} days")
            write_stat(f"\t\tMaximum Days Between Scans: {max_days_between_scans} days")
            write_stat(f"\t\tMinimum Days Between Scans: {next_smallest} days")

            # Sex, Received Treatment, Symptoms, Location,
            # Patient Classification, Treatment Type
            copy_df = self.merged_data.copy()
            unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
            counts_braf = unique_pat["BRAF Status"].value_counts()
            counts_sex = unique_pat["Sex"].value_counts()
            counts_received_treatment = unique_pat["Received Treatment"].value_counts()
            counts_symptoms = unique_pat["Symptoms"].value_counts()
            counts_histology = unique_pat["Histology"].value_counts()
            counts_location = unique_pat["Location"].value_counts()
            counts_patient_classification = unique_pat[
                "Patient Classification"
            ].value_counts()
            counts_treatment_type = unique_pat["Treatment Type"].value_counts()

            write_stat(f"\t\tReceived Treatment: {counts_received_treatment}")
            write_stat(f"\t\tSymptoms: {counts_symptoms}")
            write_stat(f"\t\tHistology: {counts_histology}")
            write_stat(f"\t\tLocation: {counts_location}")
            write_stat(f"\t\tSex: {counts_sex}")
            write_stat(f"\t\tPatient Classification: {counts_patient_classification}")
            write_stat(f"\t\tTreatment Type: {counts_treatment_type}")
            write_stat(f"\t\tBRAF Status: {counts_braf}")

            # Volume Change
            filtered_data = self.merged_data[self.merged_data["Volume Change"] != 0]
            median_volume_change = filtered_data["Volume Change"].median()
            max_volume_change = filtered_data["Volume Change"].max()
            min_volume_change = filtered_data["Volume Change"].min()
            write_stat(f"\t\tMedian Volume Change: {median_volume_change} %")
            write_stat(f"\t\tMaximum Volume Change: {max_volume_change} %")
            write_stat(f"\t\tMinimum Volume Change: {min_volume_change} %")

            # Volume Change Rate
            filtered_data = self.merged_data[
                self.merged_data["Volume Change Rate"] != 0
            ]
            median_volume_change_rate = filtered_data["Volume Change Rate"].median()
            max_volume_change_rate = filtered_data["Volume Change Rate"].max()
            min_volume_change_rate = filtered_data["Volume Change Rate"].min()
            write_stat(
                f"\t\tMedian Volume Change Rate: {median_volume_change_rate} %/day"
            )
            write_stat(
                f"\t\tMaximum Volume Change Rate: {max_volume_change_rate} %/day"
            )
            write_stat(
                f"\t\tMinimum Volume Change Rate: {min_volume_change_rate} %/day"
            )

            # Normalized Volume
            median_normalized_volume = self.merged_data["Normalized Volume"].median()
            max_normalized_volume = self.merged_data["Normalized Volume"].max()
            min_normalized_volume = self.merged_data["Normalized Volume"].min()
            write_stat(f"\t\tMedian Normalized Volume: {median_normalized_volume} mm^3")
            write_stat(f"\t\tMaximum Normalized Volume: {max_normalized_volume} mm^3")
            write_stat(f"\t\tMinimum Normalized Volume: {min_normalized_volume} mm^3")

            # Volume
            mm3_to_cm3 = 1000
            median_volume = self.merged_data["Volume"].median()
            max_volume = self.merged_data["Volume"].max()
            min_volume = self.merged_data["Volume"].min()
            write_stat(f"\t\tMedian Volume: {median_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMaximum Volume: {max_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMinimum Volume: {min_volume / mm3_to_cm3} cm^3")

            # Baseline volume
            median_baseline_volume = self.merged_data["Baseline Volume"].median()
            max_baseline_volume = self.merged_data["Baseline Volume"].max()
            min_baseline_volume = self.merged_data["Baseline Volume"].min()
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
            median_follow_up = self.merged_data["Follow-Up Time"].median()
            max_follow_up = self.merged_data["Follow-Up Time"].max()
            min_follow_up = self.merged_data["Follow-Up Time"].min()
            median_follow_up_y = median_follow_up / 365.25
            max_follow_up_y = max_follow_up / 365.25
            min_follow_up_y = min_follow_up / 365.25
            write_stat(f"\t\tMedian Follow-Up Time: {median_follow_up_y:.2f} years")
            write_stat(f"\t\tMaximum Follow-Up Time: {max_follow_up_y:.2f} years")
            write_stat(f"\t\tMinimum Follow-Up Time: {min_follow_up_y:.2f} years")

            # get the ids of the patients with the three highest normalized volumes that do not repeat
            top_normalized_volumes = self.merged_data.nlargest(3, "Normalized Volume")
            top_volumes = self.merged_data.nlargest(3, "Volume")
            self.merged_data["Absolute Volume Change"] = self.merged_data[
                "Volume Change"
            ].abs()
            self.merged_data["Absolute Volume Change Rate"] = self.merged_data[
                "Volume Change Rate"
            ].abs()
            top_volume_changes = self.merged_data.nlargest(3, "Absolute Volume Change")
            top_volume_change_rates = self.merged_data.nlargest(
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

        print(f"\t\tSaved summary statistics to {file_path}.")

    def generate_distribution_plots(self, output_dir):
        """
        Violin plots.
        """
        data = self.merged_data.copy()
        sns.set_palette(NORD_PALETTE)
        # Create a figure with subplots in a single column
        _, axs = plt.subplots(4, 1, figsize=(8, 18))
        # Violin plot for "Follow-Up Time" distribution per dataset
        data["Follow-Up Time (Years)"] = data["Follow-Up Time"] / 365.25
        sns.boxplot(x="Dataset", y="Follow-Up Time (Years)", data=data, ax=axs[0])
        axs[0].set_title("Distribution of Follow-Up Time")
        axs[0].set_xlabel("Dataset")
        axs[0].set_ylabel("Follow-Up Time [days]")
        # Violin plot for number of scans per patient per dataset
        scans_per_patient = (
            self.merged_data.groupby(["Dataset", "Patient_ID"])
            .size()
            .reset_index(name="Number of Scans")
        )
        # Filter out patients with less than 3 scans
        scans_per_patient = scans_per_patient[scans_per_patient["Number of Scans"] >= 3]
        sns.violinplot(
            x="Dataset", y="Number of Scans", data=scans_per_patient, ax=axs[1]
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
        axs[1].set_title("Distribution of Number of Scans per Patient")
        axs[1].set_xlabel("Dataset")
        axs[1].set_ylabel("Number of Scans")
        # Violin plot for follow-up interval distribution per dataset
        sns.violinplot(y="Dataset", x="Days Between Scans", data=data, ax=axs[2])
        axs[2].set_title("Distribution of Follow-Up Intervals")
        axs[2].set_ylabel("Dataset")
        axs[2].set_xlabel("Time Between Scans [days]")
        # Stacked bar plot for progression classification per dataset
        classification_counts = (
            data.groupby(["Dataset", "Patient Classification"]).size().unstack()
        )
        colors = [NORD_PALETTE[1], NORD_PALETTE[0], NORD_PALETTE[2]]
        classification_percentages = (
            classification_counts.div(classification_counts.sum(axis=1), axis=0) * 100
        )
        classification_percentages.plot(kind="barh", ax=axs[3], color=colors, stacked=True)
        axs[3].set_title("Patient Classification per Dataset")
        axs[3].set_ylabel("Dataset")
        axs[3].set_xlabel("Percentage")
        axs[3].legend(title="Patient Classification", loc="center", ncol=3, fancybox=True, shadow=True)
        # Add percentages at the center of each bar
        for _, p in enumerate(axs[3].patches):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            axs[3].text(
                x + width / 2,
                y + height / 2,
                f"{width:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )
        # Adjust the spacing between subplots
        plt.tight_layout()
        # Display the plot
        file_name = os.path.join(output_dir, "dataset_comparison.png")
        plt.savefig(file_name, dpi=300)    
    
    def create_cohort_table(self, categorical_vars, continuous_vars):
        """
        Create a table comparing the two cohorts based on the variables of interest.
        """
        cohort_var = "Dataset"
        data = self.merged_data.copy()
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
                    if self.check_assumptions(var, var, cohort1_data, "t-test") and self.check_assumptions(var, var, cohort2_data, "t-test"):
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

        return cohort_table

    ##########################################
    # EFS RELATED ANALYSIS AND VISUALIZATION #
    ##########################################
    def time_to_event_analysis(self, prefix, output_dir, stratify_by=None):
        """
        Perform a Kaplan-Meier survival analysis on time-to-event data for tumor progression.

        Parameters:
        - prefix (str): Prefix used for naming the output file.

        The method fits the survival curve using the KaplanMeierFitter on the pre-treatment data,
        saves the plot image, and prints a confirmation message.
        """
        # pre_treatment_data as df copy for KM curves
        print("\tPerforming time-to-event analysis: Kaplan-Meier Survival Analysis.")        
        analysis_data_pre = self.merged_data.copy()
        progression_data = analysis_data_pre.groupby("Patient_ID").apply(
            self.calculate_progression
        )
        progression_data = progression_data.reset_index(drop=False)
        # consider on the one side the merging back to the first data frame and continue with the analysis dataframe at the same time
        self.merged_data = pd.merge(
            self.merged_data, progression_data, on="Patient_ID", how="left"
        )
        self.merged_data["Time Since Diagnosis"] = self.merged_data["Time Since Diagnosis"].astype("category")
        self.merged_data["Age Group at Progression"] = self.merged_data["Age Group at Progression"].astype("category")
        analysis_data_pre = pd.merge(
            analysis_data_pre,
            progression_data[["Patient_ID", "Age at First Progression", "Time to Progression"]],
            on="Patient_ID",
            how="left",
        )
        
        analysis_data_pre["Event_Occurred"] = ~analysis_data_pre[
            "Age at First Progression"
        ].isna()
        # Compare the results of both approaches and check if they match
        analysis_data_pre["Duration"] = np.where(
            analysis_data_pre["Event_Occurred"],
            analysis_data_pre["Time to Progression"],
            #analysis_data_pre["Age at First Progression"] - analysis_data_pre["Age at First Diagnosis"],
            analysis_data_pre["Follow-Up Time"],
            #np.nan
        )
        
        analysis_data_pre = analysis_data_pre.dropna(
            subset=["Duration", "Event_Occurred"]
        )

        for element in stratify_by:
            if element is not None:
                self.kaplan_meier_analysis(
                    analysis_data_pre, output_dir, element, prefix
                )
            else:
                self.kaplan_meier_analysis(analysis_data_pre, output_dir, prefix=prefix)
                self.cox_proportional_hazards_analysis(analysis_data_pre, output_dir)

    def kaplan_meier_analysis(self, data, output_dir, stratify_by=None, prefix=""):
        """
        Kaplan-Meier survival analysis for time-to-event data.
        """
        surv_dir = os.path.join(output_dir, "survival_plots")
        os.makedirs(surv_dir, exist_ok=True)
        colors = sns.color_palette(NORD_PALETTE, n_colors=len(NORD_PALETTE))
        analysis_data_pre = data.drop_duplicates(subset=['Patient_ID'], keep='first')
        kmf = KaplanMeierFitter()

        if stratify_by and stratify_by in analysis_data_pre.columns:
            unique_categories = analysis_data_pre[stratify_by].unique()
            if len(unique_categories) > 1:
                groups = []
                kmfs = []
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, category in enumerate(unique_categories):
                    category_data = analysis_data_pre[analysis_data_pre[stratify_by] == category]
                    if category_data.empty:
                        continue
                    kmf = KaplanMeierFitter()

                    kmf.fit(
                        category_data["Duration"],
                        event_observed=category_data["Event_Occurred"],
                        label=str(category),
                    )
                    kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=colors[i % len(colors)])
                    kmfs.append(kmf)
                    groups.append((category, category_data))
                
                plt.title(f"Stratified Survival Function by {stratify_by}", fontsize=14)
                plt.xlabel("Days since Diagnosis", fontsize=12)
                plt.ylabel("Survival Probability", fontsize=12)
                plt.legend(title=stratify_by, loc="best", fontsize=10)
                if kmfs:
                    add_at_risk_counts(*kmfs, ax=ax)
                
                # Save the combined plot
                survival_plot = os.path.join(surv_dir, f"{prefix}_survival_plot_category_{stratify_by}.png")
                plt.tight_layout()
                plt.savefig(survival_plot, dpi=300)
                plt.close(fig)
                print(f"\t\tSaved survival KaplanMeier curve for {stratify_by}.")

                # Perform pairwise log-rank tests
                pairwise_results = []
                for (cat1, group1), (cat2, group2) in combinations(groups, 2):
                    result = logrank_test(
                        group1["Duration"], group2["Duration"],
                        event_observed_A=group1["Event_Occurred"], event_observed_B=group2["Event_Occurred"]
                    )
                    p_value = result.p_value
                    display_p_value = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
                    pairwise_results.append((cat1, cat2, display_p_value))
                
                # Save pairwise log-rank test results
                results_file = os.path.join(surv_dir, f"{prefix}_pairwise_logrank_results_{stratify_by}.txt")
                with open(results_file, "w", encoding='utf-8') as f:
                    f.write("Combination\tp-value\n")
                    for cat1, cat2, display_p_value in pairwise_results:
                        f.write(f"{cat1} vs {cat2}\t{display_p_value}\n")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            kmf.fit(
                analysis_data_pre["Duration"],
                event_observed=analysis_data_pre["Event_Occurred"],
            )
            kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True)
            ax.set_title("Survival function of Tumor Progression")
            ax.set_xlabel("Days since Diagnosis")
            ax.set_ylabel("Survival Probability")
            add_at_risk_counts(kmf, ax=ax)
            
            # Save the plot
            survival_plot = os.path.join(surv_dir, f"{prefix}_survival_plot.png")
            plt.tight_layout()
            plt.savefig(survival_plot, dpi=300)
            plt.close(fig)
            print("\t\tSaved survival KaplanMeier curve.")
    
    def calculate_progression(self, group):
        """
        Calculate the age at first progression and time to progression for each patient.
        """
        baseline_volume = group.iloc[0]["Baseline Volume"]
        progression_threshold = baseline_volume * float(
            f"1.{self.progression_threshold}"
        )

        # Map categories to numerical values
        category_mapping = {
            "0-1 years": 1,
            "1-3 years": 3,
            "3-5 years": 5,
            "5-7 years": 7,
            "7-10 years": 10,
            "10+ years": 20
        }
        inverse_mapping = {v: k for k, v in category_mapping.items()}

        progression_mask = group["Volume"] >= progression_threshold
        if progression_mask.any():
            first_progression_index = progression_mask.idxmax()
            age_at_first_progression = group.loc[first_progression_index, "Age"]
            age_at_first_diagnosis = group.iloc[0]["Age at First Diagnosis"]
            time_to_progression = age_at_first_progression - age_at_first_diagnosis
            age_group_at_progression = group.loc[first_progression_index, "Age Group"]
            
            # calculate time period since diagnosis at progression
            time_period_since_diagnosis_value  = group.loc[first_progression_index, "Time Period Since Diagnosis"]
            if isinstance(time_period_since_diagnosis_value, str):
                time_period_since_diagnosis_numeric = category_mapping[time_period_since_diagnosis_value]
            else:
                time_period_since_diagnosis_numeric = time_period_since_diagnosis_value.map(category_mapping).astype(float)
            time_period_since_diagnosis_at_progression = inverse_mapping[time_period_since_diagnosis_numeric]
            
            volume_change_threshold = baseline_volume * float(
                f"1.{self.volume_change_threshold}"
            )  # 10%
            volume_change_mask = group["Volume"] >= volume_change_threshold

            if volume_change_mask.any():
                first_volume_change_index = volume_change_mask.idxmax()
                age_at_volume_change = group.loc[first_volume_change_index, "Age"]
                if (
                    pd.notnull(age_at_first_progression)
                    and age_at_first_progression > age_at_volume_change
                ):
                    time_gap = age_at_first_progression - age_at_volume_change
                else:
                    time_gap = 0
            else:
                time_gap = 0
        else:
            age_at_first_progression = np.nan
            age_at_volume_change = np.nan
            time_period_since_diagnosis_numeric = group["Time Period Since Diagnosis"].map(category_mapping).astype(float)
            max_time_period_numeric = time_period_since_diagnosis_numeric.max()
            time_period_since_diagnosis_at_progression = inverse_mapping[max_time_period_numeric]
            time_to_progression = np.nan
            time_gap = np.nan
            age_group_at_progression = group["Age Group"].iloc[-1]

        time_to_progression_years = time_to_progression / 365.25
        time_gap_years = time_gap / 365.25
        return pd.Series(
            {
                "Age at First Progression": age_at_first_progression,
                "Age Group at Progression": age_group_at_progression,
                "Age at Volume Change": age_at_volume_change,
                "Time to Progression": time_to_progression,
                "Time to Progression Years": time_to_progression_years,
                "Time Gap": time_gap,
                "Time Gap Years": time_gap_years,
                "Time Since Diagnosis": time_period_since_diagnosis_at_progression,
            }
        )

    def preprocess_data_hz_model(self, data):
        """
        Preprocess the data for Cox proportional hazards analysis.
        """
        print("\tPerforming time-to-event analysis: Cox-Hazard Survival Analysis.")        
        analysis_data = data.copy()
        list_of_columns = [
            "Location",
            #"Symptoms",
            "Histology",
            "BRAF Status",
            #"Sex",
            #"Received Treatment",
            #"Baseline Volume cm3",
            #"Treatment Type",
            #"Age Group at Diagnosis",
            #"Coefficient of Variation",
            #"Relative Volume Change Pct",
            #"Change Type",
            #"Change Trend",
            #"Change Acceleration",
            "Duration",
            "Event_Occurred"
        ]
        
        analysis_data.drop(columns=analysis_data.columns[~analysis_data.columns.isin(list_of_columns)], inplace=True)
        # Identify column types and adjust: continuous > scaling, categorical > encoding
        categorical_columns = analysis_data.select_dtypes(
            include=["category"]
        ).columns.tolist()
        analysis_data = pd.get_dummies(analysis_data, columns=categorical_columns)

        print(f"\t\tTotal number of rows before filtering: {analysis_data.shape[0]}")
        inf_mask = np.isinf(analysis_data).any(axis=1)
        if inf_mask.any():
            print(f"Warning: Removing {inf_mask.sum()} rows with infinite values.")
            analysis_data = analysis_data[~inf_mask]
            print(f"Total number of rows after INF filtering: {analysis_data.shape[0]}")
        nan_rows = analysis_data.isnull().any(axis=1)
        if nan_rows.any():
            print(f"Warning: {nan_rows.sum()} rows contain NaN values.")
            analysis_data = analysis_data[~nan_rows]
            print(f"Total number of rows after NaN filtering: {analysis_data.shape[0]}")

        # continuous_columns = analysis_data.select_dtypes(include=[np.number]).columns.difference(analysis_data.columns[analysis_data.columns.str.contains('_')]).tolist()
        # scaler = StandardScaler()
        # analysis_data[continuous_columns] = scaler.fit_transform(analysis_data[continuous_columns])

        if (analysis_data["Duration"] <= 0).any():
            raise ValueError(
                "The 'Duration' column contains non-positive values. Please ensure all durations are positive."
            )

        # calculate_vif(analysis_data.drop(columns=["Duration", "Event_Occurred"]), checks=True)
        # Handle missing values
        analysis_data = analysis_data.dropna(subset=["Duration", "Event_Occurred"])

        return analysis_data

    def cox_proportional_hazards_analysis(self, data, output_dir):
        """
        Cox proportional hazards analysis for time-to-event data.
        """
        analysis_data = self.preprocess_data_hz_model(data)
        print(analysis_data.dtypes)

        cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=4)
        cph.fit(analysis_data, duration_col="Duration", event_col="Event_Occurred")
        # check = proportional_hazard_test(cph, analysis_data)
        # check.print_summary()
        cph.print_summary()

        # Visualize the survival curves
        ax = cph.plot()
        ax.set_title("Survival Curves from Cox Model")
        ax.set_xlabel("Days since Diagnosis")
        ax.set_ylabel("Survival Probability")
        surv_dir = os.path.join(output_dir, "survival_plots")
        os.makedirs(surv_dir, exist_ok=True)
        cox_plot = os.path.join(surv_dir, "cox_proportional_hazards_plot.png")
        plt.savefig(cox_plot, dpi=300)
        plt.close()
        print("\t\tSaved Cox proportional hazards plot.")

        # Visualize the partial effects of covariates
        plt.figure(figsize=(8, 6))
        cph.plot_partial_effects_on_outcome(
            covariates=["BRAF Status_Fusion", "BRAF Status_V600E"],
            values=[[0, 1], [10, 20]],
            plot_baseline=False,
        )
        plt.title("Log-Log Survival Curves for Covariates")
        plt.xlabel("Log-Time")
        plt.ylabel("Log-Survival Probability")
        log_log_plot = os.path.join(surv_dir, "log_log_survival_plot.png")
        plt.savefig(log_log_plot, dpi=300)
        plt.close()

        # Visualize the concordance index
        print(analysis_data["Duration"].shape[0])
        print(analysis_data["Event_Occurred"].shape[0])
        c_index = concordance_index(
            analysis_data["Duration"],
            -cph.predict_expectation(analysis_data),
            analysis_data["Event_Occurred"],
        )
        plt.figure(figsize=(8, 6))
        plt.bar(["C-index"], [c_index], color="skyblue", edgecolor="black")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title(f"Concordance Index: {c_index:.3f}")
        c_index_plot = os.path.join(surv_dir, "c_index_plot.png")
        plt.savefig(c_index_plot, dpi=300)
        plt.close()

        print("\t\tSaved C-index plot.")

    def visualize_time_gap(self, output_dir):
        """
        Visualize the distribution of time gaps between volume change and progression.
        """
        progression_data = self.merged_data.groupby("Patient_ID").apply(
            self.calculate_progression
        )
        time_gap_data = progression_data["Time Gap"].dropna()
        time_gap_data = time_gap_data[time_gap_data > 0]  # Filter out non-positive values
        time_to_progression = progression_data["Time to Progression"].dropna()
        time_to_progression = time_to_progression[time_to_progression > 0]  # Filter out non-positive values
        
        _, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(
            time_gap_data, bins=25, kde=True, color="skyblue", edgecolor="black", ax=ax
        )

        plt.xlabel("Time Gap (Days)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Time Gap between Volume Change and Progression")

        # Add summary statistics to the plot
        mean_gap = np.mean(time_gap_data)
        median_gap = np.median(time_gap_data)
        ax.axvline(
            mean_gap, color="red", linestyle="--", label=f"Mean: {mean_gap:.2f} days"
        )
        ax.axvline(
            median_gap,
            color="green",
            linestyle="--",
            label=f"Median: {median_gap:.2f} days",
        )
        ax.legend()

        surv_dir = os.path.join(output_dir, "survival_plots")
        os.makedirs(surv_dir, exist_ok=True)
        time_gap_plot = os.path.join(surv_dir, "time_gap_plot.png")
        plt.savefig(time_gap_plot, dpi=300)
        plt.close()

        print("\t\tSaved time gap plot.")

        _, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(
            time_to_progression, bins=25, kde=True, color="skyblue", edgecolor="black", ax=ax
        )
        plt.xlabel("Time to Progression (Days)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Time to Progression")
        mean_progression = np.mean(time_to_progression)
        median_progression = np.median(time_to_progression)
        ax.axvline(
            mean_progression,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_progression:.2f} days",
        )
        ax.axvline(
            median_progression,
            color="green",
            linestyle="--",
            label=f"Median: {median_progression:.2f} days",
        )
        ax.legend()
        time_to_progression_plot = os.path.join(surv_dir, "time_to_progression_plot.png")
        plt.savefig(time_to_progression_plot, dpi=300)
        plt.close()
        print("\t\tSaved time to progression plot.")

    #################
    # MAIN ANALYSIS #
    #################

    def run_analysis(self, output_correlations, output_stats):
        """
        Run a comprehensive analysis pipeline consisting of data separation,
        sensitivity analysis, propensity score matching, main analysis, corrections
        for multiple comparisons, trend analysis, and feature engineering.

        This method orchestrates the overall workflow of the analysis process,
        including data preparation, statistical analysis, and results interpretation.
        """
        step_idx = 1

        if correlation_cfg.SENSITIVITY:
            print(f"Step {step_idx}: Performing Sensitivity Analysis...")

            pre_treatment_vars = [
                "Volume",
                "Volume Change",
            ]

            for pre_var in pre_treatment_vars:
                print(
                    f"\tPerforming sensitivity analysis on pre-treatment variables {pre_var}..."
                )
                self.merged_data = sensitivity_analysis(
                    self.merged_data,
                    pre_var,
                    z_threshold=correlation_cfg.SENSITIVITY_THRESHOLD,
                )

            step_idx += 1

        if correlation_cfg.PROPENSITY:
            print(f"Step {step_idx}: Performing Propensity Score Matching...")
            for data in [self.merged_data]:
                treatment_column = "Received Treatment"
                covariate_columns = ["Normalized Volume", "Volume Change"]

                data[treatment_column] = data[treatment_column].map({True: 1, False: 0})
                propensity_scores = calculate_propensity_scores(
                    data, treatment_column, covariate_columns
                )
                matched_data = perform_propensity_score_matching(
                    data, propensity_scores, treatment_column, caliper=self.caliper
                )
                smd_results = check_balance(
                    matched_data, covariate_columns, treatment_column
                )

                visualize_smds(smd_results, path=output_stats)

            step_idx += 1

        if correlation_cfg.ANALYSIS_PRE_TREATMENT:
            prefix = f"{self.cohort}_pre_treatment"
            print(f"Step {step_idx}: Starting main analyses {prefix}...")

            if self.merged_data.isnull().values.any():
                string_columns = self.merged_data.select_dtypes(
                    include=["string"]
                ).columns
                self.merged_data[string_columns] = self.merged_data[
                    string_columns
                ].apply(pd.to_numeric, errors="coerce")
                self.merged_data.replace(np.nan, np.inf, inplace=True)

            # Survival analysis
            stratify_by_list = [
                "Location",
                "Sex",
                "BRAF Status",
                "Age Group at Diagnosis",
                "Time Period Since Diagnosis",
                "Symptoms",
                "Histology",
                "Received Treatment",
                None,
            ]
            self.time_to_event_analysis(
                prefix, output_dir=output_stats, stratify_by=stratify_by_list
            )
            self.visualize_time_gap(output_dir=output_stats)

            # Trajectories & Trend analysis
            self.trajectories(prefix, output_dir=output_stats)

            # Tumor stability
            # self.analyze_tumor_stability(
            #     data=self.merged_data,
            #     output_dir=output_stats,
            #     volume_weight=correlation_cfg.VOLUME_WEIGHT,
            #     growth_weight=correlation_cfg.GROWTH_WEIGHT,
            #     change_threshold=correlation_cfg.CHANGE_THRESHOLD,
            # )
            #consistency_check(self.merged_data)

            if self.merged_data.isnull().values.any():
                print(self.merged_data)
                print(self.merged_data.isnull().sum())
                self.merged_data.replace(np.nan, np.inf, inplace=True)

            # Descriptive statistics for table1 in paper
            self.printout_stats(prefix=prefix, output_file_path=output_stats)
            if self.cohort == "JOINT":
                self.generate_distribution_plots(output_dir=output_stats)
            plot_histo_distributions(self.merged_data, output_dir=output_stats)

            # Correlations between variables
            #self.analyze_pre_treatment(
            #    prefix=prefix,
            #    output_dir=output_correlations,
            #)
            # self.perform_logistic_regression()

            step_idx += 1

        if correlation_cfg.CORRECTION:
            print(f"Step {step_idx}: Starting Corrections...")
            print("\tBonferroni Correction... ")
            alpha = correlation_cfg.CORRECTION_ALPHA

            corrected_p_values_bonf = bonferroni_correction(self.p_values, alpha=alpha)
            visualize_p_value_bonferroni_corrections(
                self.p_values, corrected_p_values_bonf, alpha, output_stats
            )
            print("\tBonferroni Correction done. ")

            print("\tFalse Discovery Rate Correction... ")
            corrected_p_values_fdr, is_rejected = fdr_correction(
                self.p_values, alpha=alpha
            )
            visualize_fdr_correction(
                self.p_values, corrected_p_values_fdr, is_rejected, alpha, output_stats
            )
            print("\tFalse Discovery Rate Correction done. ")

            step_idx += 1

        if correlation_cfg.FEATURE_ENG:
            print(f"Step {step_idx}: Starting Feature Engineering...")
            save_for_deep_learning(
                self.merged_data, output_stats, prefix="pre-treatment"
            )
            # save_for_deep_learning(self.post_treatment_data, output_stats, prefix="post-treatment")
            step_idx += 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if correlation_cfg.COHORT == "JOINT":
        data_paths = {
            "clinical_data_paths": correlation_cfg.CLINICAL_CSV_PATHS,
            "volumes_data_paths": [
                correlation_cfg.VOLUMES_DATA_PATHS["bch"],
                correlation_cfg.VOLUMES_DATA_PATHS["cbtn"],
            ],
        }
    else:
        data_paths = {
            "clinical_data_paths": [correlation_cfg.CLINICAL_CSV_PATHS[
                correlation_cfg.COHORT.lower()
            ]],
            "volumes_data_paths": [
                correlation_cfg.VOLUMES_DATA_PATHS[correlation_cfg.COHORT.lower()]
            ],
        }

    analysis = TumorAnalysis(data_paths, cohort=correlation_cfg.COHORT)

    os.makedirs(correlation_cfg.OUTPUT_DIR_CORRELATIONS, exist_ok=True)
    os.makedirs(correlation_cfg.OUTPUT_DIR_STATS, exist_ok=True)

    analysis.run_analysis(
        correlation_cfg.OUTPUT_DIR_CORRELATIONS, correlation_cfg.OUTPUT_DIR_STATS
    )
