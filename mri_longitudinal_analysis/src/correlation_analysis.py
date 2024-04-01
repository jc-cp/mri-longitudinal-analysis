# pylint: disable=too-many-lines
"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations, stability and trend analysis.
"""
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro
from cfg.src import correlation_cfg
from lifelines import KaplanMeierFitter
from utils.helper_functions import (
    bonferroni_correction,
    chi_squared_test,
    f_one,
    point_bi_serial,
    perform_propensity_score_matching,
    calculate_propensity_scores,
    sensitivity_analysis,
    spearman_correlation,
    ttest,
    zero_fill,
    check_balance,
    visualize_smds,
    visualize_p_value_bonferroni_corrections,
    fdr_correction,
    visualize_fdr_correction,
    save_for_deep_learning,
    categorize_age_group,
    calculate_group_norms_and_stability,
    classify_patient,
    plot_trend_trajectories,
    plot_individual_trajectories,
    calculate_percentage_change,
    visualize_tumor_stability,
    consistency_check,
    kruskal_wallis_test,
    fisher_exact_test,
    logistic_regression_analysis,
    # calculate_vif,
)   


class TumorAnalysis:
    """
    A class to perform tumor analysis using clinical and volumetric data.
    """

    ##################################
    # DATA LOADING AND PREPROCESSING #
    ##################################

    def __init__(self, clinical_data_path, volumes_data_paths, cohort):
        """
        Initialize the TumorAnalysis class.

        Parameters:
            clinical_data_file (str): Path to the clinical data CSV file.
            volumes_data_file (str): Path to the tumor volumes data CSV file.
        """
        pd.options.display.float_format = '{:.3f}'.format
        self.merged_data = pd.DataFrame()
        self.clinical_data_reduced = pd.DataFrame()
        self.post_treatment_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.p_values = []
        self.coef_values = []
        self.progression_threshold = correlation_cfg.PROGRESSION_THRESHOLD
        self.stability_threshold = correlation_cfg.STABILITY_THRESHOLD
        self.high_risk_threshold = correlation_cfg.HIGH_RISK_THRESHOLD
        self.angle = correlation_cfg.ANGLE
        self.caliper = correlation_cfg.CALIPER
        self.sample_size_plots = correlation_cfg.SAMPLE_SIZE
        self.cohort = cohort
        print("Step 0: Initializing TumorAnalysis class...")

        self.validate_files(clinical_data_path, volumes_data_paths)
        patient_ids_volumes = self.load_volumes_data(volumes_data_paths)
        if self.cohort == "BCH":
            self.load_clinical_data_bch(clinical_data_path, patient_ids_volumes)
        else:
            self.load_clinical_data_cbtn(clinical_data_path, patient_ids_volumes)
        self.merge_data()
        self.aggregate_summary_statistics()

    def validate_files(self, clinical_data_path, volumes_data_paths):
        """
        Check if the specified clinical data and volume data files exist.

        Parameters:
        - clinical_data_path (str): Path to the clinical data file.
        - volumes_data_paths (list): List containing paths to volume data files.

        Raises:
        - FileNotFoundError: If any of the files specified do not exist.

        Prints a validation message if all files exist.
        """
        missing_files = [
            path for path in [clinical_data_path] + volumes_data_paths if not os.path.exists(path)
        ]
        if missing_files:
            raise FileNotFoundError(f"The following files could not be found: {missing_files}")
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

        self.clinical_data["Age at First Diagnosis"] = (
            self.clinical_data["Date First Diagnosis"] - self.clinical_data["Date of Birth"]
        ).dt.days / 365.25

        self.clinical_data["Date of last clinical follow-up"] = pd.to_datetime(
            self.clinical_data["Date of last clinical follow-up"], dayfirst=True
        )
        self.clinical_data["Age at Last Clinical Follow-Up"] = (
            self.clinical_data["Date of last clinical follow-up"]
            - self.clinical_data["Date of Birth"]
        ).dt.days / 365.25

        self.clinical_data["Date First Treatment"] = pd.to_datetime(
            self.clinical_data["First Treatment"], dayfirst=True
        )

        self.clinical_data["Age at First Treatment"] = (
            self.clinical_data["Date First Treatment"] - self.clinical_data["Date of Birth"]
        ).dt.days / 365.25

        self.clinical_data["Received Treatment"] = (
            self.clinical_data["Age at First Treatment"].notna().map({True: "Yes", False: "No"})
        )

        self.clinical_data["Treatment Type"] = self.extract_treatment_types_bch()

        self.clinical_data["Age Difference"] = (
            self.clinical_data["Age at Last Clinical Follow-Up"] - self.clinical_data["Age at First Diagnosis"]
        )

        self.clinical_data["Follow-Up"] = self.clinical_data["Follow-Up"] / 365.25
        print(f"\tFollow-Up: {self.clinical_data['Follow-Up'].max(), self.clinical_data['Follow-Up'].min(), self.clinical_data['Follow-Up'].median()}")
        print(f"\tAge Difference: {self.clinical_data['Age Difference'].max(), self.clinical_data['Age Difference'].min(), self.clinical_data['Age Difference'].median()}")
        
        self.clinical_data["Follow-Up Time"] = np.where(
            self.clinical_data["Follow-Up"].notna(), self.clinical_data["Follow-Up"], self.clinical_data["Age Difference"]
        )
        self.clinical_data["Time to Treatment"] = np.where(
            self.clinical_data["Age at First Treatment"].notna(),
            (
                self.clinical_data["Age at First Treatment"]
                - self.clinical_data["Age at First Diagnosis"]
            ),
            self.clinical_data["Age Difference"],
        )

        # Apply the type conversions according to the dictionary
        for column, dtype in correlation_cfg.BCH_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        all_relevant_columns = (
            list(correlation_cfg.BCH_DTYPE_MAPPING.keys()) + correlation_cfg.BCH_DATETIME_COLUMNS
        )
        self.clinical_data_reduced = self.clinical_data[all_relevant_columns].copy()
        self.clinical_data_reduced["BCH MRN"] = (
            self.clinical_data_reduced["BCH MRN"].astype(str).str.zfill(7)
        )
        self.clinical_data_reduced = self.clinical_data_reduced[
            self.clinical_data_reduced["BCH MRN"].isin(patient_ids_volumes)
        ]
        print(f"\tFiltered clinical data has length {len(self.clinical_data_reduced)}.")

        print("\tParsed clinical data.")

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
        self.clinical_data["CBTN Subject ID"] = self.clinical_data["CBTN Subject ID"].astype(str)

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
            correlation_cfg.CBTN_GLIOMA_TYPES, self.clinical_data["Diagnoses"], map_type="histology"
        )

        # Age Last Clinical Follow Up
        self.clinical_data["Age at Last Clinical Follow-Up"] = self.clinical_data[
            "Age at Last Known Clinical Status"
        ]

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
            list(correlation_cfg.CBTN_DTYPE_MAPPING.keys()) + correlation_cfg.CBTN_DATETIME_COLUMNS
        )

        self.clinical_data_reduced = self.clinical_data[all_relevant_columns].copy()

        self.clinical_data_reduced = self.clinical_data_reduced[
            self.clinical_data_reduced["CBTN Subject ID"].isin(patient_ids_volumes)
        ]
        print(f"\tFiltered CBTN clinical data has length {len(self.clinical_data_reduced)}.")

        print("\tParsed CBTN clinical data.")

    def load_volumes_data(self, volumes_data_paths):
        """
        Load volumes data from specified paths. Each path contains CSV files for different patients.
        The data from each file is loaded, processed, and concatenated into a single DataFrame.

        Parameters:
        - volumes_data_paths (list): List containing paths to directories of volume data CSV files.

        The function updates the `self.volumes_data` attribute with the concatenated DataFrame.
        """

        data_frames = []
        for volumes_data_path in volumes_data_paths:
            all_files = [f for f in os.listdir(volumes_data_path) if f.endswith(".csv")]
            print(f"\tVolume data found: {len(all_files)}.")

            for file in all_files:
                patient_id = file.split("_")[0]

                patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
                # Adjust patient id
                patient_df["Patient_ID"] = patient_id
                if self.cohort == "BCH":
                    patient_df["Patient_ID"] = (
                        patient_df["Patient_ID"].astype(str).str.zfill(7).astype("string")
                    )

                data_frames.append(patient_df)

        self.volumes_data = pd.concat(data_frames, ignore_index=True)
        if self.cohort == "BCH":
            self.volumes_data["Date"] = pd.to_datetime(self.volumes_data["Date"], format="%d/%m/%Y")

        # CBTN Follow up time
        if self.cohort == "CBTN":
            follow_up_times = {}
            for patient_id in self.volumes_data["Patient_ID"].unique():
                patient_data = self.volumes_data[self.volumes_data["Patient_ID"] == patient_id]
                min_date = patient_data["Age"].min()
                max_date = patient_data["Age"].max()
                follow_up = max_date - min_date
                follow_up_times[patient_id] = follow_up
            self.volumes_data["Follow-Up Time"] = self.volumes_data["Patient_ID"].map(
                follow_up_times
            )

        # Patient IDs in volumes data
        patient_ids_volumes = set(self.volumes_data["Patient_ID"].unique())

        self.volumes_data = self.volumes_data.rename(columns={"Volume Growth[%]": "Volume Change",
                                                              "Volume Growth[%] Rate": "Volume Change Rate",
                                                              "Volume Growth[%] Avg": "Volume Change Avg",
                                                              "Volume Growth[%] Std": "Volume Change Std"})

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
        # TODO: Adjust this to the other logic
        for _, row in self.clinical_data.iterrows():
            treatments = []

            if row["Surgical Resection"] == "Yes":
                treatments.append("Surgery")

            if row["Systemic therapy before radiation"] == "Yes":
                treatments.append("Chemotherapy")

            if row["Radiation as part of initial treatment"] == "Yes":
                treatments.append("Radiation")

            if len(treatments) == 1 and treatments[0] == "Radiation":
                treatment_list.append("Surgery and Radiation")
            elif len(treatments) == 0:
                treatment_list.append("No Treatment")
            elif len(treatments) == 1:
                treatment_list.append(f"{treatments[0]} Only")
            elif len(treatments) == 2:
                treatment_list.append(f"{treatments[0]} and {treatments[1]}")
            elif len(treatments) == 3:
                treatment_list.append("All Treatments")
        
        return treatment_list

    def extract_treatment_info_cbtn(self):
        """
        Extracts treatment information for CBTN data.

        Updates self.clinical_data with the columns:
        [Age at First Diagnosis, Treatment Type,
        Received Treatment, Time to Treatment, Age at First Treatment]
        """

        # Extract the first age recorded in volumes data for each patient
        first_age_volumes = self.volumes_data.groupby("Patient_ID")["Age"].min()

        # Initialize new columns in clinical_data
        self.clinical_data["Age at First Diagnosis"] = None
        self.clinical_data["Treatment Type"] = None
        self.clinical_data["Received Treatment"] = None
        self.clinical_data["Time to Treatment"] = None
        self.clinical_data["Age at First Treatment"] = None
        # FIXME: clinal progression removed
        #self.clinical_data["Tumor Progression"] = None
        #self.clinical_data["Age at First Progression"] = None
        #self.clinical_data["Time to Progression"] = None

        # Loop through each patient in clinical_data
        for idx, row in self.clinical_data.iterrows():
            patient_id = str(row["CBTN Subject ID"])
            age_at_event = int(row["Age at Event Days"])
            surgery = row["Surgery"]
            chemotherapy = row["Chemotherapy"]
            radiation = row["Radiation"]
            efsur = row["Event Free Survival"]
            osur = row["Overall Survival"]
            efsur = int(efsur) if str(efsur).isnumeric() else float("inf")
            osur = int(osur) if str(osur).isnumeric() else float("inf")

            # FIXME: Progression
            # progression = "Yes" if efsur < osur else "No"
            # self.clinical_data.at[idx, "Tumor Progression"] = progression

            # Age at First Diagnosis
            if surgery == "Yes":
                age_at_first_diagnosis = int(first_age_volumes.get(patient_id, age_at_event))
            else:
                age_at_first_diagnosis = age_at_event

            self.clinical_data.at[idx, "Age at First Diagnosis"] = age_at_first_diagnosis

            # Treatment Type
            treatments = []
            if surgery == "Yes":
                treatments.append("Surgery")
            if chemotherapy == "Yes":
                treatments.append("Chemotherapy")
            if radiation == "Yes":
                treatments.append("Radiation")
            treatment_type = ", ".join(treatments) if treatments else "No Treatment"

            # TODO: Fix this manual correction
            if treatment_type != "Surgery, Chemotherapy, Radiation":
                self.clinical_data.at[idx, "Treatment Type"] = treatment_type

            # Received Treatment
            received_treatment = "Yes" if treatments else "No"
            self.clinical_data.at[idx, "Received Treatment"] = received_treatment

            # Time to Treatment and Age at First Treatment
            if received_treatment == "Yes":
                age_at_radiation_start = row["Age at Radiation Start"]
                age_at_chemotherapy_start = row["Age at Chemotherapy Start"]
                age_at_radiation_start = (
                    int(age_at_radiation_start)
                    if str(age_at_radiation_start).isnumeric()
                    else float("inf")
                )
                age_at_chemotherapy_start = (
                    int(age_at_chemotherapy_start)
                    if str(age_at_chemotherapy_start).isnumeric()
                    else float("inf")
                )
                age_at_first_treatment = min(
                    age_at_chemotherapy_start, age_at_radiation_start, age_at_event
                )

                self.clinical_data.at[idx, "Age at First Treatment"] = age_at_first_treatment
                self.clinical_data.at[idx, "Time to Treatment"] = (
                    age_at_first_treatment - age_at_first_diagnosis
                )

            # FIXME: Calculate Age at First Progression and Time to Progression
            # if pd.notna(efsur) and pd.notna(osur):
            #     if efsur < osur:
            #         self.clinical_data.at[idx, "Tumor Progression"] = "Yes"
            #         if pd.notna(row["Age at First Treatment"]):
            #             age_at_first_progression = row["Age at First Treatment"] + efsur
            #         else:
            #             # Use Age at First Diagnosis + EFS if Age at First Treatment is not available
            #             age_at_first_progression = age_at_first_diagnosis + efsur
            #         self.clinical_data.at[
            #             idx, "Age at First Progression"
            #         ] = age_at_first_progression
            #         self.clinical_data.at[idx, "Time to Progression"] = (
            #             age_at_first_progression - age_at_first_diagnosis
            #         )
            #     else:
            #         self.clinical_data.at[idx, "Progression"] = "No"

        self.clinical_data["Age at First Diagnosis"] = pd.to_numeric(
            self.clinical_data["Age at First Diagnosis"], errors="coerce"
        )
        self.clinical_data["Age at Last Clinical Follow-Up"] = pd.to_numeric(
            self.clinical_data["Age at Last Clinical Follow-Up"], errors="coerce"
        )

        # Fill missing values
        self.clinical_data.fillna(
            {
                "Time to Treatment": 0,
                "Age at First Treatment": age_at_first_diagnosis,
                #FIXME 
                #"Age at First Progression": 0,
                #"Time to Progression": 0,
            },
            inplace=True,
        )

    def merge_data(self):
        """
        Merge reduced clinical data with the volumes data based on patient ID,
        excluding redundant columns.

        This function updates the `self.merged_data` attribute with the merged DataFrame.
        """

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
        self.merged_data["Age Group"] = self.merged_data.apply(categorize_age_group, axis=1).astype(
            "category"
        )
        # Reset index after merging
        self.merged_data.reset_index(drop=True, inplace=True)
        print("\tMerged clinical and volume data.")

    def aggregate_summary_statistics(self):
        """
        Calculate summary statistics for specified columns in the merged data.

        This function updates the `self.merged_data` with new columns for mean,
        median, and standard deviation for each of the specified columns.
        """

        def cumulative_stats(group, variable):
            group[f"{variable} CumMean"] = group[variable].expanding().mean()
            group[f"{variable} CumMedian"] = group[variable].expanding().median()
            group[f"{variable} CumStd"] = group[variable].expanding().std().fillna(0)
            return group

        def rolling_stats(group, variable, window_size=3, min_periods=1):
            group[f"{variable} RollMean"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).mean()
            )
            group[f"{variable} RollMedian"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).median()
            )
            group[f"{variable} RollStd"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).std().fillna(0)
            )
            return group

        for var in [
            "Volume",
            "Normalized Volume",
            "Volume Change",
        ]:
            grouped = self.merged_data.groupby("Patient_ID", as_index=False)
            self.merged_data = self.merged_data.groupby("Patient_ID", as_index=False).apply(
                cumulative_stats, var
            )
            self.merged_data.reset_index(drop=True, inplace=True)
            self.merged_data = self.merged_data.groupby("Patient_ID", as_index=False).apply(
                rolling_stats, var
            )
            self.merged_data.reset_index(drop=True, inplace=True)

        print("\tAdded rolling and accumulative summary statistics.")

    ###################################
    # DATA ANALYSIS AND VISUALIZATION #
    ###################################
    def analyze_pre_treatment(self, prefix, output_dir):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, etc.
        """
        print("\tPre-treatment Correlations:")

        # variable types
        categorical_vars = [
            "Location",
            "Symptoms",
            "Histology",
            "Treatment Type",
            "Age Group",
            "BRAF Status",
            "Sex",
            "Tumor Classification",
            "Received Treatment",

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
            "Follow-Up Time"
        ]
        outcome_var = "Patient Classification Binary"
                
        # for full blown out comparison uncomment the following lines
        # for num_var in numerical_vars:
        #     for cat_var in categorical_vars:
        #         if self.merged_data[cat_var].nunique() == 2:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 output_dir,
        #                 method="t-test",
        #             )
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 output_dir,
        #                 method="point-biserial",
        #             )
        #         else:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 output_dir,
        #                 method="ANOVA",
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
        #                 output_dir,
        #                 method="spearman",
        #             )
        #             self.analyze_correlation(
        #                 num_var,
        #                 other_num_var,
        #                 self.merged_data,
        #                 prefix,
        #                 output_dir,
        #                 method="pearson",
        #             )
        # aggregated_data = (
        #     self.merged_data.sort_values("Date").groupby("Patient_ID", as_index=False).last()
        # )
        # for cat_var in categorical_vars:
        #     for other_cat_var in categorical_vars:
        #         if cat_var != other_cat_var:
        #             self.analyze_correlation(
        #                 cat_var,
        #                 other_cat_var,
        #                 aggregated_data,
        #                 prefix,
        #                 output_dir,
        #             )
        
        # Univariate analysis
        patient_constant_vars = [
            "Location", "Symptoms", "Histology", "BRAF Status",
            "Sex", "Received Treatment", "Age at First Diagnosis",
            "Age at Last Clinical Follow-Up", "Baseline Volume", "Treatment Type"
        ]
        time_varying_vars = [
            "Age", "Age Group", "Days Between Scans", "Volume", "Normalized Volume",
            "Volume Change", "Volume Change Rate", "Follow-Up Time"
        ]
        all_vars = patient_constant_vars + time_varying_vars
        pooled_results = pd.DataFrame(columns=['Main Category', 'Subcategory', 'OR', 'Lower', 'Upper', 'p'])
        for variable in all_vars:
            print(f"\t\tAnalyzing {variable}...")
            pooled_results = self.univariate_analysis(variable, outcome_var, output_dir, pooled_results)
        self.plot_forest_plot(pooled_results, output_dir)

    def analyze_correlation(self, x_val, y_val, data, prefix, output_dir, method="spearman"):
        """
        Perform and print the results of a statistical test to analyze the correlation
        between two variables.

        Parameters:
        - x_val (str): The name of the first variable.
        - y_val (str): The name of the second variable.
        - data (DataFrame): The data containing the variables.
        - prefix (str): The prefix to be used for naming visualizations.
        - method (str): The statistical method to be used (default is "spearman").

        Updates the class attributes with the results of the test and prints the outcome.
        """
        test_result, coef, p_val = None, None, None  # Initialize test_result
        test_type = ""
        x_dtype = data[x_val].dtype
        y_dtype = data[y_val].dtype

        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            coef, p_val = spearman_correlation(data[x_val], data[y_val])
            test_result = (coef, p_val)
            test_type = f"correlation {method}"
        elif pd.api.types.is_categorical_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            categories = data[x_val].nunique()
            if categories == 2:
                if method == "t-test":
                    t_stat, p_val = ttest(data, x_val, y_val)
                    test_result = (t_stat, p_val)
                    test_type = "t-test"
                if method == "point-biserial":
                    coef, p_val = point_bi_serial(data, x_val, y_val)
                    test_result = (coef, p_val)
                    test_type = "point-biserial"
            else:
                groups = [group[y_val].dropna() for name, group in data.groupby(x_val)]
                normality_tests = [shapiro(group)[1] for group in groups]
                if all(p > 0.05 for p in normality_tests):  # If all groups pass normality test
                    f_stat, p_val = f_one(data, x_val, y_val)
                    test_result = (f_stat, p_val)
                    test_type = "ANOVA"
                else:
                    test_stat, p_val = kruskal_wallis_test(data, x_val, y_val)
                    test_result = (test_stat, p_val)
                    test_type = "Kruskal-Wallis"
        
        elif pd.api.types.is_categorical_dtype(x_dtype) and pd.api.types.is_categorical_dtype(
            y_dtype
        ):
            if data[x_val].nunique() == 2 and data[y_val].nunique() == 2:
                odds_ratio, p_val = fisher_exact_test(data, x_val, y_val)
                test_result = (odds_ratio, p_val)
                test_type = "Fisher's Exact"
            else:
                chi2, p_val, _, _ = chi_squared_test(data, x_val, y_val)
                test_result = (chi2, p_val)
                test_type = "chi-squared"

        if test_result:
            print(
                f"\t\t{x_val} and {y_val} - {test_type.title()} Test: Statistic={test_result[0]},"
                f" P-value={test_result[1]}"
            )
            self.visualize_statistical_test(
                x_val, y_val, data, test_result, prefix, output_dir, test_type, method=method
            )

            self.p_values.append(p_val)
            if coef is not None:
                self.coef_values.append(coef)
        else:
            print(
                f"\t\tCould not perform analysis on {x_val} and {y_val} due to incompatible data"
                " types."
            )

    def univariate_analysis(self, variable, outcome_var, output_dir, pooled_results):
        """
        Perform univariate logistic regression analysis for a given variable.
        """
        data, y = self.prepare_data_for_univariate_analysis(variable, outcome_var)        
        
        if data is not None and not data.empty:
            X = data.drop(columns=[outcome_var], errors='ignore')
            try:
                result = logistic_regression_analysis(y, X)
                #print(result.summary2())
                self.visualize_univariate_analysis_alternative(result, variable, output_dir)
                pooled_results = self.pool_results(result, variable, pooled_results)
                print(f"\t\t\tModel fitted successfully with {variable}.")
            except ExceptionGroup as e:
                print(f"\t\tError fitting model with {variable}: {e}")
        else:
            print(f"\t\tNo data available for {variable}.")

        return pooled_results

    def prepare_data_for_univariate_analysis(self, variable, outcome_var):
        """
        Prepare the data for univariate logistic regression analysis.
        This function handles patient-constant and time-varying variables differently.
        """
        # Handling patient-constant variables
        if variable in ["Location", "Symptoms", "Histology", "Treatment Type", 
                        "BRAF Status", "Sex", 
                        "Received Treatment", "Age at First Diagnosis", 
                        "Age at Last Clinical Follow-Up", "Baseline Volume"]:
            data_agg = self.merged_data.groupby('Patient_ID').agg({variable: 'first', outcome_var: 'first'}).reset_index()
        else:
            # Handling time-varying variables
            data_agg = self.merged_data[[variable, outcome_var]].dropna()
        
        # For categorical variables, convert them to dummy variables
        if variable in ["Location", "Symptoms", "Histology", "Treatment Type", "BRAF Status", "Sex", "Received Treatment", "Age Group"]:
            if variable in data_agg.columns:
                data_agg[variable] = data_agg[variable].astype(str)
                data_agg = pd.get_dummies(data_agg, columns=[variable], drop_first=True)
                for col in data_agg.columns:
                    data_agg[col] = pd.to_numeric(data_agg[col], errors='coerce')
        
        # Ensure outcome_var is binary numeric
        data_agg[outcome_var] = pd.to_numeric(data_agg[outcome_var], errors='coerce').fillna(0).astype(int)
        relevant_columns = [col for col in data_agg.columns if col == outcome_var or col.startswith(variable+'_') or col == 'const']
        data_agg = data_agg[relevant_columns].dropna()
        if data_agg.isnull().values.any():
            print(f"\t\tWarning: Missing values detected for {variable}. Dropping missing values.")
            data_agg.fillna(0, inplace=True)
        data_agg = data_agg.drop(columns=['Patient_ID'], errors='ignore')
        
        if 'const' not in data_agg.columns:
            data_agg = sm.add_constant(data_agg)
        
        if outcome_var in data_agg:
            y = data_agg[outcome_var]
            X = data_agg.drop(columns=[outcome_var], errors='ignore')
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
                # if "Treatment Type" in col:
                #     print(f"{col}:")
                #     print(X[col].value_counts())
                #     print(X[col].min())
                #     print(X[col].max())

            # print(X.shape, y.shape)
            # print(X.dtypes)
            # print(y.dtypes)
            # for col in X.columns:
            #     if not np.issubdtype(X[col].dtype, np.number):
            #         print(f"Non-numeric data found in column: {col}")
            # print(X.isnull().all())  # This checks if any column has all NaN values
            
            #calculate_vif(X)
            return X, y
        else:
            return None, None
        
    def visualize_univariate_analysis(self, model_result, variable_base_name, output_dir):
        """
        Visualize the results of univariate logistic regression analysis,
        creating plots for both categorical and numerical variables that
        display coefficients or odds ratios along with their confidence intervals.
        
        Args:
        - result: The result object from logistic regression analysis.
        - variable: The name of the variable being analyzed.
        - output_dir: The directory where the plot images will be saved.
        """
        
        coeffs = model_result.params.drop('const', errors='ignore')
        conf = model_result.conf_int().drop('const', errors='ignore')
        p_values = model_result.pvalues.drop('const', errors='ignore')

        if variable_base_name in ["Location", "Symptoms", "Histology", "Treatment Type", 
                              "BRAF Status", "Sex", "Received Treatment", "Age Group"]:
            values_to_plot = np.exp(coeffs)
            lower_error = values_to_plot - np.exp(conf[0])
            upper_error = np.exp(conf[1]) - values_to_plot
        else:
            values_to_plot = coeffs
            lower_error = coeffs - conf[0]
            upper_error = conf[1] - coeffs

        error_bars = np.vstack([lower_error, upper_error])
        
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        categories = values_to_plot.index
        ax.bar(categories, values_to_plot, yerr=error_bars, color='skyblue', align='center', alpha=0.7, capsize=4)
        ax.set_ylabel('Odds Ratio' if variable_base_name in ["Location", "Symptoms", "Histology", "Treatment Type", 
                                                            "BRAF Status", "Sex", "Received Treatment", "Age Group"] 
                    else 'Log Odds')

        ax.set_title(f'{variable_base_name} Effect Size with 95% CI')
        ax.axhline(1 if variable_base_name in ["Location", "Symptoms", "Histology", "Treatment Type", 
                                            "BRAF Status", "Sex", "Received Treatment", "Age Group"] else 0, 
                color='red', linestyle='--')
        
        # Annotate p-values
        for i, p_value in enumerate(p_values):
            ax.text(i, ax.get_ylim()[1], f'p={p_value:.2e}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.xlabel('Categories' if variable_base_name in ["Location", "Symptoms", "Histology", "Treatment Type", 
                                                        "BRAF Status", "Sex", "Received Treatment", "Age Group"] else 'Variable')
        plt.tight_layout()
        
        file_name = f"{variable_base_name}_univariate_analysis.png"
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()

    def visualize_univariate_analysis_alternative(self, model_result, variable_base_name, output_dir):
            """
            Visualize the results of univariate logistic regression analysis,
            creating plots for both categorical and numerical variables that
            display coefficients or odds ratios along with their confidence intervals.
            
            Args:
            - result: The result object from logistic regression analysis.
            - variable: The name of the variable being analyzed.
            - output_dir: The directory where the plot images will be saved.
            """
            coeffs = model_result.params
            conf = model_result.conf_int()
            odds_ratios = np.exp(coeffs)  # Convert coefficients to odds ratios for interpretation
            
            # Calculate error bars from confidence intervals
            error_bars = [odds_ratios - np.exp(conf[0]), np.exp(conf[1]) - odds_ratios]


            if variable_base_name in ["Location", "Symptoms", "Histology", "Treatment Type", 
                            "BRAF Status", "Sex", "Received Treatment", "Age Group"]:
                # Categorical variable: Plot as Odds Ratios
                plt.figure(figsize=(8, 4))
                plt.errorbar(x=odds_ratios.index, y=odds_ratios, yerr=error_bars, fmt='o', color='black',
                            ecolor='lightgray', elinewidth=3, capsize=0)
                plt.xticks(rotation=45)
                plt.title(f'Odds Ratios with 95% CI for {variable_base_name}')
                plt.ylabel('Odds Ratio')
                plt.xlabel('Categories')
            else:
                # Numerical variable: Plot as Coefficients
                plt.figure(figsize=(8, 4))
                plt.errorbar(x=coeffs.index, y=coeffs, yerr=[conf[1] - coeffs, coeffs - conf[0]], fmt='o', color='black',
                            ecolor='lightgray', elinewidth=3, capsize=0)
                plt.title(f'Coefficients with 95% CI for {variable_base_name}')
                plt.ylabel('Log Odds')
                plt.xlabel('Variable')

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{variable_base_name}_analysis_plots_alternative.png")
            plt.close()

    def plot_forest_plot(self, pooled_results, output_dir):
        """
        Create a forest plot from the pooled results of univariate analyses.

        Args:
            pooled_results: DataFrame with columns 'Variable', 'HR', 'Lower', 'Upper', and 'p'.
            output_file: File path to save the forest plot image.
        """
        expected_columns = {'Main Category', 'Subcategory', 'OR', 'Lower', 'Upper', 'p'}
        if not expected_columns.issubset(pooled_results.columns):
            missing_cols = expected_columns - set(pooled_results.columns)
            raise ValueError(f"The DataFrame is missing the following required columns: {missing_cols}")


        # sort pooled results alphabetically, then clear out non-positive and infinite values
        pooled_results = pooled_results[(pooled_results['HR'] > 0) & (pooled_results['Lower'] > 0) & (pooled_results['Upper'] > 0)]
        pooled_results.replace([np.inf, -np.inf], np.nan, inplace=True)
        pooled_results.dropna(subset=['HR', 'Lower', 'Upper', 'p'], inplace=True)
        pooled_results.sort_values(by=['Main Category', 'Subcategory'], ascending=[True, True])
        pooled_results.reset_index(drop=True, inplace=True)

        max_hr = np.percentile(pooled_results['Upper'], 90)
        pooled_results = pooled_results[pooled_results['Upper'] <= max_hr]
        
        # General plot settings + x parameters
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.3, right=0.7)

        ax.set_xscale('log')
        ax.set_xlim(left=0.01, right=100)
        ax.set_xlabel('Odd Ratios')
        ax.axvline(x=1, linestyle='--', color='red', lw=1)

        # Categories handling and colors
        unique_main_categories = pooled_results['Main Category'].unique()
        colormap = plt.get_cmap('tab10')
        colors = [colormap(i) for i in range(len(unique_main_categories))]
        category_colors = {cat: color for cat, color in zip(unique_main_categories, colors)}

        # Plot elements and error bars
        for i, row in pooled_results.iterrows():
            main_category = row['Main Category']
            subcategory = row['Subcategory']
            is_reference = 'Reference' in subcategory  # Replace with appropriate check for your reference category
            
            #fmt = 's' if is_reference else 'o'
            ax.errorbar(row['HR'], i, xerr=[[row['HR'] - row['Lower']], [row['Upper'] - row['HR']]],
                        fmt='o', color=category_colors[main_category], ecolor='gray', elinewidth=1, capsize=3)
           

        # annotations on the right
        ax.margins(x=1)  # Add margin to the right side of the plot to make space for annotations
        fig.canvas.draw()  # Need to draw the canvas to update axes positions

        # Get the bounds of the axes in figure space
        ax_bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Calculate the figure and axes widths in inches
        fig_width_inches = fig.get_size_inches()[0]
        axes_width_inches = ax_bounds.width
        annotation_x_position = ax_bounds.x1 + 0.01 * fig_width_inches
        
        # Calculate the position for the annotations (a bit to the right of the axes)
        annotation_x_position = ax_bounds.x1 + 0.01 * fig_width_inches
        for i, row in pooled_results.iterrows():
            if i == pooled_results.index[0]:
                ax.text(annotation_x_position + (40 * axes_width_inches), 20, 'HR', ha='left', va='center', fontsize=10, fontweight='bold')
                ax.text(annotation_x_position + (100 * axes_width_inches) , 20, '95% CI', ha='left', va='center', fontsize=10, fontweight='bold')
                ax.text(annotation_x_position + (600 * axes_width_inches), 20, 'P-val', ha='left', va='center', fontsize=10, fontweight='bold')
            
            if 'Reference' not in row['Subcategory']:
                ax.text(annotation_x_position+ (40 * axes_width_inches), i, f"{row['HR']:.2f}", ha='left', va='center', fontsize=8)
                ax.text(annotation_x_position+ (100 * axes_width_inches) , i, f"({row['Lower']:.2f}-{row['Upper']:.2f})", ha='left', va='center', fontsize=8)
                ax.text(annotation_x_position+ (600 * axes_width_inches), i, f"{row['p']:.3f}", ha='left', va='center', fontsize=8)
            else:
                ax.text(annotation_x_position + 5, i, 'Reference', ha='left', va='center', fontsize=8)
        
        
        # Annotations on the left
        subcategory_counts = pooled_results['Subcategory'].value_counts().to_dict()
        y_tick_positions = []
        copy_df = self.merged_data.copy()
        unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])

        annotation_left_x_position = 0.01
        for i, row in pooled_results.iterrows():
            category = row['Main Category']
            subcategory = row['Subcategory']
            count = unique_pat[category].value_counts().get(subcategory, 0)
            ax.text(annotation_left_x_position, i, f"{category} - {subcategory} - {count}", ha='right', va='center', fontsize=8, color='gray', transform=ax.transData)
           
        

        #ax.set_yticks(y_tick_positions)
        ax.set_yticklabels([])
        ax.text(-0.35, 1.01, 'Variables and \n Subgroups', ha='right', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
        ax.text(-0.2, 1.01, 'Count (n)', ha='left', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
       
        # Add title, grid, and layout
        ax.set_title('Univariate Analysis Forest Plot')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout(rect=[0, 0, 1, 0])
        
        output_file = os.path.join(output_dir, 'forest_plot.png')
        plt.savefig(output_file)
        plt.close()

    def pool_results(self, result, var_name, pooled_results=None):
        """
        Pool the results of univariate analysis to create a forest plot.

        Args:
            result: Result object from univariate analysis.
            pooled_results: DataFrame to store pooled results.
        """
        if not isinstance(pooled_results, pd.DataFrame) or pooled_results is None:
            pooled_results = pd.DataFrame(columns=['Main Category', 'Subcategory', 'HR', 'Lower', 'Upper', 'p'])
        

        if result is not None:
            for variable in result.params.index[1:]:
                coef = result.params[variable]
                conf = result.conf_int().loc[variable].values
                p_val = result.pvalues[variable]

                var_split = variable.split('_')
                main_category = var_name
                subcategory = ' '.join(var_split[1:]) if len(var_split) > 1 else main_category

                # Append the results to the pooled DataFrame
                new_row = pd.DataFrame({
                    'Main Category': var_name.replace('_', ' '),
                    'Subcategory' : subcategory.replace('_', ' ').strip(),
                    'OR': np.exp(coef),
                    'Lower': np.exp(conf[0]),
                    'Upper': np.exp(conf[1]),
                    'p': p_val
                }, index=[0])

                pooled_results = pd.concat([pooled_results, new_row], ignore_index=True)
        return pooled_results

    def visualize_statistical_test(
        self,
        x_val,
        y_val,
        data,
        test_result,
        prefix,
        output_dir,
        test_type="correlation",
        method="spearman",
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
            method (str): The correlation method used ('pearson' or 'spearman'), if applicable.
        """
        stat, p_val = test_result
        title = f"{x_val} vs {y_val} ({test_type.capitalize()}) \n"
        num_patients = data["Patient_ID"].nunique()

        units = {
            "Age": "days",
            "Age Group": "days",
            "Date": "date",
            "Time to Treatment": "days",
            "Volume": "mm³",
            "Volume RollMean": "mm³",
            "Volume RollMedian": "mm³",
            "Volume RollStd": "mm³",
            "Volume CumMean": "mm³",
            "Volume CumMedian": "mm³",
            "Volume CumStd": "mm³",
            "Volume Change": "%",
            "Volume Change RollMean": "%",
            "Volume Change RollMedian": "%",
            "Volume Change RollStd": "%",
            "Volume Change CumMean": "%",
            "Volume Change CumMedian": "%",
            "Volume Change CumStd": "%",
            "Baseline Volume": "mm³",
        }

        x_unit = units.get(x_val, "")
        y_unit = units.get(y_val, "")

        # Plot based on test type
        if test_type in ["t-test", "ANOVA", "Kruskal-Wallis"]:
            means = data.groupby(x_val)[y_val].mean()
            stds = data.groupby(x_val)[y_val].std()
            se = stds / np.sqrt(data.groupby(x_val)[y_val].count())  # Standard error
            plt.xticks(rotation=90, fontsize="small")
            # Bar plot with error bars showing the standard error
            _, ax = plt.subplots()
            means.plot(kind="bar", yerr=se, capsize=4, ax=ax, color="skyblue", ecolor="black")
            title += f"Statistic: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type == "point-biserial":
            sns.boxplot(x=x_val, y=y_val, data=data)
            title += f"Correlation Coefficient: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type == "Fisher's Exact":
            contingency_table = pd.crosstab(data[y_val], data[x_val])
            sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="g")
            title += f"Odds Ratio: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"        
        elif test_type == "correlation":
            sns.scatterplot(x=x_val, y=y_val, data=data)
            sns.regplot(x=x_val, y=y_val, data=data, scatter=False, color="blue")
            title += (
                f"{method.title()} correlation coefficient: {stat:.2f}, P-value:"
                f" {p_val:.3e} (N={num_patients})"
            )
        elif test_type == "chi-squared":
            contingency_table = pd.crosstab(data[y_val], data[x_val])
            sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="g")
            title += f"Chi2: {stat:.2f}, P-value: {p_val:.3e}, (N={num_patients})"

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
            output_dir, f"{prefix}_{x_val}_vs_{y_val}_{test_type}_{method}.png"
        )
        plt.savefig(save_file)
        plt.close()

        # If the test type is correlation, create a heatmap for the correlation matrix
        if test_type == "correlation":
            plt.figure(figsize=(10, 8))

            numeric_data = data.select_dtypes(include=[np.number])

            sns.heatmap(
                numeric_data.corr(method=method),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                square=True,
                linewidths=0.5,
            )
            plt.title(f"Heatmap of {method.capitalize()} Correlation")
            plt.tight_layout()
            heat_map_file = os.path.join(output_dir, f"{prefix}_{method}_correlation_heatmap.png")
            plt.savefig(heat_map_file)
            plt.close()

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
        pre_treatment_data["Time since First Scan"] = pre_treatment_data.groupby("Patient_ID")[
            "Age"
        ].transform(lambda x: (x - x.iloc[0]))
        pre_treatment_data.reset_index(drop=True, inplace=True)
        
        self.merged_data.sort_values(by=["Patient_ID", "Age"], inplace=True)
        self.merged_data["Time since First Scan"] = pre_treatment_data["Time since First Scan"]
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

            sample_ids = pre_treatment_data["Patient_ID"].drop_duplicates().sample(n=sample_size)
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
            unit="%",
        )
        volume_change_rate_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_volume_change_rate_trajectories_plot.png"
        )
        plot_individual_trajectories(
            volume_change_rate_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Volume Change Rate",
            unit="% / day",
        )
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
                self.stability_threshold,
                self.high_risk_threshold,
                angle=self.angle,
            )
            for patient_id in patients_ids
        }
        data["Classification"] = data["Patient_ID"].map(patient_classifications)
        data['Patient Classification Binary'] = pd.to_numeric(data['Classification'].apply(lambda x: 1 if x == 'Progressor' else 0), errors='coerce').fillna(0).astype(int)
        # Save to original dataframe
        classifications_series = pd.Series(patient_classifications)
        self.merged_data["Patient Classification"] = (
            self.merged_data["Patient_ID"].map(classifications_series).astype("category")
        )
        self.merged_data["Patient Classification Binary"] = data['Patient Classification Binary']

        # Plots
        output_filename = os.path.join(output_dir, f"{prefix}_trend_analysis.png")
        plot_trend_trajectories(data, output_filename, column_name, unit="mm^3")
        print("\t\tSaved trend analysis plot.")

    def analyze_tumor_stability(
        self, data, output_dir, volume_weight=0.5, growth_weight=0.5, change_threshold=20
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
        volume_column = "Normalized Volume"
        volume_change_column = "Volume Change"
        data = calculate_group_norms_and_stability(data, volume_column, volume_change_column)
        # Calculate the overall volume change for each patient
        data["Overall Volume Change"] = data["Patient_ID"].apply(
            lambda x: calculate_percentage_change(data, x, volume_column)
        )
        # Calculate the Stability Index using weighted scores
        data["Stability Index"] = (
            volume_weight * data["Volume Stability Score"]
            + growth_weight * data["Growth Stability Score"]
        )

        # Normalize the Stability Index to have a mean of 1
        data["Stability Index"] /= np.mean(data["Stability Index"])

        significant_volume_change = abs(data["Overall Volume Change"]) >= change_threshold
        stable_subset = data.loc[~significant_volume_change, "Stability Index"]
        mean_stability_index = stable_subset.mean()
        std_stability_index = stable_subset.std()
        num_std_dev = 2
        stability_threshold = mean_stability_index + (num_std_dev * std_stability_index)

        data["Tumor Classification"] = data.apply(
            lambda row: "Unstable"
            if abs(row["Overall Volume Change"]) >= change_threshold
            or row["Stability Index"] > stability_threshold
            else "Stable",
            axis=1,
        ).astype("category")

        # Map the 'Stability Index' and 'Tumor Classification' to the
        # self.merged_data using the maps
        m_data = pd.merge(
            self.merged_data,
            data[["Patient_ID", "Age", "Stability Index", "Tumor Classification", "Overall Volume Change"]],
            on=["Patient_ID", "Age"],
            how="left",
        )

        self.merged_data = m_data
        self.merged_data.reset_index(drop=True, inplace=True)
        visualize_tumor_stability(data, output_dir, stability_threshold, change_threshold)
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

            # Sex, Received Treatment, Progression, Symptoms, Location,
            # Patient Classification, Treatment Type
            copy_df = self.merged_data.copy()
            unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
            counts_sex = unique_pat["Sex"].value_counts()
            counts_received_treatment = unique_pat["Received Treatment"].value_counts()
            counts_symptoms = unique_pat["Symptoms"].value_counts()
            counts_histology = unique_pat["Histology"].value_counts()
            counts_location = unique_pat["Location"].value_counts()
            counts_patient_classification = unique_pat["Patient Classification"].value_counts()
            counts_treatment_type = unique_pat["Treatment Type"].value_counts()

            write_stat(f"\t\tReceived Treatment: {counts_received_treatment}")
            write_stat(f"\t\tSymptoms: {counts_symptoms}")
            write_stat(f"\t\tHistology: {counts_histology}")
            write_stat(f"\t\tLocation: {counts_location}")
            write_stat(f"\t\tSex: {counts_sex}")
            write_stat(f"\t\tPatient Classification: {counts_patient_classification}")
            write_stat(f"\t\tTreatment Type: {counts_treatment_type}")

            # Volume Change
            filtered_data = self.merged_data[self.merged_data["Volume Change"] != 0]
            median_volume_change = filtered_data["Volume Change"].median()
            max_volume_change = filtered_data["Volume Change"].max()
            min_volume_change = filtered_data["Volume Change"].min()
            write_stat(f"\t\tMedian Volume Change: {median_volume_change} %")
            write_stat(f"\t\tMaximum Volume Change: {max_volume_change} %")
            write_stat(f"\t\tMinimum Volume Change: {min_volume_change} %")
            
            # Volume Change Rate
            filtered_data = self.merged_data[self.merged_data["Volume Change Rate"] != 0]
            median_volume_change_rate = filtered_data["Volume Change Rate"].median()
            max_volume_change_rate = filtered_data["Volume Change Rate"].max()
            min_volume_change_rate = filtered_data["Volume Change Rate"].min()
            write_stat(f"\t\tMedian Volume Change Rate: {median_volume_change_rate} %/day")
            write_stat(f"\t\tMaximum Volume Change Rate: {max_volume_change_rate} %/day")
            write_stat(f"\t\tMinimum Volume Change Rate: {min_volume_change_rate} %/day")
            
            

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
            write_stat(f"\t\tMedian Baseline Volume: {median_baseline_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMaximum Baseline Volume: {max_baseline_volume / mm3_to_cm3} cm^3")
            write_stat(f"\t\tMinimum Baseline Volume: {min_baseline_volume / mm3_to_cm3} cm^3")

            # Follow-Up time
            median_follow_up = self.merged_data["Follow-Up Time"].median()
            max_follow_up = self.merged_data["Follow-Up Time"].max()
            min_follow_up = self.merged_data["Follow-Up Time"].min()
            median_follow_up_months = median_follow_up
            max_follow_up_months = max_follow_up
            min_follow_up_months = min_follow_up 
            write_stat(f"\t\tMedian Follow-Up Time: {median_follow_up_months:.2f} years")
            write_stat(f"\t\tMaximum Follow-Up Time: {max_follow_up_months:.2f} years")
            write_stat(f"\t\tMinimum Follow-Up Time: {min_follow_up_months:.2f} years")
            
            
            # get the ids of the patients with the three highest normalized volumes that do not repeat
            top_normalized_volumes = self.merged_data.nlargest(3, "Normalized Volume")
            top_volumes = self.merged_data.nlargest(3, "Volume")
            self.merged_data['Absolute Volume Change'] = self.merged_data['Volume Change'].abs()
            self.merged_data['Absolute Volume Change Rate'] = self.merged_data['Volume Change Rate'].abs()
            top_volume_changes = self.merged_data.nlargest(3, 'Absolute Volume Change')
            top_volume_change_rates = self.merged_data.nlargest(3, 'Absolute Volume Change Rate')
            patient_ids_highest_norm_volumes = top_normalized_volumes["Patient_ID"].tolist()
            patient_ids_highest_volumes = top_volumes["Patient_ID"].tolist()
            patient_ids_highest_volume_changes = top_volume_changes["Patient_ID"].tolist()
            patient_ids_highest_volume_change_rates = top_volume_change_rates["Patient_ID"].tolist()
            write_stat(f"\t\tPatients with highest normalized volumes: {patient_ids_highest_norm_volumes}")
            write_stat(f"\t\tPatients with highest volumes: {patient_ids_highest_volumes}")
            write_stat(f"\t\tPatients with highest volume changes: {patient_ids_highest_volume_changes}")
            write_stat(f"\t\tPatients with highest volume change rates: {patient_ids_highest_volume_change_rates}")
            
        print(f"\t\tSaved summary statistics to {file_path}.")
    
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
        print(f"\tAnalyzing time to event for {stratify_by}:")
        # pre_treatment_data as df copy for KM curves
        # only patients who showed tumor progression before the first treatment
        analysis_data_pre = self.merged_data.copy()
        analysis_data_pre = analysis_data_pre[
            analysis_data_pre["Age at First Diagnosis"]
            < analysis_data_pre["Age at First Treatment"]
        ]
        analysis_data_pre.loc[:, "Duration"] = (
            analysis_data_pre["Age at First Progression"]
            - analysis_data_pre["Age at First Diagnosis"]
        )

        analysis_data_pre["Event_Occurred"] = ~analysis_data_pre["Age at First Progression"].isna()
        analysis_data_pre = analysis_data_pre.dropna(subset=["Duration", "Event_Occurred"])
        kmf = KaplanMeierFitter()

        if stratify_by and stratify_by in analysis_data_pre.columns:
            for category in analysis_data_pre[stratify_by].unique():
                category_data = analysis_data_pre[analysis_data_pre[stratify_by] == category]
                kmf.fit(
                    category_data["Duration"],
                    event_observed=category_data["Event_Occurred"],
                    label=str(category),
                )
                ax = kmf.plot_survival_function()
            plt.title(f"Stratified Survival Function by {stratify_by}")

        else:
            kmf.fit(
                analysis_data_pre["Duration"], event_observed=analysis_data_pre["Event_Occurred"]
            )
            ax = kmf.plot_survival_function()
            ax.set_title("Survival function of Tumor Progression")

        ax.set_xlabel("Days since Diagnosis")
        ax.set_ylabel("Survival Probability")

        survival_plot = os.path.join(
            output_dir, f"{prefix}_survival_plot_category_{stratify_by}.png"
        )
        plt.savefig(survival_plot, dpi=300)
        plt.close()
        print(f"\t\tSaved survival KaplanMeier curve for {stratify_by}.")

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
            post_treatment_vars = [
                "Volume",
                "Volume Change",
            ]

            for pre_var in pre_treatment_vars:
                print(f"\tPerforming sensitivity analysis on pre-treatment variables {pre_var}...")
                self.merged_data = sensitivity_analysis(
                    self.merged_data,
                    pre_var,
                    z_threshold=correlation_cfg.SENSITIVITY_THRESHOLD,
                )

            for post_var in post_treatment_vars:
                print(
                    f"\tPerforming sensitivity analysis on post-treatment variables {post_var}..."
                )
                self.post_treatment_data = sensitivity_analysis(
                    self.post_treatment_data,
                    post_var,
                    z_threshold=correlation_cfg.SENSITIVITY_THRESHOLD,
                )

            step_idx += 1

        if correlation_cfg.PROPENSITY:
            print(f"Step {step_idx}: Performing Propensity Score Matching...")
            print(self.merged_data.dtypes)
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
                smd_results = check_balance(matched_data, covariate_columns, treatment_column)

                visualize_smds(smd_results, path=output_stats)

            step_idx += 1

        if correlation_cfg.ANALYSIS_PRE_TREATMENT:
            prefix = f"{self.cohort}_pre_treatment"
            print(f"Step {step_idx}: Starting main analyses {prefix}...")
            
            if self.merged_data.isnull().values.any():
                self.merged_data.replace(np.nan, np.inf, inplace=True)
            # Survival analysis
            # stratify_by_list = [
            #     "Location",
            #     "Sex",
            #     "BRAF Status",
            #     "Age Group",
            #     "Symptoms",
            #     "Histology",
            # ]
            # for element in stratify_by_list:
            #     self.time_to_event_analysis(prefix, output_dir=output_stats, stratify_by=element)
            
            # Trajectories & Trend analysis
            self.trajectories(prefix, output_dir=output_stats)

            # Tumor stability
            self.analyze_tumor_stability(
                data=self.merged_data,
                output_dir=output_stats,
                volume_weight=correlation_cfg.VOLUME_WEIGHT,
                growth_weight=correlation_cfg.GROWTH_WEIGHT,
                change_threshold=correlation_cfg.CHANGE_THRESHOLD,
            )

            if self.merged_data.isnull().values.any():
                print(self.merged_data.isnull().sum())
                #self.merged_data.replace(np.nan, np.inf, inplace=True)
                
            # Descriptive statistics for table1 in paper
            #print(self.merged_data.dtypes)
            self.printout_stats(prefix=prefix, output_file_path=output_stats)

            # Correlations between variables
            self.analyze_pre_treatment(
                prefix=prefix,
                output_dir=output_correlations,
            )
            # self.perform_logistic_regression()

            # Last consistency check
            consistency_check(self.merged_data)

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
            corrected_p_values_fdr, is_rejected = fdr_correction(self.p_values, alpha=alpha)
            visualize_fdr_correction(
                self.p_values, corrected_p_values_fdr, is_rejected, alpha, output_stats
            )
            print("\tFalse Discovery Rate Correction done. ")

            step_idx += 1

        if correlation_cfg.FEATURE_ENG:
            print(f"Step {step_idx}: Starting Feature Engineering...")
            save_for_deep_learning(self.merged_data, output_stats, prefix="pre-treatment")
            # save_for_deep_learning(self.post_treatment_data, output_stats, prefix="post-treatment")
            step_idx += 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    analysis = TumorAnalysis(
        correlation_cfg.CLINICAL_CSV,
        [correlation_cfg.VOLUMES_CSV],
        cohort=correlation_cfg.COHORT,
    )

    os.makedirs(correlation_cfg.OUTPUT_DIR_CORRELATIONS, exist_ok=True)
    os.makedirs(correlation_cfg.OUTPUT_DIR_STATS, exist_ok=True)

    analysis.run_analysis(correlation_cfg.OUTPUT_DIR_CORRELATIONS, correlation_cfg.OUTPUT_DIR_STATS)
