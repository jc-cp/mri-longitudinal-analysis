"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations and treatments.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from cfg import correlation_cfg
from lifelines import KaplanMeierFitter
from scipy.signal import find_peaks
from utils.helper_functions import (
    bonferroni_correction,
    chi_squared_test,
    f_one,
    pearson_correlation,
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
    process_race_ethnicity,
    categorize_age_group,
    calculate_group_norms_and_stability,
)


class TumorAnalysis:
    """
    A class to perform tumor analysis using clinical and volumetric data.
    """

    def __init__(self, clinical_data_path, volumes_data_paths):
        """
        Initialize the TumorAnalysis class.

        Parameters:
            clinical_data_file (str): Path to the clinical data CSV file.
            volumes_data_file (str): Path to the tumor volumes data CSV file.
        """
        self.merged_data = pd.DataFrame()
        self.clinical_data_reduced = pd.DataFrame()
        self.post_treatment_data = pd.DataFrame()
        self.pre_treatment_data = pd.DataFrame()
        self.p_values = []
        self.coef_values = []
        print("Step 0: Initializing TumorAnalysis class...")
        self.validate_files(clinical_data_path, volumes_data_paths)
        self.load_clinical_data(clinical_data_path)
        self.load_volumes_data(volumes_data_paths)
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

    def load_clinical_data(self, clinical_data_path):
        """
        Load clinical data from a CSV file, parse the clinical data to
        categorize diagnosis into glioma types, and categorize other
        relevant fields, and reduce the data to relevant columns for analysis.
        Updates the `self.clinical_data_reduced` attribute.

        Parameters:
        - clinical_data_path (str): Path to the clinical data file.

        The function updates the `self.clinical_data` attribute with processed data.
        """
        self.clinical_data = pd.read_csv(clinical_data_path)

        self.clinical_data["Treatment_Type"] = self.extract_treatment_types()
        self.clinical_data["BCH MRN"] = zero_fill(self.clinical_data["BCH MRN"], 7)

        diagnosis_to_glioma_type = {
            "astrocytoma": "Astrocytoma",
            "JPA": "Astrocytoma",
            "xanthoastrocytoma": "Astrocytoma",
            "tectal": "Tectal Glioma",
            "glioneuronal neoplasm": "Glioneuronal Neoplasm",
            "glioneuroma": "Glioneuronal Neoplasm",
            "DNET": "Glioneuronal Neoplasm",
            "pseudotumor cerebri": "Glioneuronal Neoplasm",
            "neuroepithelial": "Glioneuronal Neoplasm",
            "ganglioglioma": "Ganglioglioma",
            "optic": "Optic Glioma",
            "NF1": "Optic Glioma",
            "low grade glioma": "Low Grade Glioma",
            "low-grade glioma": "Low Grade Glioma",
            "low grade neoplasm": "Low Grade Glioma",
            "IDH": "Low Grade Glioma",
            "oligodendroglioma ": "Low Grade Glioma",
            "infiltrating glioma": "Low Grade Glioma",
        }

        def map_diagnosis(diagnosis):
            for keyword, glioma_type in diagnosis_to_glioma_type.items():
                if keyword.lower() in diagnosis.lower():
                    return glioma_type
            return "Other"

        self.clinical_data["Glioma_Type"] = self.clinical_data["Pathologic diagnosis"].apply(
            map_diagnosis
        )

        self.clinical_data["Sex"] = self.clinical_data["Sex"].apply(
            lambda x: "Female" if x == "Female" else "Male"
        )
        self.clinical_data["Race"] = self.clinical_data["Race/Ethnicity"]
        self.clinical_data["Race"] = self.clinical_data["Race"].apply(process_race_ethnicity)
        self.clinical_data["Mutations"] = self.clinical_data.apply(
            lambda row: "Yes"
            if row["BRAF V600E mutation"] == "Yes"
            or row["BRAF fusion"] == "Yes"
            or row["FGFR fusion"] == "Yes"
            else "No",
            axis=1,
        )
        self.clinical_data["Date_of_Diagnosis"] = pd.to_datetime(
            self.clinical_data["Date of MRI diagnosis"], dayfirst=True
        )
        self.clinical_data["Date_of_First_Progression"] = pd.to_datetime(
            self.clinical_data["Date of First Progression"], dayfirst=True
        )
        self.clinical_data["Tumor_Progression"] = self.clinical_data["Progression"].map(
            {"Yes": True, "No": False, None: False}
        )

        dtype_mapping = {
            "BCH MRN": "string",
            "Glioma_Type": "category",
            "Sex": "category",
            "Race": "category",
            "Mutations": "category",
            "Treatment_Type": "category",
            "Tumor_Progression": "bool",
        }

        # Apply the type conversions according to the dictionary
        for column, dtype in dtype_mapping.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        datetime_columns = ["Date_of_Diagnosis", "Date_of_First_Progression"]
        all_relevant_columns = list(dtype_mapping.keys()) + datetime_columns
        self.clinical_data_reduced = self.clinical_data[all_relevant_columns]

        print("\tParsed clinical data.")

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
            for file in all_files:
                patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
                patient_id = file.split(".")[0]
                patient_df["Patient_ID"] = patient_id
                patient_df["Patient_ID"] = (
                    patient_df["Patient_ID"].astype(str).str.zfill(7).astype("string")
                )
                data_frames.append(patient_df)

            print(f"\tLoaded volume data {volumes_data_path}.")
        self.volumes_data = pd.concat(data_frames, ignore_index=True)
        self.volumes_data["Date"] = pd.to_datetime(self.volumes_data["Date"], format="%Y-%m-%d")

    def extract_treatment_types(self):
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

            if len(treatments) == 0:
                treatment_list.append("No Treatment")
            elif len(treatments) == 1:
                treatment_list.append(f"{treatments[0]} Only")
            elif len(treatments) == 2:
                treatment_list.append(f"{treatments[0]} and {treatments[1]}")
            elif len(treatments) == 3:
                treatment_list.append("All Treatments")

        return treatment_list

    def merge_data(self):
        """
        Merge reduced clinical data with the volumes data based on patient ID,
        excluding redundant columns.

        This function updates the `self.merged_data` attribute with the merged DataFrame.
        """
        self.merged_data = pd.merge(
            self.clinical_data_reduced,
            self.volumes_data,
            left_on=["BCH MRN"],
            right_on=["Patient_ID"],
            how="right",
        )
        self.merged_data = self.merged_data.drop(columns=["BCH MRN"])
        self.merged_data["Age_Group"] = self.merged_data.apply(categorize_age_group, axis=1).astype(
            "category"
        )
        print("\tMerged clinical and volume data.")

    def aggregate_summary_statistics(self):
        """
        Calculate summary statistics for specified columns in the merged data.

        This function updates the `self.merged_data` with new columns for mean,
        median, and standard deviation for each of the specified columns.
        """

        def cumulative_stats(group, variable):
            group[f"{variable}_CumMean"] = group[variable].expanding().mean()
            group[f"{variable}_CumMedian"] = group[variable].expanding().median()
            group[f"{variable}_CumStd"] = group[variable].expanding().std()
            return group

        def rolling_stats(group, variable, window_size=3, min_periods=1):
            group[f"{variable}_RollMean"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).mean()
            )
            group[f"{variable}_RollMedian"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).median()
            )
            group[f"{variable}_RollStd"] = (
                group[variable].rolling(window=window_size, min_periods=min_periods).std()
            )
            return group

        for var in ["Volume", "Growth[%]"]:
            self.merged_data = self.merged_data.groupby("Patient_ID", as_index=False).apply(
                cumulative_stats, var
            )
            self.merged_data = self.merged_data.groupby("Patient_ID", as_index=False).apply(
                rolling_stats, var
            )

        print("\tAdded rolling and accumulative summary statistics.")

    def longitudinal_separation(self):
        """
        Separate the merged data into two DataFrames based on whether the
        data is from before or after the first treatment date.

        This function updates `self.pre_treatment_data` and `self.post_treatment_data`
        with the separated data.
        """
        pre_treatment_data_frames = []
        post_treatment_data_frames = []

        for patient_id, data in self.merged_data.groupby("Patient_ID"):
            first_treatment_date = self.extract_treatment_dates(patient_id)
            received_treatment = not pd.isna(first_treatment_date)

            data["Date_of_First_Treatment"] = first_treatment_date
            data["Received_Treatment"] = received_treatment

            pre_treatment = (
                data[data["Date"] < first_treatment_date] if received_treatment else data
            )
            post_treatment = (
                data[data["Date"] >= first_treatment_date] if received_treatment else pd.DataFrame()
            )

            pre_treatment_data_frames.append(pre_treatment)

            if not post_treatment.empty:  # Only append if there is post-treatment data, i.e. if
                # treatment was received at any point, avoid patients with no treatment
                post_treatment_data_frames.append(post_treatment)

        # Concatenate the list of DataFrames into a single DataFrame for pre and post treatment
        self.pre_treatment_data = pd.concat(pre_treatment_data_frames, ignore_index=True)
        self.post_treatment_data = pd.concat(post_treatment_data_frames, ignore_index=True)

    def extract_treatment_dates(self, patient_id):
        """
        Extract the dates of treatments from the clinical data for a specific patient.

        Parameters:
        - patient_id (str): The ID of the patient.

        Returns:
        - treatment_dates (dict): A dictionary of treatment types and their corresponding
        dates for the specified patient.
        """
        patient_data = self.clinical_data[self.clinical_data["BCH MRN"] == patient_id].iloc[0]

        treatment_dates = {}

        if patient_data["Surgical Resection"] == "Yes":
            treatment_dates["Surgery"] = patient_data["Date of first surgery"]

        if patient_data["Systemic therapy before radiation"] == "Yes":
            treatment_dates["Chemotherapy"] = patient_data["Date of Systemic Therapy Start"]

        if patient_data["Radiation as part of initial treatment"] == "Yes":
            treatment_dates["Radiation"] = patient_data["Start Date of Radiation"]

        # print(f"\tPatient {patient_id} - Treatment Dates: {treatment_dates}")
        treatment_dates = [
            pd.to_datetime(date, dayfirst=True)
            for date in treatment_dates.values()
            if pd.notnull(date)
        ]

        first_treatment_date = min(treatment_dates, default=pd.NaT)
        return first_treatment_date

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
        test_result = None  # Initialize test_result
        test_type = ""
        x_dtype = data[x_val].dtype
        y_dtype = data[y_val].dtype

        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            if method == "pearson":
                coef, p_val = pearson_correlation(data[x_val], data[y_val])
            elif method == "spearman":
                coef, p_val = spearman_correlation(data[x_val], data[y_val])
            print(
                f"\t\t{x_val} and {y_val} - {method.title()} Correlation Coefficient: {coef},"
                f" P-value: {p_val}"
            )
            test_result = (coef, p_val)
            test_type = "correlation"
        elif pd.api.types.is_categorical_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            categories = data[x_val].nunique()
            if categories == 2 and method == "t-test":
                t_stat, p_val = ttest(data, x_val, y_val)
                print(
                    f"\t\tT-test for {x_val} and {y_val} - t-statistic: {t_stat}, P-value: {p_val}"
                )
                test_result = (t_stat, p_val)
                test_type = "t-test"
            elif categories == 2 and method == "point-biserial":
                coef, p_val = point_bi_serial(data, x_val, y_val)
                print(
                    f"\t\tPoint-Biserial Correlation for {x_val} and {y_val} - Coefficient:"
                    f" {coef}, P-value: {p_val}"
                )
                test_result = (coef, p_val)
                test_type = "point-biserial"
            else:
                # For more than two categories, use ANOVA
                f_stat, p_val = f_one(data, x_val, y_val)
                print(
                    f"\t\tANOVA for {x_val} and {y_val} - F-statistic: {f_stat}, P-value: {p_val}"
                )
                test_result = (f_stat, p_val)
                test_type = "ANOVA"
        elif pd.api.types.is_categorical_dtype(x_dtype) and pd.api.types.is_categorical_dtype(
            y_dtype
        ):
            chi2, p_val, _, _ = chi_squared_test(data, x_val, y_val)
            print(f"\t\tChi-Squared test for {x_val} and {y_val} - Chi2: {chi2}, P-value: {p_val}")
            test_result = (chi2, p_val)
            test_type = "chi-squared"

        if test_result:
            # Visualize the statistical test
            self.visualize_statistical_test(
                x_val, y_val, data, test_result, prefix, output_dir, test_type, method=method
            )

            self.p_values.append(p_val)
            if test_type == "correlation":
                self.coef_values.append(coef)
        else:
            print(
                f"\t\tCould not perform analysis on {x_val} and {y_val} due to incompatible data"
                " types."
            )

    def analyze_pre_treatment(self, prefix, output_dir, correlation_method="spearman"):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, and race.
        """
        print("\tPre-treatment Correlations:")
        # Data preparation for correlations
        self.pre_treatment_data["Received_Treatment"] = (
            self.pre_treatment_data["Received_Treatment"]
            .map({1: True, 0: False})
            .astype("category")
        )
        self.pre_treatment_data["Tumor_Progression"] = self.pre_treatment_data[
            "Tumor_Progression"
        ].astype("category")
        # self.time_to_treatment_effect()

        # variable types
        categorical_vars = [
            "Glioma_Type",
            # "Race",
            "Treatment_Type",
            "Age_Group",
            "Sex",
            "Mutations",
            "Received_Treatment",
            "Tumor_Progression",
        ]
        numerical_vars = [
            "Age",
            "Volume",
            "Growth[%]",
            "Volume_CumMean",
            "Volume_CumMedian",
            "Volume_CumStd",
            "Volume_RollMean",
            "Volume_RollMedian",
            "Volume_RollStd",
            "Growth[%]_CumMean",
            "Growth[%]_CumMedian",
            "Growth[%]_CumStd",
            "Growth[%]_RollMean",
            "Growth[%]_RollMedian",
            "Growth[%]_RollStd",
            # "Time_to_Treatment",
        ]

        for num_var in numerical_vars:
            for cat_var in categorical_vars:
                if self.pre_treatment_data[cat_var].nunique() == 2:
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method="t-test",
                    )
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method="point-biserial",
                    )
                else:
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method="ANOVA",
                    )
            for other_num_var in numerical_vars:
                if (other_num_var != num_var) and (
                    not other_num_var.startswith(("Growth[%]_", "Volume_"))
                ):
                    self.analyze_correlation(
                        num_var,
                        other_num_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method="spearman",
                    )
                    self.analyze_correlation(
                        num_var,
                        other_num_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method="pearson",
                    )
        for cat_var in categorical_vars:
            for other_cat_var in categorical_vars:
                if cat_var != other_cat_var:
                    self.analyze_correlation(
                        cat_var,
                        other_cat_var,
                        self.pre_treatment_data,
                        prefix,
                        output_dir,
                        method=correlation_method,
                    )

    def analyze_post_treatment(self, prefix, output_dir, correlation_method="spearman"):
        """
        Analyze data for post-treatment cases. This involves finding correlations between
        variables such as treatment types, tumor volume changes, and specific mutations.
        """
        print("Post-treatment Correlations:")
        self.analyze_correlation(
            "Treatment_Type",
            "Tumor_Volume_Change",
            self.post_treatment_data,
            prefix,
            output_dir,
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Tumor_Volume_Change",
            self.post_treatment_data,
            prefix,
            output_dir,
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Treatment_Response",
            self.post_treatment_data,
            prefix,
            output_dir,
            method=correlation_method,
        )

        # Chi-Squared test
        # chi2, p_val = chi_squared_test(
        #     self.post_treatment_data["Mutation_Type"], self.post_treatment_data["Treatment_Type"]
        # )
        # print(
        #     f"Chi-Squared test between Mutation_Type and Treatment_Type: Chi2: {chi2}, P-value:"
        #     f" {p_val}"
        # )

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

        # Plot based on test type
        if test_type == "correlation":
            sns.scatterplot(x=x_val, y=y_val, data=data)
            sns.regplot(x=x_val, y=y_val, data=data, scatter=False, color="blue")
            title += f"{method.title()} correlation coefficient: {stat:.2f}, P-value: {p_val:.3e}"
        elif test_type == "t-test":
            sns.barplot(x=x_val, y=y_val, data=data)
            title += f"T-statistic: {stat:.2f}, P-value: {p_val:.3e}"
        elif test_type == "point-biserial":
            sns.boxplot(x=x_val, y=y_val, data=data)
            title += f"Point-Biserial Correlation Coefficient: {stat:.2f}, P-value: {p_val:.3e}"
        elif test_type == "ANOVA":
            sns.boxplot(x=x_val, y=y_val, data=data)
            plt.xticks(rotation=90, fontsize="small")
            title += f"F-statistic: {stat:.2f}, P-value: {p_val:.3e}"
        elif test_type == "chi-squared":
            contingency_table = pd.crosstab(data[x_val], data[y_val])
            sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="g")
            title += f"Chi2: {stat:.2f}, P-value: {p_val:.3e}"

        plt.title(title)
        plt.xlabel(x_val)
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

    def plot_individual_trajectories(self, name, data, sample_size):
        """
        Plot the individual growth trajectories for a sample of patients.

        Parameters:
        - name (str): The filename to save the plot image.

        This method samples a n number of unique patient IDs from the pre-treatment data,
        plots their tumor growth percentage over time, and saves the plot to the specified filename.
        """
        plt.figure(figsize=(10, 6))

        # Error handling for sample size
        unique_patient_count = data["Patient_ID"].nunique()
        if sample_size > unique_patient_count:
            print(
                f"\t\tSample size {sample_size} is greater than the number of unique patients"
                f" {unique_patient_count}. Using {unique_patient_count} instead."
            )
            sample_size = unique_patient_count

        # Get the sample IDs and data
        sample_ids = data["Patient_ID"].drop_duplicates().sample(n=sample_size)
        sampled_data = data[data["Patient_ID"].isin(sample_ids)]
        # Cutoff the data at 4000 days
        sampled_data = sampled_data[sampled_data["Time_since_First_Scan"] <= 4000]
        # included_treatments = [
        #     "Surgery Only",
        #     "Chemotherapy Only",
        #     "Radiation Only",
        #     "No Treatment",
        # ]
        # sampled_data = sampled_data[sampled_data["Treatment_Type"].isin(included_treatments)]

        # sampled_data["Normalized_Growth"] = sampled_data["Growth[%]"] / sampled_data["Volume"]
        sampled_data["Normalized_Growth"] = sampled_data["Growth[%]_RollMean"]
        # Get the median every 6 months
        median_data = (
            sampled_data.groupby(
                pd.cut(
                    sampled_data["Time_since_First_Scan"],
                    pd.interval_range(
                        start=0, end=sampled_data["Time_since_First_Scan"].max(), freq=182
                    ),
                )
            )["Normalized_Growth"]
            .median()
            .reset_index()
        )
        # Get the median and mean data based on the datapoints available
        # median_data = (
        #     sampled_data.groupby("Time_since_First_Scan")["Normalized_Growth"]
        #     .median()
        #     .reset_index()
        # )

        ax = sns.lineplot(
            x="Time_since_First_Scan",
            y="Normalized_Growth",
            hue="Treatment_Type",  # "Patient_ID",
            data=sampled_data,
            palette="CMRmap",
            legend="brief",  # Set to 'brief' or 'full' if you want the legend
            alpha=0.5,  # Set lower alpha for better visibility when lines overlap
        )

        # mean_data = sampled_data.groupby("Time_since_First_Scan")["Growth_pct"].mean().reset_index()
        # sns.lineplot(
        #     x="Time_since_First_Scan",
        #     y="Growth_pct",
        #     data=mean_data,
        #     color="blue",
        #     linestyle="--",
        #     label="Mean Trajectory",
        # )

        sns.lineplot(
            x=median_data["Time_since_First_Scan"].apply(lambda x: x.mid),
            y="Normalized_Growth",
            data=median_data,
            color="blue",
            linestyle="--",
            label="Median Trajectory",
        )

        plt.xlabel("Days Since First Scan")
        plt.ylabel("Tumor Growth Pct -- Rolling Mean")
        plt.title("Individual Growth Trajectories")

        # Adjust the legend to only include the categories present in the filtered data
        # handles, labels = ax.get_legend_handles_labels()
        # filtered_labels_handles = dict(zip(labels, handles))
        # included_handles = [
        #     filtered_labels_handles[label]
        #     for label in included_treatments
        #     if label in filtered_labels_handles
        # ]
        # ax.legend(included_handles, included_treatments)

        plt.legend()
        plt.savefig(name)
        plt.close()

    def plot_growth_predictions(self, data, filename):
        """
        Plot the actual versus predicted tumor growth percentages over time.

        Parameters:
        - filename (str): The filename to save the plot image.

        This method plots the actual and predicted growth percentages from the
        pre-treatment data and saves the plot as an image.
        """
        # TODO: Rethink the prediction plot and what exactly should be plotted.
        sns.lineplot(
            x="Time_since_First_Scan",
            y="Growth_pct",
            data=data,
            alpha=0.5,
            color="blue",
            label="Actual Growth",
        )
        sns.lineplot(
            x="Time_since_First_Scan",
            y="Predicted_Growth_pct",
            data=data,
            color="red",
            label="Predicted Growth",
        )
        plt.xlabel("Days Since First Scan")
        plt.ylabel("Tumor Growth Percentage")
        plt.title("Actual vs Predicted Tumor Growth Over Time")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def model_growth_trajectories(self, prefix, output_dir):
        """
        Model the growth trajectories of patients using a mixed-effects linear model.

        Parameters:
        - prefix (str): Prefix used for naming the output files.

        The method models the tumor growth percentage as a function of time since the first
        scan for each patient, separates the individual and predicted growth trajectories, and
        saves the plots to files.
        """
        print("\tModeling growth trajectories:")
        pre_treatment_data = self.pre_treatment_data.copy()
        pre_treatment_data.sort_values(by=["Patient_ID", "Date"], inplace=True)
        pre_treatment_data["Time_since_First_Scan"] = pre_treatment_data.groupby("Patient_ID")[
            "Date"
        ].transform(lambda x: (x - x.min()).dt.days)
        pre_treatment_data["Growth_pct"] = pd.to_numeric(
            pre_treatment_data["Growth[%]"], errors="coerce"
        )
        pre_treatment_data = pre_treatment_data.dropna(
            subset=["Growth_pct", "Time_since_First_Scan"]
        )

        # TODO: adjust for post-treatment data

        try:
            # Ensure we have enough data points per patient
            sufficient_data_patients = (
                pre_treatment_data.groupby("Patient_ID")
                .filter(lambda x: len(x) >= 3)["Patient_ID"]
                .unique()
            )
            filtered_data = pre_treatment_data[
                pre_treatment_data["Patient_ID"].isin(sufficient_data_patients)
            ]

            # Reset index after filtering
            filtered_data.reset_index(drop=True, inplace=True)

            # Continue with filtered data
            if filtered_data.empty:
                print("No patients have enough data points for mixed-effects model analysis.")
                return

            model = sm.MixedLM.from_formula(
                "Growth_pct ~ Time_since_First_Scan",
                re_formula="~Time_since_First_Scan",
                groups=filtered_data["Patient_ID"],
                data=filtered_data,
            )
            # pylint: disable=unexpected-keyword-arg
            result = model.fit(reml=False, method="nm", maxiter=200)
            # Using method Nelder-Mead optimization, although not really needed
            # 200 iterations for kernel smoothed data
            # result = model.fit()  # Using default optimization, fails to converge

            if not result.converged:
                print("\t\tModel did not converge, try simplifying the model or check the data.")
                return None
            else:
                # print(result.summary())
                print("\t\tModel converged.")
                pre_treatment_data["Predicted_Growth_pct"] = result.predict(pre_treatment_data)
                prediciton_plot = os.path.join(output_dir, f"{prefix}_growth_predictions.png")
                self.plot_growth_predictions(data=pre_treatment_data, filename=prediciton_plot)
                print("\t\tSaved growth predictions plot.")
        except ValueError as err:
            print(f"ValueError: {err}")

        finally:
            growth_trajectories_plot = os.path.join(
                output_dir, f"{prefix}_growth_trajectories_plot.png"
            )
            self.plot_individual_trajectories(
                growth_trajectories_plot,
                data=pre_treatment_data,
                sample_size=correlation_cfg.SAMPLE_SIZE,
            )
            print("\t\tSaved growth trajectories plot.")

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
        analysis_data_pre = self.pre_treatment_data.copy()
        analysis_data_pre = analysis_data_pre[
            self.pre_treatment_data["Date_of_First_Progression"]
            < self.pre_treatment_data["Date_of_First_Treatment"]
        ]
        analysis_data_pre.loc[:, "Duration"] = (
            analysis_data_pre["Date_of_First_Progression"] - analysis_data_pre["Date_of_Diagnosis"]
        ).dt.days

        analysis_data_pre["Event_Occurred"] = ~analysis_data_pre["Date_of_First_Progression"].isna()

        # TODO: Add cases of survival in the post treatment setting, adjust the data accordingly
        # TODO: Add a if/else condition based on the prefix and generalize data_handling

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
            output_dir, f"{prefix}_survival_plot_stratifyby_{stratify_by}.png"
        )
        plt.savefig(survival_plot, dpi=300)
        plt.close()
        print(f"\t\tSaved survival KaplanMeier curve for {stratify_by}.")

    def time_to_treatment_effect(self):
        """
        Analyze the correlation between the time to treatment and tumor growth.

        Parameters:
        - prefix (str): Prefix used for the output files.
        - path (str): The directory path to save output files.

        The method analyzes the correlation, fits a linear model to predict growth based on time to treatment, and visualizes the effect.
        """
        # print("\tAnalyzing time to treatment effect:")
        for patient_id in self.pre_treatment_data["Patient_ID"].unique():
            patient_data = self.pre_treatment_data[
                self.pre_treatment_data["Patient_ID"] == patient_id
            ]

            # Assuming 'Last_Scan_Date' is the column with the last date for the patient
            if patient_data["Date_of_First_Treatment"].isna().any():
                last_scan_date = patient_data["Date"].max()
                self.pre_treatment_data.loc[
                    patient_data.index, "Date_of_First_Treatment"
                ] = last_scan_date

        self.pre_treatment_data["Time_to_Treatment"] = (
            self.pre_treatment_data["Date_of_First_Treatment"]
            - self.pre_treatment_data["Date_of_Diagnosis"]
        ).dt.days

        for _, row in self.pre_treatment_data.iterrows():
            if pd.isna(row["Time_to_Treatment"]):
                reason = ""
                if pd.isna(row["Date_of_First_Treatment"]):
                    reason += "Missing Date_of_First_Treatment. "
                if pd.isna(row["Date_of_Diagnosis"]):
                    reason += "Missing Date_of_Diagnosis. "
                print(
                    f"Patient ID {row['Patient_ID']} has NaN in Time_to_Treatment. Reason: {reason}"
                )

        # filtered_data = self.pre_treatment_data.dropna(subset=["Time_to_Treatment", "Growth[%]"])
        # filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan).dropna(
        #     subset=["Time_to_Treatment", "Growth[%]"]
        # )
        # self.analyze_correlation(
        #     "Time_to_Treatment", "Growth[%]", filtered_data, prefix, output_dir, method="pearson"
        # )

        # model = sm.OLS(
        #     filtered_data["Growth[%]"], sm.add_constant(filtered_data["Time_to_Treatment"])
        # )
        # results = model.fit()
        # # print(results.summary())

        # filtered_data["Predicted_Growth"] = results.predict(
        #     sm.add_constant(filtered_data["Time_to_Treatment"])
        # )

    def analyze_tumor_stability(self, data, output_dir, volume_weight=0.5, growth_weight=0.5):
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
        data = calculate_group_norms_and_stability(data, output_dir)

        # Calculate the Stability Index using weighted scores
        data["Stability_Index"] = volume_weight * data["Volume_Stability_Score"] + growth_weight * (
            1 / (data["Growth[%]_RollStd"] + 1)
        )

        # Normalize the Stability Index to have a mean of 1
        data["Stability_Index"] /= np.mean(data["Stability_Index"])

        # Define thresholds for classification based on the normalized Stability Index
        # Use the 50th percentile (median) as the threshold by default, can be adjusted if needed
        stability_threshold = np.median(data["Stability_Index"])

        data["Tumor_Classification"] = data["Stability_Index"].apply(
            lambda x: "Stable" if x <= stability_threshold else "Unstable"
        )
        classification_distribution = data["Tumor_Classification"].value_counts(normalize=True)

        # Visualization of Tumor Classification
        plt.figure(figsize=(10, 6))
        sns.countplot(x="Age_Group", hue="Tumor_Classification", data=data)
        plt.title("Count of Stable vs. Unstable Tumors Across Age Groups")
        plt.xlabel("Age Group")
        plt.ylabel("Count")
        plt.legend(title="Tumor Classification")
        plt.tight_layout()
        filename = os.path.join(output_dir, "tumor_classification.png")
        plt.savefig(filename)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.pie(
            classification_distribution,
            labels=classification_distribution.index,
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Proportion of Stable vs. Unstable Tumors")
        plt.axis("equal")
        filename_dist = os.path.join(output_dir, "tumor_classification_distribution.png")
        plt.savefig(filename_dist)
        plt.close()

        # Enhanced Scatter Plot with Labels for Extremes
        plt.figure(figsize=(18, 6))
        ax = sns.scatterplot(
            x="Date", y="Stability_Index", hue="Tumor_Classification", data=data, alpha=0.6
        )
        extremes = data.nlargest(5, "Stability_Index")  # Adjust the number of points as needed
        for i, point in extremes.iterrows():
            ax.text(point["Date"], point["Stability_Index"], str(point["Patient_ID"]))
        plt.title("Scatter Plot of Stability Index Over Time by Classification")
        plt.xlabel("Date")
        plt.ylabel("Stability Index")
        plt.legend(title="Tumor Classification")
        plt.tight_layout()
        filename_scatter = os.path.join(output_dir, "stability_index_scatter.png")
        plt.savefig(filename_scatter)
        plt.close()

    def run_analysis(self, output_correlations, output_stats):
        """
        Run a comprehensive analysis pipeline consisting of data separation,
        sensitivity analysis, propensity score matching, main analysis, corrections
        for multiple comparisons, trend analysis, and feature engineering.

        This method orchestrates the overall workflow of the analysis process,
        including data preparation, statistical analysis, and results interpretation.
        """
        step_idx = 1

        print(f"Step {step_idx}: Separating data into pre- and post-treatment dataframes...")
        self.longitudinal_separation()
        # print(self.pre_treatment_data["Patient_ID"].nunique())
        # print(self.post_treatment_data["Patient_ID"].nunique())

        assert self.pre_treatment_data.columns.all() == self.post_treatment_data.columns.all()
        assert (len(self.pre_treatment_data) + len(self.post_treatment_data)) == len(
            self.merged_data
        )
        print(
            "\tData separated, assertation for same columns in separated dataframes and"
            " corresponding lenghts passed."
        )
        step_idx += 1

        if correlation_cfg.SENSITIVITY:
            print(f"Step {step_idx}: Performing Sensitivity Analysis...")

            pre_treatment_vars = [
                "Volume",
                "Growth[%]",
                "Volume_RollMean",
                "Volume_RollStd",
                "Growth[%]_RollMean",
                "Growth[%]_RollStd",
            ]
            post_treatment_vars = [
                "Volume",
                "Growth[%]",
                "Volume_CumMean",
                "Volume_CumStd",
                "Growth[%]_CumMean",
                "Growth[%]_CumStd",
            ]

            for pre_var in pre_treatment_vars:
                print(f"\tPerforming sensitivity analysis on pre-treatment variables {pre_var}...")
                self.pre_treatment_data = sensitivity_analysis(
                    self.pre_treatment_data,
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
            # print(self.pre_treatment_data["Patient_ID"].nunique())
            # print(self.post_treatment_data["Patient_ID"].nunique())

        if correlation_cfg.PROPENSITY:
            print(f"Step {step_idx}: Performing Propensity Score Matching...")

            for data in [self.pre_treatment_data]:
                treatment_column = "Received_Treatment"
                covariate_columns = ["Age", "Volume", "Growth[%]"]

                data[treatment_column] = data[treatment_column].map({True: 1, False: 0})
                propensity_scores = calculate_propensity_scores(
                    data, treatment_column, covariate_columns
                )
                matched_data = perform_propensity_score_matching(
                    data, propensity_scores, treatment_column, caliper=0.05
                )
                smd_results = check_balance(matched_data, covariate_columns, treatment_column)

                visualize_smds(smd_results, path=output_stats)

            step_idx += 1

        if correlation_cfg.ANLYSIS:
            print(f"Step {step_idx}: Starting main analyses...")
            prefix = "pre-treatment"
            # print(self.pre_treatment_data.dtypes)
            # self.analyze_pre_treatment(
            #     correlation_method=correlation_cfg.CORRELATION_PRE_TREATMENT, prefix=prefix, output_dir=output_correlations
            # )

            # print(self.pre_treatment_data.dtypes)

            # Survival analysis
            stratify_by_list = ["Glioma_Type", "Sex", "Mutations", "Age_Group"]
            for element in stratify_by_list:
                self.time_to_event_analysis(prefix, output_dir=output_stats, stratify_by=element)

            # self.analyze_time_to_treatment_effect(prefix, output_correlations)
            self.model_growth_trajectories(prefix, output_dir=output_stats)

            self.analyze_tumor_stability(
                data=self.pre_treatment_data,
                output_dir=output_stats,
            )

            # prefix = "post-treatment"
            # self.analyze_post_treatment(correlation_method=correlation_cfg.CORRELATION_POST_TREATMENT, prefix=prefix)

            step_idx += 1

        if correlation_cfg.CORRECTION:
            print(f"Step {step_idx}: Starting Corrections...")
            print("\tBonferroni Correction... ")
            alpha = correlation_cfg.CORRECTION_ALPHA

            corrected_p_values_bonf = bonferroni_correction(self.p_values, alpha=alpha)
            visualize_p_value_bonferroni_corrections(
                self.p_values, corrected_p_values_bonf, alpha, output_stats
            )
            print("\tFalse Discovery Rate Correction... ")
            corrected_p_values_fdr, is_rejected = fdr_correction(self.p_values, alpha=alpha)
            visualize_fdr_correction(
                self.p_values, corrected_p_values_fdr, is_rejected, alpha, output_stats
            )
            step_idx += 1

        if correlation_cfg.TRENDS:
            print(f"Step {step_idx}: Starting Trend Analysis...")
            # TODO: Add trend analysis here
            step_idx += 1

        if correlation_cfg.FEATURE_ENG:
            print(f"Step {step_idx}: Starting Feature Engineering...")
            save_for_deep_learning(self.pre_treatment_data, output_stats, prefix="pre-treatment")
            save_for_deep_learning(self.post_treatment_data, output_stats, prefix="post-treatment")
            step_idx += 1


if __name__ == "__main__":
    analysis = TumorAnalysis(
        correlation_cfg.CLINICAL_CSV,
        [correlation_cfg.VOLUMES_CSVs_45, correlation_cfg.VOLUMES_CSVs_63],
    )
    analysis.run_analysis(correlation_cfg.OUTPUT_DIR_CORRELATIONS, correlation_cfg.OUTPUT_DIR_STATS)


# def extract_trends(self, data):
#     trends = {}

#     if data.empty:
#         return trends

#     # For demonstration, assume we are interested in a time series of tumor_volume
#     time_series = data.sort_values("Date")["tumor_volume"]

#     # Polynomial Curve Fitting (2nd degree here)
#     x = np.linspace(0, len(time_series) - 1, len(time_series))
#     y = time_series.values
#     coefficients = np.polyfit(x, y, 2)
#     poly = np.poly1d(coefficients)

#     # Store the polynomial coefficients as features
#     trends["poly_coef"] = coefficients

#     # Find peaks and valleys
#     peaks, _ = find_peaks(time_series)
#     valleys, _ = find_peaks(-1 * time_series)

#     # Store number of peaks and valleys as features
#     trends["num_peaks"] = len(peaks)
#     trends["num_valleys"] = len(valleys)

#     # Calculate the mean distance between peaks as a feature (if feasible)
#     if len(peaks) > 1:
#         trends["mean_peak_distance"] = np.mean(np.diff(peaks))

#     # Calculate the mean distance between valleys as a feature (if feasible)
#     if len(valleys) > 1:
#         trends["mean_valley_distance"] = np.mean(np.diff(valleys))

#     return trends

# def trend_analysis(self, prefix):
#    self.pre_treatment_data["Time_since_Diagnosis"] = (
#     self.pre_treatment_data["Date"]
#     - self.pre_treatment_data["Date_of_Diagnosis"]
# ).dt.days

#     model = sm.OLS(
#         self.pre_treatment_data["Growth[%]"],
#         sm.add_constant(self.pre_treatment_data["Time_since_Diagnosis"])
#     )
#     results = model.fit()
#     print(results.summary())

#     # Visualize trend over time using regression results
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.regplot(
#         x="Time_since_Diagnosis", y="Growth[%]", data=self.pre_treatment_data, ax=ax,
#         line_kws={'label': f"y={results.params['Time_since_Diagnosis']:.2f}x+{results.params['const']:.2f}"}
#     )
#     ax.legend()
#     plt.title("Trend of Tumor Growth Over Time")
#     plt.xlabel("Time since Diagnosis (days)")
#     plt.ylabel("Tumor Growth (%)")
#     plt.tight_layout()
#     plt.savefig(f"{prefix}_trend_analysis.png")
#     plt.close()
