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
            if not post_treatment.empty:
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

        print(f"Patient {patient_id} - Treatment Dates: {treatment_dates}")
        treatment_dates = [
            pd.to_datetime(date, dayfirst=True)
            for date in treatment_dates.values()
            if pd.notnull(date)
        ]

        first_treatment_date = min(treatment_dates, default=pd.NaT)
        return first_treatment_date

    def analyze_correlation(self, x_val, y_val, data, prefix, method="spearman"):
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
                x_val, y_val, data, test_result, prefix, test_type, method=method
            )

            self.p_values.append(p_val)
            if test_type == "correlation":
                self.coef_values.append(coef)
        else:
            print(
                f"\t\tCould not perform analysis on {x_val} and {y_val} due to incompatible data"
                " types."
            )

    def analyze_pre_treatment(self, prefix, correlation_method="spearman"):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, and race.
        """
        print("\tPre-treatment Correlations:")
        # Data preparation for correlations
        self.pre_treatment_data["Received_Treatment"] = self.pre_treatment_data[
            "Received_Treatment"
        ].map({1: True, 0: False})
        self.time_to_treatment_effect()

        # variable types
        binary_vars = ["Tumor_Progression", "Received_Treatment"]
        categorical_vars = [
            "Glioma_Type",
            # "Race",
            "Treatment_Type",
            "Age_Group",
            "Sex",
            "Mutations",
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
            "Time_to_Treatment",
        ]

        for num_var in numerical_vars:
            for bin_var in binary_vars:
                self.analyze_correlation(
                    num_var,
                    bin_var,
                    self.pre_treatment_data,
                    prefix,
                    method=correlation_method,
                )
            for cat_var in categorical_vars:
                for corr_method in ["t-test", "point-biserial"]:
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.pre_treatment_data,
                        prefix,
                        method=corr_method,
                    )
            for other_num_var in numerical_vars:
                if other_num_var.startswith(("Growth[%]_", "Volume_")):
                    continue
                if other_num_var == num_var:
                    continue
                self.analyze_correlation(num_var, other_num_var, self.pre_treatment_data, prefix)

        for bin_var in binary_vars:
            for cat_var in categorical_vars:
                self.analyze_correlation(
                    cat_var, bin_var, self.pre_treatment_data, prefix, method=correlation_method
                )
                for other_cat_var in categorical_vars:
                    if cat_var == other_cat_var:
                        continue
                    self.analyze_correlation(
                        cat_var,
                        other_cat_var,
                        self.pre_treatment_data,
                        prefix,
                        method=correlation_method,
                    )
            for other_bin_var in binary_vars:
                if bin_var == other_bin_var:
                    continue
                self.analyze_correlation(bin_var, other_bin_var, self.pre_treatment_data, prefix)

    def analyze_post_treatment(self, prefix, correlation_method="spearman"):
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
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Tumor_Volume_Change",
            self.post_treatment_data,
            prefix,
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Treatment_Response",
            self.post_treatment_data,
            prefix,
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
        self, x_val, y_val, data, test_result, prefix, test_type="correlation", method="spearman"
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
        save_dir = correlation_cfg.OUTPUT_DIR
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

        save_file = os.path.join(save_dir, f"{prefix}_{x_val}_vs_{y_val}_{test_type}.png")
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
            heat_map_file = os.path.join(save_dir, f"{prefix}_{method}_correlation_heatmap.png")
            plt.savefig(heat_map_file)
            plt.close()

    def plot_individual_trajectories(self, name, sample_size):
        """
        Plot the individual growth trajectories for a sample of patients.

        Parameters:
        - name (str): The filename to save the plot image.

        This method samples a n number of unique patient IDs from the pre-treatment data,
        plots their tumor growth percentage over time, and saves the plot to the specified filename.
        """
        plt.figure(figsize=(10, 6))

        sample_ids = self.pre_treatment_data["Patient_ID"].drop_duplicates().sample(n=sample_size)
        sampled_data = self.pre_treatment_data[
            self.pre_treatment_data["Patient_ID"].isin(sample_ids)
        ]

        sns.lineplot(
            x="Time_since_First_Scan",
            y="Growth_pct",
            hue="Patient_ID",
            data=sampled_data,
            legend="brief",  # Set to 'brief' or 'full' if you want the legend
            alpha=0.5,  # Set lower alpha for better visibility when lines overlap
        )
        plt.xlabel("Days Since First Scan")
        plt.ylabel("Tumor Growth Percentage")
        plt.title("Individual Growth Trajectories")
        plt.legend()
        plt.savefig(name)
        plt.close()

    def plot_growth_predictions(self, filename):
        """
        Plot the actual versus predicted tumor growth percentages over time.

        Parameters:
        - filename (str): The filename to save the plot image.

        This method plots the actual and predicted growth percentages from the pre-treatment data and saves the plot as an image.
        """
        sns.scatterplot(
            x="Time_since_First_Scan",
            y="Growth_pct",
            data=self.pre_treatment_data,
            alpha=0.5,
            label="Actual Growth",
        )
        sns.lineplot(
            x="Time_since_First_Scan",
            y="Predicted_Growth_pct",
            data=self.pre_treatment_data,
            color="red",
            label="Predicted Growth",
        )
        plt.xlabel("Days Since First Scan")
        plt.ylabel("Tumor Growth Percentage")
        plt.title("Actual vs Predicted Tumor Growth Over Time")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def time_to_event_analysis(self, prefix):
        """
        Perform a Kaplan-Meier survival analysis on time-to-event data for tumor progression.

        Parameters:
        - prefix (str): Prefix used for naming the output file.

        The method fits the survival curve using the KaplanMeierFitter on the pre-treatment data, saves the plot image, and prints a confirmation message.
        """
        print("\tAnalyzing time to event:")
        pre_treatment_data = self.pre_treatment_data.loc[
            self.pre_treatment_data["Date_of_First_Progression"]
            < self.pre_treatment_data["Date_of_First_Treatment"]
        ].copy()

        pre_treatment_data.loc[:, "Duration"] = (
            pre_treatment_data["Date_of_First_Progression"]
            - pre_treatment_data["Date_of_Diagnosis"]
        ).dt.days

        pre_treatment_data["Event_Occurred"] = self.pre_treatment_data["Tumor_Progression"]
        kmf = KaplanMeierFitter()
        kmf.fit(pre_treatment_data["Duration"], event_observed=pre_treatment_data["Event_Occurred"])
        ax = kmf.plot_survival_function()
        ax.set_title("Survival function of Tumor Progression")
        ax.set_xlabel("Days since Diagnosis")
        ax.set_ylabel("Survival Probability")

        survival_plot = os.path.join(correlation_cfg.OUTPUT_DIR, f"{prefix}_survival_plot.png")
        plt.savefig(survival_plot, dpi=300)
        plt.close()
        print("\t\tSaved survival KaplanMeier curve.")

    def model_growth_trajectories(self, prefix):
        """
        Model the growth trajectories of patients using a mixed-effects linear model.

        Parameters:
        - prefix (str): Prefix used for naming the output files.

        The method models the tumor growth percentage as a function of time since the first
        scan for each patient, separates the individual and predicted growth trajectories, and
        saves the plots to files.
        """
        print("\tModeling growth trajectories:")
        self.pre_treatment_data.sort_values(by=["Patient_ID", "Date"], inplace=True)

        self.pre_treatment_data["Time_since_First_Scan"] = self.pre_treatment_data.groupby(
            "Patient_ID"
        )["Date"].transform(lambda x: (x - x.min()).dt.days)

        self.pre_treatment_data["Growth_pct"] = pd.to_numeric(
            self.pre_treatment_data["Growth[%]"], errors="coerce"
        )
        self.pre_treatment_data = self.pre_treatment_data.dropna(
            subset=["Growth_pct", "Time_since_First_Scan"]
        )

        # Ensure we have enough data points per patient
        sufficient_data_patients = (
            self.pre_treatment_data.groupby("Patient_ID")
            .filter(lambda x: len(x) >= 3)["Patient_ID"]
            .unique()
        )
        filtered_data = self.pre_treatment_data[
            self.pre_treatment_data["Patient_ID"].isin(sufficient_data_patients)
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
        result = model.fit(reml=False, method="nm", maxiter=100)  # Using Nelder-Mead optimization
        # result = model.fit() # Using default optimization, fails to converge

        if not result.converged:
            print("Model did not converge, try simplifying the model or check the data.")
            return None
        else:
            print(result.summary())
            print("\t\tModel converged.")
            self.pre_treatment_data["Predicted_Growth_pct"] = result.predict(
                self.pre_treatment_data
            )

            growth_trajectories_plot = os.path.join(
                correlation_cfg.OUTPUT_DIR, f"{prefix}_growth_trajectories_plot.png"
            )
            prediciton_plot = os.path.join(
                correlation_cfg.OUTPUT_DIR, f"{prefix}_growth_predictions.png"
            )

            self.plot_individual_trajectories(
                growth_trajectories_plot, sample_size=correlation_cfg.SAMPLE_SIZE
            )
            self.plot_growth_predictions(prediciton_plot)
            print("\t\tSaved growth trajectories plot.")

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

        for index, row in self.pre_treatment_data.iterrows():
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
        #     "Time_to_Treatment", "Growth[%]", filtered_data, prefix, method="pearson"
        # )

        # model = sm.OLS(
        #     filtered_data["Growth[%]"], sm.add_constant(filtered_data["Time_to_Treatment"])
        # )
        # results = model.fit()
        # # print(results.summary())

        # filtered_data["Predicted_Growth"] = results.predict(
        #     sm.add_constant(filtered_data["Time_to_Treatment"])
        # )

    def analyze_tumor_stability(self, unchanging_threshold=0.05):
        """
        Analyze the stability of tumors based on their growth rates.

        Returns:
        - stable_patients_data (DataFrame): Data of patients with stable tumors.

        This method identifies tumors with negligible growth over time and analyzes
        the characteristics of patients with these stable tumors.
        """
        # Calculate the absolute change in tumor growth for each scan
        self.pre_treatment_data["Absolute_Growth_change"] = (
            self.pre_treatment_data.groupby("Patient_ID")["Growth[%]"].diff().abs()
        )

        patients_with_significant_change = self.pre_treatment_data.groupby("Patient_ID")[
            "Growth[%]"
        ].apply(lambda x: (x.abs() > unchanging_threshold).any())

        # Get lists of patient IDs for stable and changing tumors
        stable_patients_ids = patients_with_significant_change[
            ~patients_with_significant_change
        ].index.tolist()
        changing_patients_ids = patients_with_significant_change[
            patients_with_significant_change
        ].index.tolist()

        # Extract data for stable and changing tumors
        stable_patients_data = self.pre_treatment_data[
            self.pre_treatment_data["Patient_ID"].isin(stable_patients_ids)
        ]
        changing_patients_data = self.pre_treatment_data[
            self.pre_treatment_data["Patient_ID"].isin(changing_patients_ids)
        ]

        print("Stable patients:", len(stable_patients_data))
        print(stable_patients_data["Patient_ID"].unique())
        print("Changing patients:", len(changing_patients_data))
        print(changing_patients_data["Patient_ID"].unique())

        self.compare_stable_unstable_distributions(stable_patients_data, changing_patients_data)

    def compare_stable_unstable_distributions(self, stable_patients_data, changing_patients_data):
        # Calculate descriptive statistics for Age and Volume for stable patients
        stable_age_stats = stable_patients_data["Age"].describe()
        stable_volume_stats = stable_patients_data["Volume"].describe()

        # Calculate descriptive statistics for Age and Volume for changing patients
        changing_age_stats = changing_patients_data["Age"].describe()
        changing_volume_stats = changing_patients_data["Volume"].describe()

        # print("Stable Tumors Age Statistics:\n", stable_age_stats)
        # print("Stable Tumors Volume Statistics:\n", stable_volume_stats)
        # print("Changing Tumors Age Statistics:\n", changing_age_stats)
        # print("Changing Tumors Volume Statistics:\n", changing_volume_stats)

        # Visualization of Age Distributions
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(stable_patients_data["Age"], color="blue", kde=True, label="Stable Tumors")
        sns.histplot(changing_patients_data["Age"], color="red", kde=True, label="Changing Tumors")
        plt.title("Distribution of Ages")
        plt.xlabel("Age")
        plt.legend()

        # Visualization of Volume Distributions
        plt.subplot(1, 2, 2)
        sns.histplot(stable_patients_data["Volume"], color="blue", kde=True, label="Stable Tumors")
        sns.histplot(
            changing_patients_data["Volume"], color="red", kde=True, label="Changing Tumors"
        )
        plt.title("Distribution of Volumes")
        plt.xlabel("Volume")
        plt.legend()

        filename = os.path.join(correlation_cfg.OUTPUT_DIR, "stable_unstable_distributions.png")
        plt.tight_layout()
        plt.savefig(filename)

        # Return the statistics for potential further analysis
        return {
            "stable_age_stats": stable_age_stats,
            "stable_volume_stats": stable_volume_stats,
            "changing_age_stats": changing_age_stats,
            "changing_volume_stats": changing_volume_stats,
        }

    def run_analysis(self):
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

                visualize_smds(smd_results, path=correlation_cfg.OUTPUT_DIR)

            step_idx += 1

        if correlation_cfg.ANLYSIS:
            print(f"Step {step_idx}: Starting main analyses...")

            prefix = "pre-treatment"
            path = correlation_cfg.OUTPUT_DIR
            self.analyze_pre_treatment(
                correlation_method=correlation_cfg.CORRELATION_PRE_TREATMENT, prefix=prefix
            )

            self.time_to_event_analysis(prefix)
            # self.analyze_time_to_treatment_effect(prefix, path)
            self.model_growth_trajectories(prefix)
            self.analyze_tumor_stability(unchanging_threshold=correlation_cfg.UNCHANGING_THRESHOLD)

            # prefix = "post-treatment"
            # self.analyze_post_treatment(correlation_method=correlation_cfg.CORRELATION_POST_TREATMENT, prefix=prefix)

            step_idx += 1

        if correlation_cfg.CORRECTION:
            print(f"Step {step_idx}: Starting Corrections...")
            print("\tBonferroni Correction... ")
            path = correlation_cfg.OUTPUT_DIR
            alpha = correlation_cfg.CORRECTION_ALPHA

            corrected_p_values_bonf = bonferroni_correction(self.p_values, alpha=alpha)
            visualize_p_value_bonferroni_corrections(
                self.p_values, corrected_p_values_bonf, alpha, path
            )
            print("\tFalse Discovery Rate Correction... ")
            corrected_p_values_fdr, is_rejected = fdr_correction(self.p_values, alpha=alpha)
            visualize_fdr_correction(
                self.p_values, corrected_p_values_fdr, is_rejected, alpha, path
            )
            step_idx += 1

        if correlation_cfg.TRENDS:
            print(f"Step {step_idx}: Starting Trend Analysis...")
            # TODO: Add trend analysis here
            step_idx += 1

        if correlation_cfg.FEATURE_ENG:
            print(f"Step {step_idx}: Starting Feature Engineering...")
            output_dir = correlation_cfg.OUTPUT_DIR
            save_for_deep_learning(self.pre_treatment_data, output_dir, prefix="pre-treatment")
            save_for_deep_learning(self.post_treatment_data, output_dir, prefix="post-treatment")
            step_idx += 1


if __name__ == "__main__":
    analysis = TumorAnalysis(
        correlation_cfg.CLINICAL_CSV,
        [correlation_cfg.VOLUMES_CSVs_45, correlation_cfg.VOLUMES_CSVs_63],
    )
    analysis.run_analysis()


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
