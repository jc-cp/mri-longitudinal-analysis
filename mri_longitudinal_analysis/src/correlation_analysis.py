# pylint: disable=too-many-lines
"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations, stability and trend analysis.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from cfg import correlation_cfg
from lifelines import KaplanMeierFitter
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
    classify_patient,
    plot_trend_trajectories,
    plot_individual_trajectories,
    calculate_percentage_change,
    visualize_tumor_stability,
    read_exclusion_list,
    consistency_check,
)


class TumorAnalysis:
    """
    A class to perform tumor analysis using clinical and volumetric data.
    """

    def __init__(self, clinical_data_path, volumes_data_paths, cohort):
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
        self.progression_threshold = correlation_cfg.PROGRESSION_THRESHOLD
        self.stability_threshold = correlation_cfg.STABILITY_THRESHOLD
        self.high_risk_threshold = correlation_cfg.HIGH_RISK_THRESHOLD
        self.angle = correlation_cfg.ANGLE
        self.caliper = correlation_cfg.CALIPER
        self.sample_size_plots = correlation_cfg.SAMPLE_SIZE
        self.exclusion_list_path = correlation_cfg.EXCLUSION_LIST_PATH
        self.cohort = cohort
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

    def map_dictionary(self, dictionary, column, type):
        """
        Maps given instance according to predefined dictionary.
        """

        def map_value(cell):
            for keyword, value in dictionary.items():
                if keyword.lower() in str(cell).casefold():
                    return value
            if type == "location":
                return "Other"
            if type == "symptoms":
                return "No symptoms (incident finding)"

        return column.apply(map_value)

    def load_clinical_data(self, clinical_data_path):
        """
        Load clinical data from a CSV file, parse the clinical data to
        categorize diagnosesand other relevant fields to reduce the data for analysis.
        Updates the `self.clinical_data_reduced` attribute.

        Parameters:
        - clinical_data_path (str): Path to the clinical data file.

        The function updates the `self.clinical_data` attribute with processed data.
        """
        self.clinical_data = pd.read_csv(clinical_data_path)

        self.clinical_data["Treatment Type"] = self.extract_treatment_types()
        self.clinical_data["BCH MRN"] = zero_fill(self.clinical_data["BCH MRN"], 7)

        self.clinical_data["Location"] = self.map_dictionary(
            correlation_cfg.BCH_LOCATION, self.clinical_data["Location of Tumor"], type="location"
        )

        self.clinical_data["Symptoms"] = self.map_dictionary(
            correlation_cfg.BCH_SYMPTOMS,
            self.clinical_data["Symptoms at diagnosis"],
            type="symptoms",
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
        self.clinical_data["Date First Diagnosis"] = pd.to_datetime(
            self.clinical_data["Date of MRI diagnosis"], dayfirst=True
        )
        self.clinical_data["Date of last clinical follow-up"] = pd.to_datetime(
            self.clinical_data["Date of last clinical follow-up"], dayfirst=True
        )
        self.clinical_data["Date First Progression"] = pd.to_datetime(
            self.clinical_data["Date of First Progression"], dayfirst=True
        )
        self.clinical_data["Follow-up Time"] = (
            self.clinical_data["Date of last clinical follow-up"]
            - self.clinical_data["Date First Diagnosis"]
        ).dt.days

        self.clinical_data["Tumor Progression"] = self.clinical_data["Progression"]

        # Apply the type conversions according to the dictionary
        for column, dtype in correlation_cfg.BCH_DTYPE_MAPPING.items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        all_relevant_columns = (
            list(correlation_cfg.BCH_DTYPE_MAPPING.keys()) + correlation_cfg.BCH_DATETIME_COLUMNS
        )
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
        exclude_patients = read_exclusion_list(self.exclusion_list_path)

        data_frames = []
        for volumes_data_path in volumes_data_paths:
            all_files = [f for f in os.listdir(volumes_data_path) if f.endswith(".csv")]
            for file in all_files:
                patient_id = file.split(".")[0]

                if patient_id in exclude_patients:
                    continue

                patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
                # Get the first volume value
                baseline_volume = patient_df["Volume"].iloc[0]
                patient_df["Baseline Volume"] = baseline_volume
                # Adjust patient id
                patient_df["Patient_ID"] = patient_id
                patient_df["Patient_ID"] = (
                    patient_df["Patient_ID"].astype(str).str.zfill(7).astype("string")
                )

                data_frames.append(patient_df)

            print(f"\tLoaded volume data {volumes_data_path}.")
        self.volumes_data = pd.concat(data_frames, ignore_index=True)
        self.volumes_data["Date"] = pd.to_datetime(self.volumes_data["Date"], format="%Y-%m-%d")
        self.volumes_data = self.volumes_data.rename(
            columns={
                "Growth[%]": "Volume Change",
                "Normalized Growth[%]": "Normalized Volume Change",
            }
        )

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
        self.merged_data["Age Group"] = self.merged_data.apply(categorize_age_group, axis=1).astype(
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
            # "Normalized Volume Change",
        ]:
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

            data["Date First Treatment"] = first_treatment_date
            data["Received Treatment"] = received_treatment
            data["No Treatment"] = False if received_treatment else True

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

        self.pre_treatment_data["Treatment Type"] = self.pre_treatment_data[
            "Treatment Type"
        ].cat.remove_unused_categories()
        self.post_treatment_data["Treatment Type"] = self.post_treatment_data[
            "Treatment Type"
        ].cat.remove_unused_categories()

        self.pre_treatment_data["Received Treatment"] = self.pre_treatment_data[
            "Received Treatment"
        ].astype("category")
        self.post_treatment_data["Received Treatment"] = self.post_treatment_data[
            "Received Treatment"
        ].astype("category")

        for patient_id in self.pre_treatment_data["Patient_ID"].unique():
            if (
                self.pre_treatment_data[self.pre_treatment_data["Patient_ID"] == patient_id][
                    "Date First Treatment"
                ]
                .isna()
                .any()
            ):
                last_scan_date = self.pre_treatment_data[
                    self.pre_treatment_data["Patient_ID"] == patient_id
                ]["Date"].max()
                self.pre_treatment_data.loc[
                    self.pre_treatment_data["Patient_ID"] == patient_id, "Date First Treatment"
                ] = last_scan_date

        self.pre_treatment_data["Time to Treatment"] = (
            self.pre_treatment_data["Date First Treatment"]
            - self.pre_treatment_data["Date First Diagnosis"]
        ).dt.days

        self.post_treatment_data["Time to Treatment"] = (
            self.post_treatment_data["Date First Treatment"]
            - self.post_treatment_data["Date First Diagnosis"]
        ).dt.days

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

    def analyze_pre_treatment(self, prefix, output_dir):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, and race.
        """
        print("\tPre-treatment Correlations:")

        # variable types
        categorical_vars = [
            "Location",
            "Symptoms",
            "Treatment Type",
            "Age Group",
            "Sex",
            "Mutations",
            "Received Treatment",
            "Tumor Progression",
            "Tumor Classification",
            "Patient Classification",
        ]
        numerical_vars = [
            "Age",
            "Volume",
            "Normalized Volume",
            "Volume Change",
            "Normalized Volume CumMean",
            "Normalized Volume CumMedian",
            "Normalized Volume CumStd",
            "Normalized Volume RollMean",
            "Normalized Volume RollMedian",
            "Normalized Volume RollStd",
            "Volume Change CumMean",
            "Volume Change CumMedian",
            "Volume Change CumStd",
            "Volume Change RollMean",
            "Volume Change RollMedian",
            "Volume Change RollStd",
            "Time to Treatment",
            "Baseline Volume",
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
            filtered_vars = [
                var
                for var in numerical_vars
                if not var.startswith(("Volume Change ", "Volume ", "Normalized"))
            ]
            for other_num_var in filtered_vars:
                if other_num_var != num_var:
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

        aggregated_data = (
            self.pre_treatment_data.sort_values("Date").groupby("Patient_ID", as_index=False).last()
        )

        for cat_var in categorical_vars:
            for other_cat_var in categorical_vars:
                if cat_var != other_cat_var:
                    self.analyze_correlation(
                        cat_var,
                        other_cat_var,
                        aggregated_data,
                        prefix,
                        output_dir,
                    )

    def analyze_post_treatment(self, prefix, output_dir):
        """
        Analyze data for post-treatment cases. This involves finding correlations between
        variables such as treatment types, tumor volume changes, and specific mutations.
        """
        print("Post-treatment Correlations:")
        # TODO: treatment type to volume change, mutations to volume change, treatment type to XYZ

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
            "Age": "years",
            "Age Group": "years",
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
        if test_type == "correlation":
            sns.scatterplot(x=x_val, y=y_val, data=data)
            sns.regplot(x=x_val, y=y_val, data=data, scatter=False, color="blue")
            title += (
                f"{method.title()} correlation coefficient: {stat:.2f}, P-value:"
                f" {p_val:.3e} (N={num_patients})"
            )
        elif test_type == "t-test":
            # sns.barplot(x=x_val, y=y_val, data=data, ci="sd")
            # Calculate group means and standard deviations
            means = data.groupby(x_val)[y_val].mean()
            stds = data.groupby(x_val)[y_val].std()

            # Plotting
            _, ax = plt.subplots()
            means.plot(kind="bar", yerr=stds, capsize=4, ax=ax, color="skyblue", ecolor="black")
            title += f"T-statistic: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type == "point-biserial":
            sns.boxplot(x=x_val, y=y_val, data=data)
            title += (
                f"Point-Biserial Correlation Coefficient: {stat:.2f}, P-value:"
                f" {p_val:.3e} (N={num_patients})"
            )
        elif test_type == "ANOVA":
            sns.boxplot(x=x_val, y=y_val, data=data)
            plt.xticks(rotation=90, fontsize="small")
            title += f"F-statistic: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
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
        # Data preparation for modeling
        pre_treatment_data = self.pre_treatment_data.copy()
        pre_treatment_data.sort_values(by=["Patient_ID", "Date"], inplace=True)
        pre_treatment_data["Time since First Scan"] = pre_treatment_data.groupby("Patient_ID")[
            "Date"
        ].transform(lambda x: (x - x.min()).dt.days)
        self.pre_treatment_data.sort_values(by=["Patient_ID", "Date"], inplace=True)
        self.pre_treatment_data["Time since First Scan"] = pre_treatment_data[
            "Time since First Scan"
        ]

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
        normalized_volume_trajectories_plot = os.path.join(
            output_dir, f"{prefix}_normalized_volume_trajectories_plot.png"
        )
        plot_individual_trajectories(
            normalized_volume_trajectories_plot,
            plot_data=pre_treatment_data,
            column="Normalized Volume",
            unit="mm^3",
        )

        category_list = [
            "Sex",
            "Mutations",
            "Tumor Progression",
            "Received Treatment",
            "Location",
        ]
        for cat in category_list:
            cat_volume_change_name = os.path.join(
                output_dir, f"{prefix}_{cat}_volume_change_trajectories_plot.png"
            )
            plot_individual_trajectories(
                cat_volume_change_name,
                plot_data=pre_treatment_data,
                column="Volume Change",
                category_column=cat,
                unit="%",
            )
            cat_normalized_volume_name = os.path.join(
                output_dir, f"{prefix}_{cat}_normalized_volume_trajectories_plot.png"
            )
            plot_individual_trajectories(
                cat_normalized_volume_name,
                plot_data=pre_treatment_data,
                column="Normalized Volume",
                category_column=cat,
                unit="mm^3",
            )

        # Trend analysis and classifciation of patients
        self.trend_analysis(pre_treatment_data, output_dir, prefix)

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
            analysis_data_pre["Date First Progression"] < analysis_data_pre["Date First Treatment"]
        ]
        analysis_data_pre.loc[:, "Duration"] = (
            analysis_data_pre["Date First Progression"] - analysis_data_pre["Date First Diagnosis"]
        ).dt.days

        analysis_data_pre["Event_Occurred"] = ~analysis_data_pre["Date First Progression"].isna()

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
            output_dir, f"{prefix}_survival_plot_category_{stratify_by}.png"
        )
        plt.savefig(survival_plot, dpi=300)
        plt.close()
        print(f"\t\tSaved survival KaplanMeier curve for {stratify_by}.")

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
        # self.pre_treatment_data using the maps
        merged_data = pd.merge(
            self.pre_treatment_data,
            data[["Patient_ID", "Date", "Stability Index", "Tumor Classification"]],
            on=["Patient_ID", "Date"],
            how="left",
        )

        self.pre_treatment_data = merged_data
        visualize_tumor_stability(data, output_dir, stability_threshold, change_threshold)
        print("\t\tSaved tumor stability plots.")

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

        # Save to original dataframe
        classifications_series = pd.Series(patient_classifications)
        self.pre_treatment_data["Patient Classification"] = (
            self.pre_treatment_data["Patient_ID"].map(classifications_series).astype("category")
        )

        # Plots
        output_filename = os.path.join(output_dir, f"{prefix}_trend_analysis.png")
        plot_trend_trajectories(data, output_filename, column_name, unit="mm^3")
        print("\t\tSaved trend analysis plot.")

    def printout_stats(self):
        """
        Descriptive statistics.
        """
        # Baseline volume
        median_baseline_volume = self.pre_treatment_data["Baseline Volume"].median()
        max_baseline_volume = self.pre_treatment_data["Baseline Volume"].max()
        min_baseline_volume = self.pre_treatment_data["Baseline Volume"].min()
        print(f"\t\tMedian Baseline Volume: {median_baseline_volume} mm^3")
        print(f"\t\tMaximum Baseline Volume: {max_baseline_volume} mm^3")
        print(f"\t\tMinimum Baseline Volume: {min_baseline_volume} mm^3")

        # Age
        median_age = self.pre_treatment_data["Age"].median()
        max_age = self.pre_treatment_data["Age"].max()
        min_age = self.pre_treatment_data["Age"].min()
        print(f"\t\tMedian Age: {median_age} years")
        print(f"\t\tMaximum Age: {max_age} years")
        print(f"\t\tMinimum Age: {min_age} years")

        # Sex, Received Treatment, Progression, Symptoms,
        # Location, Patient Classification, Treatment Type
        copy_df = self.pre_treatment_data.copy()
        unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
        counts_sex = unique_pat["Sex"].value_counts()
        counts_progression = unique_pat["Tumor Progression"].value_counts()
        counts_received_treatment = unique_pat["Received Treatment"].value_counts()
        counts_symptoms = unique_pat["Symptoms"].value_counts()
        counts_location = unique_pat["Location"].value_counts()
        counts_patient_classification = unique_pat["Patient Classification"].value_counts()
        counts_treatment_type = unique_pat["Treatment Type"].value_counts()
        print(f"\t\tReceived Treatment: {counts_received_treatment}")
        print(f"\t\tSymptoms: {counts_symptoms}")
        print(f"\t\tLocation: {counts_location}")
        print(f"\t\tSex: {counts_sex}")
        print(f"\t\tProgression: {counts_progression}")
        print(f"\t\tPatient Classification: {counts_patient_classification}")
        print(f"\t\tTreatment Type: {counts_treatment_type}")

        # Volume Change
        median_volume_change = self.pre_treatment_data["Volume Change"].median()
        max_volume_change = self.pre_treatment_data["Volume Change"].max()
        min_volume_change = self.pre_treatment_data["Volume Change"].min()
        print(f"\t\tMedian Volume Change: {median_volume_change} %")
        print(f"\t\tMaximum Volume Change: {max_volume_change} %")
        print(f"\t\tMinimum Volume Change: {min_volume_change} %")

        # Nomralized Volume
        median_normalized_volume = self.pre_treatment_data["Normalized Volume"].median()
        max_normalized_volume = self.pre_treatment_data["Normalized Volume"].max()
        min_normalized_volume = self.pre_treatment_data["Normalized Volume"].min()
        print(f"\t\tMedian Normalized Volume: {median_normalized_volume} mm^3")
        print(f"\t\tMaximum Normalized Volume: {max_normalized_volume} mm^3")
        print(f"\t\tMinimum Normalized Volume: {min_normalized_volume} mm^3")

        # Follow-up time
        median_follow_up = self.pre_treatment_data["Follow-up Time"].median()
        max_follow_up = self.pre_treatment_data["Follow-up Time"].max()
        min_follow_up = self.pre_treatment_data["Follow-up Time"].min()
        print(f"\t\tMedian Follow-Up Time: {median_follow_up} days")
        print(f"\t\tMaximum Follow-Up Time: {max_follow_up} days")
        print(f"\t\tMinimum Follow-Up Time: {min_follow_up} days")

    def run_analysis(self, output_correlations, output_stats):
        """
        Run a comprehensive analysis pipeline consisting of data separation,
        sensitivity analysis, propensity score matching, main analysis, corrections
        for multiple comparisons, trend analysis, and feature engineering.

        This method orchestrates the overall workflow of the analysis process,
        including data preparation, statistical analysis, and results interpretation.
        """
        step_idx = 1

        if correlation_cfg.SEPARATION:
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

            # Total number of unique patient IDs in both pre-treatment and post-treatment
            # datasets matches the number in the original dataset. This ensures that no
            # patients were lost during the data separation process.
            unique_ids_original = set(self.merged_data["Patient_ID"].unique())
            unique_ids_pre = set(self.pre_treatment_data["Patient_ID"].unique())
            unique_ids_post = set(self.post_treatment_data["Patient_ID"].unique())

            if unique_ids_original != unique_ids_pre.union(unique_ids_post):
                print("\tError: Mismatch in patient IDs between original and separated datasets.")
            else:
                print("\tAll patient IDs are consistent.")
            # Date First Treatment is consistent across all records within each dataset.
            # It should be the same in both pre-treatment and post-treatment data for any
            # given patient.
            for patient_id in unique_ids_original:
                treatment_dates_pre = self.pre_treatment_data[
                    self.pre_treatment_data["Patient_ID"] == patient_id
                ]["Date First Treatment"].unique()

                treatment_dates_post = self.post_treatment_data[
                    self.post_treatment_data["Patient_ID"] == patient_id
                ]["Date First Treatment"].unique()

                # Check if there's more than one unique treatment date or
                # inconsistent dates between pre and post datasets
                if (
                    len(treatment_dates_pre) > 1
                    or len(treatment_dates_post) > 1
                    or (
                        len(treatment_dates_pre) == 1
                        and len(treatment_dates_post) == 1
                        and treatment_dates_pre[0] != treatment_dates_post[0]
                    )
                ):
                    print(f"\tError: Inconsistent treatment dates for patient {patient_id}")
            print("\tTreatment dates are consistent for all other patients.")

            # All dates in the pre-treatment data are indeed before Date First Treatment
            # and all dates in the post-treatment data are on or after this date.
            date_range_issues = False
            for patient_id in unique_ids_pre:
                first_treatment_date = self.extract_treatment_dates(patient_id)

                # Check if the treatment date is not set (NaN) and use the last scan
                # date in that case
                if pd.isna(first_treatment_date):
                    first_treatment_date = self.pre_treatment_data[
                        self.pre_treatment_data["Patient_ID"] == patient_id
                    ]["Date"].max()

                # Pre-treatment data should be before the first treatment date
                if not all(
                    self.pre_treatment_data[self.pre_treatment_data["Patient_ID"] == patient_id][
                        "Date"
                    ]
                    <= first_treatment_date
                ):
                    print(f"\tError: Pre-treatment date inconsistency for patient {patient_id}")
                    date_range_issues = True
            for patient_id in unique_ids_post:
                first_treatment_date = self.extract_treatment_dates(patient_id)

                # Check if the treatment date is not set (NaN) and
                # use the last scan date in that case
                if pd.isna(first_treatment_date):
                    first_treatment_date = self.post_treatment_data[
                        self.post_treatment_data["Patient_ID"] == patient_id
                    ]["Date"].max()

                # Post-treatment data should be on or after the first treatment date
                if not all(
                    self.post_treatment_data[self.post_treatment_data["Patient_ID"] == patient_id][
                        "Date"
                    ]
                    >= first_treatment_date
                ):
                    print(f"\tError: Post-treatment date inconsistency for patient {patient_id}")
                    date_range_issues = True
            if not date_range_issues:
                print("\tAll date ranges are consistent.")

            # Check for any missing or NaN values in critical columns after the separation process.
            for column in self.pre_treatment_data.columns:
                if column == "Date First Progression":
                    # Check for missing Date First Progression only if Tumor Progression is True
                    missing_progression_data = self.pre_treatment_data[
                        (self.pre_treatment_data["Tumor Progression"] is True)
                        & (self.pre_treatment_data["Date First Progression"].isna())
                    ]
                    if not missing_progression_data.empty:
                        print(f"\tError: Missing data in pre_treatment column - {column}")
                elif self.pre_treatment_data[column].isna().any():
                    print(f"\tError: Missing data in pre_treatment column - {column}")
            print("\tNo more missing data in pre_treatment columns.")
            for column in self.post_treatment_data.columns:
                if column == "Date First Progression":
                    # Group the conditions correctly
                    missing_progression_data = self.post_treatment_data[
                        (self.post_treatment_data["Tumor Progression"] is True)
                        & (self.post_treatment_data["Date First Progression"].isna())
                    ]
                    if not missing_progression_data.empty:
                        print(f"\tError: Missing data in post-treatment column - {column}")
                elif self.post_treatment_data[column].isna().any():
                    print(f"\tError: Missing data in post-treatment column - {column}")
            print("\tNo more missing data in post_treatment columns.")

            step_idx += 1

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
            print(self.pre_treatment_data.dtypes)
            for data in [self.pre_treatment_data]:
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

            # Survival analysis
            stratify_by_list = ["Location", "Sex", "Mutations", "Age Group", "Symptoms"]
            for element in stratify_by_list:
                self.time_to_event_analysis(prefix, output_dir=output_stats, stratify_by=element)

            # Growth trajectories & Trend analysis
            self.model_growth_trajectories(prefix, output_dir=output_stats)

            self.printout_stats()

            # Tumor stability
            self.analyze_tumor_stability(
                data=self.pre_treatment_data,
                output_dir=output_stats,
                volume_weight=correlation_cfg.VOLUME_WEIGHT,
                growth_weight=correlation_cfg.GROWTH_WEIGHT,
                change_threshold=correlation_cfg.CHANGE_THRESHOLD,
            )

            # Correlations between variables
            self.analyze_pre_treatment(
                prefix=prefix,
                output_dir=output_correlations,
            )

            # Last consistency check
            consistency_check(self.pre_treatment_data)

            step_idx += 1

        if correlation_cfg.ANALYSIS_POST_TREATMENT:
            prefix = "post-treatment"
            print(f"Step {step_idx}: Starting main analyses {prefix}...")
            # self.analyze_post_treatment(prefix=prefix)

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
            save_for_deep_learning(self.pre_treatment_data, output_stats, prefix="pre-treatment")
            # save_for_deep_learning(self.post_treatment_data, output_stats, prefix="post-treatment")
            step_idx += 1


if __name__ == "__main__":
    analysis_bch = TumorAnalysis(
        correlation_cfg.CLINICAL_CSV_BCH,
        [correlation_cfg.VOLUMES_CSV_BCH],
        cohort="BCH",
    )

    os.makedirs(correlation_cfg.OUTPUT_DIR_CORRELATIONS_BCH, exist_ok=True)
    os.makedirs(correlation_cfg.OUTPUT_DIR_STATS_BCH, exist_ok=True)

    analysis_bch.run_analysis(
        correlation_cfg.OUTPUT_DIR_CORRELATIONS_BCH, correlation_cfg.OUTPUT_DIR_STATS_BCH
    )
