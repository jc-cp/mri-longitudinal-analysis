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
    calculate_stats,
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
        missing_files = [
            path
            for path in [clinical_data_path, volumes_data_paths[0], volumes_data_paths[1]]
            if not os.path.exists(path)
        ]
        if missing_files:
            raise FileNotFoundError(f"The following files could not be found: {missing_files}")
        print("\tValidated files.")

    def load_clinical_data(self, clinical_data_path):
        self.clinical_data = pd.read_csv(clinical_data_path)
        self.clinical_data["Treatment_Type"] = self.extract_treatment_types()
        self.clinical_data["Treatment_Type"] = self.clinical_data["Treatment_Type"].astype(
            "category"
        )

        self.clinical_data["BCH MRN"] = zero_fill(self.clinical_data["BCH MRN"], 7)
        self.parse_clinical_data()

    def parse_clinical_data(self):
        diagnosis_to_glioma_type = {
            "astrocytoma": "Astrocytoma",
            "optic": "Optic Gliona",
            "tectal": "Tectal Glioma",
            "ganglioglioma": "Ganglioglioma",
            "glioneuronal neoplasm": "Glioneuronal Neoplasm",
            "DNET": "DNET",
            "low grade glioma": "Plain Low Grade Glioma",
            "other": "Other",
        }

        def map_diagnosis(diagnosis):
            for keyword, glioma_type in diagnosis_to_glioma_type.items():
                if keyword.lower() in diagnosis.lower():
                    return glioma_type
            return "Other"

        self.clinical_data["Glioma_Type"] = (
            self.clinical_data["Pathologic diagnosis"].apply(map_diagnosis).astype("category")
        )
        self.clinical_data["Sex"] = (
            self.clinical_data["Sex"].apply(lambda x: "Female" if x == "Female" else "Male")
        ).astype("category")
        self.clinical_data["Race"] = self.clinical_data["Race/Ethnicity"].astype("category")
        self.clinical_data["Mutations"] = self.clinical_data.apply(
            lambda row: "Yes"
            if row["BRAF V600E mutation"] == "Yes"
            or row["BRAF fusion"] == "Yes"
            or row["FGFR fusion"] == "Yes"
            else "No",
            axis=1,
        ).astype("category")
        self.clinical_data["Date_of_Diagnosis"] = pd.to_datetime(
            self.clinical_data["Date of MRI diagnosis"], dayfirst=True
        )
        self.clinical_data["Date_of_First_Progression"] = pd.to_datetime(
            self.clinical_data["Date of First Progression"], dayfirst=True
        )
        self.clinical_data["Tumor_Progression"] = (
            self.clinical_data["Progression"].map({"Yes": True, "No": False, None: False})
        ).astype(bool)

        relevant_columns = [
            "Treatment_Type",
            "BCH MRN",
            "Glioma_Type",
            "Sex",
            "Race",
            "Mutations",
            "Date_of_First_Progression",
            "Date_of_Diagnosis",
            "Tumor_Progression",
        ]

        self.clinical_data_reduced = self.clinical_data[relevant_columns]
        print("\tParsed clinical data.")

    def load_volumes_data(self, volumes_data_paths):
        data_frames = []
        for volumes_data_path in volumes_data_paths:
            all_files = [f for f in os.listdir(volumes_data_path) if f.endswith(".csv")]
            for file in all_files:
                patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
                patient_id = file.split(".")[0]
                patient_df["Patient_ID"] = patient_id
                patient_df["Patient_ID"] = patient_df["Patient_ID"].astype(str).str.zfill(7)
                patient_df["Date"] = pd.to_datetime(
                    patient_df["Date"], errors="coerce", format="%Y-%m-%d"
                )
                data_frames.append(patient_df)

            print(f"\tLoaded volume data {volumes_data_path}.")
        self.volumes_data = pd.concat(data_frames, ignore_index=True)

    def extract_treatment_types(self):
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
        self.clinical_data_reduced.loc[:, "BCH MRN"] = self.clinical_data_reduced["BCH MRN"].astype(
            str
        )
        grouped_volume_data = self.volumes_data.groupby("Patient_ID").agg(list).reset_index()
        self.merged_data = pd.merge(
            self.clinical_data_reduced,
            grouped_volume_data,
            left_on=["BCH MRN"],
            right_on=["Patient_ID"],
            how="right",
        )
        self.merged_data = self.merged_data.drop(columns=["BCH MRN"])
        print("\tMerged data.")

    def aggregate_summary_statistics(self):
        for column in ["Growth[%]", "Age", "Volume"]:
            self.merged_data[
                [f"{column}_mean", f"{column}_median", f"{column}_std"]
            ] = self.merged_data.apply(lambda row: calculate_stats(row, column), axis=1)
        print("\tAggregated summary statistics.")

    def longitudinal_separation(self):
        pre_treatment_data_frames = []
        post_treatment_data_frames = []

        for patient_id, data in self.merged_data.groupby("Patient_ID"):
            treatment_dates = self.extract_treatment_dates(patient_id)

            pre_treatment_df, post_treatment_df = self.perform_separation(data, treatment_dates)
            pre_treatment_data_frames.append(pre_treatment_df)
            post_treatment_data_frames.append(post_treatment_df)

        # Concatenate and convert to appropriate data types
        self.pre_treatment_data = pd.concat(pre_treatment_data_frames, ignore_index=True)
        self.pre_treatment_data["Sex"] = self.pre_treatment_data["Sex"].astype("category")
        self.pre_treatment_data["Glioma_Type"] = self.pre_treatment_data["Glioma_Type"].astype(
            "category"
        )
        self.pre_treatment_data["Race"] = self.pre_treatment_data["Race"].astype("category")
        self.pre_treatment_data["Mutations"] = self.pre_treatment_data["Mutations"].astype(
            "category"
        )
        self.pre_treatment_data["Treatment_Type"] = self.pre_treatment_data[
            "Treatment_Type"
        ].astype("category")

        self.pre_treatment_data["Patient_ID"] = self.pre_treatment_data["Patient_ID"].astype(int)
        self.pre_treatment_data["Tumor_Progression"] = self.pre_treatment_data[
            "Tumor_Progression"
        ].astype(bool)

        # TODO: concatenate and convert to proper data types
        self.post_treatment_data = pd.concat(post_treatment_data_frames, ignore_index=True)

    def extract_treatment_dates(self, patient_id):
        first_row = self.clinical_data[self.clinical_data["BCH MRN"] == patient_id].iloc[0]

        treatment_dates = {}

        if first_row["Surgical Resection"] == "Yes":
            treatment_dates["Surgery"] = first_row["Date of first surgery"]

        if first_row["Systemic therapy before radiation"] == "Yes":
            treatment_dates["Chemotherapy"] = first_row["Date of Systemic Therapy Start"]

        if first_row["Radiation as part of initial treatment"] == "Yes":
            treatment_dates["Radiation"] = first_row["Start Date of Radiation"]

        # print(f"Patient {patient_id} - Treatment Dates: {treatment_dates}")
        return treatment_dates

    def perform_separation(self, data, treatment_dates):
        treatment_dates = {
            k: pd.to_datetime(v, errors="coerce", dayfirst=True)
            for k, v in treatment_dates.items()
            if pd.notnull(v)
        }

        first_treatment_date = min(treatment_dates.values(), default=pd.Timestamp.max)

        pre_treatment_rows = {col: [] for col in data.columns}
        post_treatment_rows = {col: [] for col in data.columns}

        pre_treatment_rows["First_Treatment_Date"] = first_treatment_date
        post_treatment_rows["First_Treatment_Date"] = first_treatment_date

        # Iterate over each row in the data for the patient
        for _, row in data.iterrows():
            dates = row["Date"]
            # Ensure that 'dates' is a list before iterating
            if isinstance(dates, list):
                for i, date_str in enumerate(dates):
                    date = pd.to_datetime(date_str, errors="coerce")
                    if pd.notnull(date):
                        # Split each element based on the treatment date
                        for col in data.columns:
                            # Check if the column contains a list and only then try to index it
                            if isinstance(row[col], list):
                                value_to_append = row[col][i]
                            else:
                                value_to_append = row[col]
                            if date < first_treatment_date:
                                pre_treatment_rows[col].append(value_to_append)
                            else:
                                post_treatment_rows[col].append(value_to_append)
            else:
                # If 'dates' is not a list, handle the scalar case here
                date = pd.to_datetime(dates, errors="coerce")
                for col in data.columns:
                    if pd.notnull(date) and date < first_treatment_date:
                        pre_treatment_rows[col].append(row[col])
                    else:
                        post_treatment_rows[col].append(row[col])

        if first_treatment_date != pd.Timestamp.max:
            pre_treatment_rows["Received_Treatment"] = ["Yes"] * len(
                pre_treatment_rows["Patient_ID"]
            )
            post_treatment_rows["Received_Treatment"] = ["Yes"] * len(
                post_treatment_rows["Patient_ID"]
            )
        else:
            pre_treatment_rows["Received_Treatment"] = ["No"] * len(
                pre_treatment_rows["Patient_ID"]
            )
            post_treatment_rows["Received_Treatment"] = ["No"] * len(
                post_treatment_rows["Patient_ID"]
            )
        # Convert the lists of series to DataFrames
        pre_treatment_df = pd.DataFrame(pre_treatment_rows)
        post_treatment_df = pd.DataFrame(post_treatment_rows)

        return pre_treatment_df, post_treatment_df

    def get_treatment_type(self, patient_row):
        if pd.notna(patient_row.get("Date of first surgery")):
            return "Surgery"
        elif patient_row.get("Systemic therapy before radiation") == "Yes":
            return "Chemotherapy"
        elif patient_row.get("Radiation as part of initial treatment") == "Yes":
            return "Radiation"
        else:
            return "Unknown"

    def analyze_correlation(self, x_val, y_val, data, prefix, method="spearman"):
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

        Returns:
            dict: Dictionary of correlation results.
        """ ""
        print("\tPre-treatment Correlations:")

        for growth_metric in ["Growth[%]", "Growth[%]_mean", "Growth[%]_std"]:
            self.analyze_correlation(
                "Glioma_Type",
                growth_metric,
                self.pre_treatment_data,
                prefix,
                method=correlation_method,
            )

        for var in ["Sex", "Mutations", "Race"]:
            for corr_method in ["t-test", "point-biserial"]:
                self.analyze_correlation(
                    var, "Growth[%]", self.pre_treatment_data, prefix, method=corr_method
                )

        for age_metric in ["Age_mean", "Age_median", "Age_std"]:
            self.analyze_correlation(
                age_metric, "Growth[%]", self.pre_treatment_data, prefix, method=correlation_method
            )

        # unchanging_tumors = self.pre_treatment_data[self.pre_treatment_data["Volume_mean"] == 0]
        # print(f"Tumors with no change in volume: {len(unchanging_tumors)}")

    def analyze_post_treatment(self, prefix, correlation_method="spearman"):
        """
        Analyze data for post-treatment cases. This involves finding correlations between
        variables such as treatment types, tumor volume changes, and specific mutations.

        Returns:
            dict: Dictionary of correlation results.
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
            test_type (str): The type of statistical test ('correlation', 't-test', 'anova', 'chi-squared').
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

    def determine_plot_type(self, data, x_var, y_var):
        """
        Determine the type of plot to use based on the data types of x_var and y_var.

        Parameters:
            data (DataFrame): The dataframe containing the data.
            x_var (str): The name of the x variable.
            y_var (str): The name of the y variable.

        Returns:
            str: The type of plot to use ('scatter', 'box', 'violin', etc.).
        """
        x_dtype = data[x_var].dtype
        y_dtype = data[y_var].dtype

        # Define logic to determine plot type
        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            return "scatter"
        elif pd.api.types.is_categorical_dtype(x_dtype) or pd.api.types.is_object_dtype(x_dtype):
            return "box"  # You can change to 'violin' or other types as preferred
        else:
            # More conditions can be added for different scenarios
            return "unknown"

    def plot_individual_trajectories(self, result, name):
        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot a line for each individual
        # unique_ids = self.pre_treatment_data["Patient_ID"].unique()
        # for patient_id in unique_ids:
        #     patient_data = self.pre_treatment_data[
        #         self.pre_treatment_data["Patient_ID"] == patient_id
        #     ]
        #     plt.plot(
        #         patient_data["Time_since_First_Scan"],
        #         patient_data["Growth_pct"],
        #         marker="o",
        #         label=f"Patient {patient_id}",
        #     )

        # Using seaborn's lineplot to create a line for each patient
        sns.lineplot(
            x="Time_since_First_Scan",
            y="Growth_pct",
            hue="Patient_ID",
            data=self.pre_treatment_data,
            legend=None,  # Set to 'brief' or 'full' if you want the legend
        )
        plt.xlabel("Days Since First Scan")
        plt.ylabel("Tumor Growth Percentage")
        plt.title("Individual Growth Trajectories")
        plt.legend()
        plt.savefig(name)
        plt.close()

    def time_to_event_analysis(self, prefix):
        print("\tAnalyzing time to event:")
        pre_treatment_data = self.pre_treatment_data.loc[
            self.pre_treatment_data["Date_of_First_Progression"]
            < self.pre_treatment_data["First_Treatment_Date"]
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

    def model_growth_trajectories(self):
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
        # if self.pre_treatment_data.groupby("Patient_ID").size().min() < 3:
        #     print("Not enough data points per patient to fit a mixed-effects model.")
        #     return None

        model = sm.MixedLM.from_formula(
            "Growth_pct ~ Time_since_First_Scan",
            re_formula="~Time_since_First_Scan",
            groups=self.pre_treatment_data["Patient_ID"],
            data=self.pre_treatment_data,
        )
        result = model.fit()
        print(result.summary())

        if not result.converged:
            print("Model did not converge, try simplifying the model or check the data.")
            return None
        else:
            growth_trajecotr_plot = os.path.join(
                correlation_cfg.OUTPUT_DIR, "growth_trajecotr_plot.png"
            )
            self.plot_individual_trajectories(result, growth_trajecotr_plot)
            print("Saved growth trajectories plot.")

    def analyze_time_to_treatment_effect(self, prefix):
        print("\tAnalyzing time to treatment effect:")
        self.pre_treatment_data["Time_to_Treatment"] = (
            self.pre_treatment_data["First_Treatment_Date"]
            - self.pre_treatment_data["Date_of_Diagnosis"]
        ).dt.days

        filtered_data = self.pre_treatment_data.dropna(subset=["Time_to_Treatment", "Growth[%]"])
        filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Time_to_Treatment", "Growth[%]"]
        )
        self.analyze_correlation(
            "Time_to_Treatment", "Growth[%]", filtered_data, prefix, method="pearson"
        )

        model = sm.OLS(
            filtered_data["Growth[%]"], sm.add_constant(filtered_data["Time_to_Treatment"])
        )
        results = model.fit()
        # print(results.summary())
        # TODO: rethink about this fitting

    def run_analysis(self):
        print("Step 1: Separating data into pre- and post-treatment dataframes...")
        self.longitudinal_separation()
        assert self.pre_treatment_data.columns.all() == self.post_treatment_data.columns.all()
        print("\tData separated, assertation for same columns in separated dataframes done.")

        if correlation_cfg.SENSITIVITY:
            print("Step 2: Performing Sensitivity Analysis...")

            pre_treatment_vars = ["Volume", "Growth[%]", "Growth[%]_mean", "Growth[%]_std"]
            post_treatment_vars = []

            for pre_var in pre_treatment_vars:
                print(f"\tPerforming sensitivity analysis on pre-treatmen variables {pre_var}...")
                self.pre_treatment_data = sensitivity_analysis(
                    self.pre_treatment_data, pre_var, z_threshold=1.5
                )

            # TODO: concatenate and convert to proper data types
            # print(self.post_treatment_data.dtypes)
            # for post_var in post_treatment_vars:
            #     self.post_treatment_data = sensitivity_analysis(
            #         self.post_treatment_data, post_var, z_threshold=2
            #     )

            #     print(
            #         "   Data after excluding outliers based on Z-score in post-treatment setting:"
            #         f" {self.post_treatment_data}"
            #     )

        if correlation_cfg.PROPENSITY:
            print("Step 3: Performing Propensity Score Matching...")

            for data in [self.pre_treatment_data]:  # self.post_treatment_data]:
                treatment_column = "Received_Treatment"
                covariate_columns = ["Age", "Volume", "Growth[%]"]

                data[treatment_column] = data[treatment_column].map({"Yes": 1, "No": 0})
                propensity_scores = calculate_propensity_scores(
                    data, treatment_column, covariate_columns
                )
                matched_data = perform_propensity_score_matching(
                    data, propensity_scores, treatment_column, caliper=0.05
                )
                smd_results = check_balance(matched_data, covariate_columns, treatment_column)

                visualize_smds(smd_results, path=correlation_cfg.OUTPUT_DIR)

        if correlation_cfg.ANLYSIS:
            print("Step 4: Starting main analyses...")

            prefix = "pre-treatment"
            self.analyze_pre_treatment(
                correlation_method=correlation_cfg.CORRELATION_PRE_TREATMENT, prefix=prefix
            )
            self.time_to_event_analysis(prefix)
            self.analyze_time_to_treatment_effect(prefix)
            # self.model_growth_trajectories()

            # prefix = "post-treatment"
            # self.analyze_post_treatment(correlation_method=correlation_cfg.CORRELATION_POST_TREATMENT, prefix=prefix)

        if correlation_cfg.CORRECTION:
            print("Step 5: Starting Corrections...")
            print("\tBonferroni Corrections: ")
            path = correlation_cfg.OUTPUT_DIR
            alpha = correlation_cfg.CORRECTION_ALPHA

            corrected_p_values_bonf = bonferroni_correction(self.p_values, alpha=alpha)
            visualize_p_value_bonferroni_corrections(
                self.p_values, corrected_p_values_bonf, alpha, path
            )

            corrected_p_values_fdr, is_rejected = fdr_correction(self.p_values, alpha=alpha)
            visualize_fdr_correction(
                self.p_values, corrected_p_values_fdr, is_rejected, alpha, path
            )

        if correlation_cfg.FEATURE_ENG:
            print("Step 6: Starting Feature Engineering...")
            # self.feature_engineering()


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
