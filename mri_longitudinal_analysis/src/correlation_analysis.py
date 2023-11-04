"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations and treatments.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from cfg import correlation_cfg
from utils.helper_functions import (
    pearson_correlation,
    spearman_correlation,
    chi_squared_test,
    bonferroni_correction,
    sensitivity_analysis,
    propensity_score_matching,
)


class TumorAnalysis:
    """
    A class to perform tumor analysis using clinical and volumetric data.
    """

    def __init__(self, clinical_data_path, volumes_data_path):
        """
        Initialize the TumorAnalysis class.

        Parameters:
            clinical_data_file (str): Path to the clinical data CSV file.
            volumes_data_file (str): Path to the tumor volumes data CSV file.
        """
        self.merged_data = pd.DataFrame()
        self.post_treatment_data = pd.DataFrame()
        self.pre_treatment_data = pd.DataFrame()

        print("Validating files...")
        self.validate_files(clinical_data_path, volumes_data_path)
        self.load_clinical_data(clinical_data_path)
        self.load_volumes_data(volumes_data_path)
        self.merge_data()
        self.aggregate_summary_statistics()

    def validate_files(self, clinical_data_path, volumes_data_path):
        missing_files = [
            path for path in [clinical_data_path, volumes_data_path] if not os.path.exists(path)
        ]
        if missing_files:
            raise FileNotFoundError(f"The following files could not be found: {missing_files}")
        print("Validated files.")

    def load_clinical_data(self, clinical_data_path):
        self.clinical_data = pd.read_csv(clinical_data_path)
        self.clinical_data["Treatment_Type"] = self.extract_treatment_types()
        self.clinical_data["BCH MRN"] = self.zero_fill(self.clinical_data["BCH MRN"], 7)
        self.parse_clinical_data()

    @staticmethod
    def zero_fill(series, width):
        return series.astype(str).str.zfill(width)

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

        self.clinical_data["Glioma_Type"] = self.clinical_data["Pathologic diagnosis"].apply(
            map_diagnosis
        )
        self.clinical_data["Sex"] = self.clinical_data["Sex"].apply(
            lambda x: "Female" if x == "Female" else "Male"
        )
        self.clinical_data["Race"] = self.clinical_data["Race/Ethnicity"].astype(str)
        self.clinical_data["Mutations"] = self.clinical_data.apply(
            lambda row: "Yes"
            if row["BRAF V600E mutation"] or row["BRAF fusion"] == "Yes"
            else "No",
            axis=1,
        )

        relevant_columns = [
            "Treatment_Type",
            "BCH MRN",
            "Glioma_Type",
            "Sex",
            "Race",
            "Mutations",
        ]
        self.clinical_data_reduced = self.clinical_data[relevant_columns]
        print("Parsed clinical data.")

    def load_volumes_data(self, volumes_data_path):
        all_files = [f for f in os.listdir(volumes_data_path) if f.endswith(".csv")]
        data_frames = []
        for file in all_files:
            patient_df = pd.read_csv(os.path.join(volumes_data_path, file))
            patient_id = file.split(".")[0]  # Assuming the ID is the first part of the filename
            patient_df["Patient_ID"] = patient_id
            patient_df["Patient_ID"] = patient_df["Patient_ID"].astype(str).str.zfill(7)
            data_frames.append(patient_df)

        self.volumes_data = pd.concat(data_frames, ignore_index=True)

        print("Loaded volume data.")

    def extract_treatment_types(self):
        treatment_list = []

        for index, row in self.clinical_data.iterrows():
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
        print("Merged data.")

    @staticmethod
    def calculate_stats(row, col_name):
        values = row[col_name]
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        return pd.Series(
            [mean_val, median_val, std_val],
            index=[f"{col_name}_mean", f"{col_name}_median", f"{col_name}_std"],
        )

    def aggregate_summary_statistics(self):
        for column in ["Growth[%]", "Age", "Volume"]:
            self.merged_data[
                [f"{column}_mean", f"{column}_median", f"{column}_std"]
            ] = self.merged_data.apply(lambda row: self.calculate_stats(row, column), axis=1)

        # print(self.merged_data.columns)
        # print(self.merged_data.head())
        print("Aggregated summary statistics.")

    def analyze_correlation(self, x_val, y_val, data, method="pearson"):
        """
        Analyze correlation between two variables.

        Parameters:
            var1, var2 (str): Column names for the variables to correlate.
            method (str): The correlation method to use ('pearson' or 'spearman').

        Returns:
            float: The correlation coefficient.
        """
        # Debugging: Print available columns and first few rows of data
        print("Available columns:", data.columns)
        print("First few rows of data:\n", data.head())

        # Drop NaN values for the specific variables you're interested in
        x_data_clean = data[x_val].dropna()
        y_data_clean = data[y_val].dropna()

        # Debugging: Print data after dropping NaN
        print(f"After dropping NaN for {x_val}: {x_data_clean}")
        print(f"After dropping NaN for {y_val}: {y_data_clean}")

        if method == "pearson":
            coef, p_val = pearson_correlation(data[x_val].dropna(), data[y_val].dropna())
        elif method == "spearman":
            coef, p_val = spearman_correlation(data[x_val].dropna(), data[y_val].dropna())

        print(f"{x_val} and {y_val} - Coefficient: {coef}, P-value: {p_val}")

        sns.scatterplot(x=x_val, y=y_val, data=data)
        plt.title(f"{x_val} vs {y_val} ({method.capitalize()} correlation)")
        plt.show()

    def analyze_pre_treatment(self, correlation_method="spearman"):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, and race.

        Returns:
            dict: Dictionary of correlation results.
        """ ""
        print("Pre-treatment Correlations:")

        # print(self.pre_treatment_data["Glioma_Type"].describe())
        # print(self.pre_treatment_data["mean"].describe())

        for metric in ["mean", "median", "std"]:
            self.analyze_correlation(
                "Glioma_Type", metric, self.pre_treatment_data, method=correlation_method
            )
            for var in ["Age", "Sex", "Mutations", "Race"]:
                self.analyze_correlation(
                    var, metric, self.pre_treatment_data, method=correlation_method
                )

        unchanging_tumors = self.pre_treatment_data[
            self.pre_treatment_data["Mean_Tumor_Volume"] == 0
        ]
        print(f"Tumors with no change in volume: {len(unchanging_tumors)}")

    def analyze_post_treatment(self, correlation_method="spearman"):
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
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Tumor_Volume_Change",
            self.post_treatment_data,
            method=correlation_method,
        )
        self.analyze_correlation(
            "Mutation_Type",
            "Treatment_Response",
            self.post_treatment_data,
            method=correlation_method,
        )

        # Chi-Squared test

        chi2, p_val = chi_squared_test(
            self.post_treatment_data["Mutation_Type"], self.post_treatment_data["Treatment_Type"]
        )
        print(
            f"Chi-Squared test between Mutation_Type and Treatment_Type: Chi2: {chi2}, P-value:"
            f" {p_val}"
        )

    def longitudinal_separation(self):
        pre_treatment_data_frames = []
        post_treatment_data_frames = []

        for patient_id, data in self.merged_data.groupby("Patient_ID"):
            treatment_dates = self.extract_treatment_dates(patient_id)

            pre_treatment_df, post_treatment_df = self.perform_separation(data, treatment_dates)
            # print(pre_treatment_df)
            # print(post_treatment_df)
            pre_treatment_data_frames.append(pre_treatment_df)
            post_treatment_data_frames.append(post_treatment_df)

        self.pre_treatment_data = pd.concat(pre_treatment_data_frames, ignore_index=True)
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

        print(f"Patient {patient_id} - Treatment Dates: {treatment_dates}")
        return treatment_dates

    def perform_separation(self, data, treatment_dates):
        treatment_dates = {
            k: pd.to_datetime(v, errors="coerce")
            for k, v in treatment_dates.items()
            if pd.notnull(v)
        }

        first_treatment_date = min(treatment_dates.values(), default=pd.Timestamp.max)

        pre_treatment_rows = {col: [] for col in data.columns}
        post_treatment_rows = {col: [] for col in data.columns}

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
                # This might happen if there's only one date and therefore one set of measurements
                date = pd.to_datetime(dates, errors="coerce")
                for col in data.columns:
                    if pd.notnull(date) and date < first_treatment_date:
                        pre_treatment_rows[col].append(row[col])
                    else:
                        post_treatment_rows[col].append(row[col])

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

    def extract_trends(self, data):
        trends = {}

        if data.empty:
            return trends

        # For demonstration, assume we are interested in a time series of tumor_volume
        time_series = data.sort_values("Date")["tumor_volume"]

        # Polynomial Curve Fitting (2nd degree here)
        x = np.linspace(0, len(time_series) - 1, len(time_series))
        y = time_series.values
        coefficients = np.polyfit(x, y, 2)
        poly = np.poly1d(coefficients)

        # Store the polynomial coefficients as features
        trends["poly_coef"] = coefficients

        # Find peaks and valleys
        peaks, _ = find_peaks(time_series)
        valleys, _ = find_peaks(-1 * time_series)

        # Store number of peaks and valleys as features
        trends["num_peaks"] = len(peaks)
        trends["num_valleys"] = len(valleys)

        # Calculate the mean distance between peaks as a feature (if feasible)
        if len(peaks) > 1:
            trends["mean_peak_distance"] = np.mean(np.diff(peaks))

        # Calculate the mean distance between valleys as a feature (if feasible)
        if len(valleys) > 1:
            trends["mean_valley_distance"] = np.mean(np.diff(valleys))

        return trends

    def feature_engineering(self):
        self.merged_data["Tumor_Volume_Change"] = (
            self.merged_data["Tumor_Volume_End"] - self.merged_data["Tumor_Volume_Start"]
        )
        self.merged_data["Treatment_Response"] = self.merged_data["Tumor_Volume_Change"].apply(
            lambda x: "Positive" if x < 0 else "Negative"
        )

    def temporal_pattern_recognition(self):
        # Assuming self.volumes_data is a time series data
        # Apply time-series clustering or RNN for pattern recognition
        pass

    def run_analysis(self):
        """
        Wrapper function to run all analyses. Init the class with clinical and volumetric data, and merge into one big dataframe.
        Then separate data into pre- and post-treatment groups, then perform separate analyses.
        """
        print("Separating data into pre- and post-treatment groups...")
        self.longitudinal_separation()
        # print(self.pre_treatment_data.head())
        # print(self.post_treatment_data.head())
        # TODO: Add the treatment date to the DF as an additional column for later plotting

        # self.analyze_pre_treatment(correlation_method=correlation_cfg.CORRELATION_PRE_TREATMENT)
        # self.analyze_post_treatment(correlation_method=correlation_cfg.CORRELATION_POST_TREATMENT)
        # self.feature_engineering()

        # Multiple Comparisons Correction Example
        if correlation_cfg.CORRECTION:
            # Adjust this test data
            p_values = [0.01, 0.05, 0.1]
            corrected_p_values = bonferroni_correction(p_values)
            print(f"Corrected P-values using Bonferroni: {corrected_p_values}")

        # Sensitivity Analysis Example
        if correlation_cfg.SENSITIVITY:
            # filtered_data = sensitivity_analysis("Tumor_Volume", z_threshold=2)
            # print(f"Data after excluding outliers based on Z-score: {filtered_data}")
            pass

        # Propensity Score Matching Example
        if correlation_cfg.PROPENSITY:
            # matched_data = propensity_score_matching("Treatment_Type", ["Age", "Sex", "Mutation_Type"])
            # print(f"Data after Propensity Score Matching: {matched_data}")
            pass


if __name__ == "__main__":
    analysis = TumorAnalysis(correlation_cfg.CLINICAL_CSV, correlation_cfg.VOLUMES_CSV)
    analysis.run_analysis()
