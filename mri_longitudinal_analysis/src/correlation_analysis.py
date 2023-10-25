"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations and treatments.
"""
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
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
        if not os.path.exists(clinical_data_path) or not os.path.exists(volumes_data_path):
            raise FileNotFoundError("One or both of the specified CSV files could not be found.")

        self.clinical_data = pd.read_csv(clinical_data_path)
        self.clinical_data["BCH MRN"] = self.clinical_data["BCH MRN"].astype(str).str.zfill(7)
        self.parse_clinical_data()

        self.load_volumes_data(volumes_data_path)
        self.merge_data()

    def parse_clinical_data(self):
        diagnosis_to_glioma_type = {
            "plain low grade glioma": "Low Grade",
            "astrocytoma": "Astrocytoma",
            "optic gliona": "Optic Gliona",
            "tectal glioma": "Tectal",
            "ganglioglioma": "Ganglioglioma",
            "glioneuronal neoplasm": "Glioneuronal",
            "DNET": "DNET",
            "other": "Other",
        }

        def map_diagnosis(diagnosis):
            for keyword, glioma_type in diagnosis_to_glioma_type.items():
                if keyword.lower() in diagnosis.lower():
                    return glioma_type
            return "Unknown"

        self.clinical_data["Glioma_Type"] = self.clinical_data["Pathologic diagnosis"].apply(
            map_diagnosis
        )
        self.clinical_data["Sex"] = self.clinical_data["Sex"].astype("category").cat.codes
        self.clinical_data["Race"] = (
            self.clinical_data["Race/Ethnicity"].astype("category").cat.codes
        )
        self.clinical_data["Mutations"] = (
            self.clinical_data["BRAF V600E mutation"] + self.clinical_data["BRAF fusion"]
        )
        print("Got clinical data.")

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
        print("Got volume data.")

    def merge_data(self):
        print("Columns in clinical_data: ", self.clinical_data.columns)
        print("Columns in volumes_data: ", self.volumes_data.columns)

        self.merged_data = pd.merge(
            self.clinical_data,
            self.volumes_data,
            left_on=["BCH MRN"],
            right_on=["Patient_ID"],
            how="right",
        )
        # TODO: Fix this, "Treatment_Type" is not defined, "Tumor_volume" has to be properly extracted!

        # self.pre_treatment_data = self.merged_data[self.merged_data["Treatment_Type"].isna()]
        # self.post_treatment_data = self.merged_data[self.merged_data["Treatment_Type"].notna()]

    def analyze_correlation(self, x_val, y_val, data, method="pearson"):
        """
        Analyze correlation between two variables.

        Parameters:
            var1, var2 (str): Column names for the variables to correlate.
            method (str): The correlation method to use ('pearson' or 'spearman').

        Returns:
            float: The correlation coefficient.
        """
        if method == "pearson":
            coef, p_val = pearson_correlation(data[x_val].dropna(), data[y_val].dropna())
        elif method == "spearman":
            coef, p_val = spearman_correlation(data[x_val].dropna(), data[y_val].dropna())

        print(f"{x_val} and {y_val} - Coefficient: {coef}, P-value: {p_val}")

        sns.scatterplot(x=x_val, y=y_val, data=data)
        plt.title(f"{x_val} vs {y_val} ({method.capitalize()} correlation)")
        plt.show()

    def analyze_pre_treatment(self):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such as initial tumor volume, age, sex, mutations, and race.

        Returns:
            dict: Dictionary of correlation results.
        """ ""
        print("Pre-treatment Correlations:")
        self.analyze_correlation(
            "Glioma_Type", "Tumor_Volume", self.pre_treatment_data, method="spearman"
        )
        for var in ["Age", "Sex", "Mutations", "Race"]:
            self.analyze_correlation(
                var, "Initial_Tumor_Volume", self.pre_treatment_data, method="spearman"
            )

        unchanging_tumors = self.pre_treatment_data[
            self.pre_treatment_data["Tumor_Volume_Change"] == 0
        ]
        print(f"Tumors with no change in volume: {unchanging_tumors}")

    def analyze_post_treatment(self):
        """
        Analyze data for post-treatment cases. This involves finding correlations between
        variables such as treatment types, tumor volume changes, and specific mutations.

        Returns:
            dict: Dictionary of correlation results.
        """
        print("Post-treatment Correlations:")
        self.analyze_correlation(
            "Treatment_Type", "Tumor_Volume_Change", self.post_treatment_data, method="spearman"
        )
        self.analyze_correlation(
            "Mutation_Type", "Tumor_Volume_Change", self.post_treatment_data, method="spearman"
        )
        self.analyze_correlation(
            "Mutation_Type", "Treatment_Response", self.post_treatment_data, method="spearman"
        )

        # Chi-Squared test

        chi2, p_val = chi_squared_test(
            self.post_treatment_data["Mutation_Type"], self.post_treatment_data["Treatment_Type"]
        )
        print(
            f"Chi-Squared test between Mutation_Type and Treatment_Type: Chi2: {chi2}, P-value:"
            f" {p_val}"
        )


if __name__ == "__main__":
    analysis = TumorAnalysis(correlation_cfg.CLINICAL_CSV, correlation_cfg.VOLUMES_CSV)
    analysis.analyze_pre_treatment()
    analysis.analyze_post_treatment()

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
