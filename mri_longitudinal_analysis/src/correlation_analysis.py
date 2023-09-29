"""
This script initializes the TumorAnalysis class with clinical and volumetric data, 
then performs various analyses including correlations and treatments.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        self.clinical_data = pd.read_csv(clinical_data_path)
        self.volumes_data = pd.read_csv(volumes_data_path)
        self.merged_data = pd.merge(
            self.clinical_data, self.volumes_data, on=["Patient_ID", "Timepoint"]
        )

        self.pre_treatment_data = self.merged_data[self.merged_data["Treatment_Type"].isna()]
        self.post_treatment_data = self.merged_data[self.merged_data["Treatment_Type"].notna()]

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
            coef, p_val = pearson_correlation(data[x].dropna(), data[y].dropna())
        elif method == "spearman":
            coef, p_val = spearman_correlation(data[x].dropna(), data[y].dropna())

        print(f"{x_val} and {y_val} - Coefficient: {coef}, P-value: {p_val}")

        sns.scatterplot(x=x, y=y, data=data)
        plt.title(f"{x_val} vs {y_val} ({method.capitalize()} correlation)")
        plt.show()

    def analyze_pre_treatment(self):
        """
        Analyze data for pre-treatment cases. This involves finding correlations
        between variables such asinitial tumor volume, age, sex, mutations, and race.

        Returns:
            dict: Dictionary of correlation results.
        """
        print("Pre-treatment Correlations:")
        self.analyze_correlation("Symptoms_Severity", "Tumor_Volume", self.pre_treatment_data)
        self.analyze_correlation(
            "Glioma_Type", "Tumor_Volume", self.pre_treatment_data, method="spearman"
        )
        self.analyze_correlation("Age", "Initial_Tumor_Volume", self.pre_treatment_data)
        self.analyze_correlation(
            "Sex", "Initial_Tumor_Volume", self.pre_treatment_data, method="spearman"
        )

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
    analysis = TumorAnalysis("clinical_data.csv", "volumes_data.csv")
    analysis.analyze_pre_treatment()
    analysis.analyze_post_treatment()

    # Multiple Comparisons Correction Example
    # p_values = [0.01, 0.05, 0.1]
    # corrected_p_values = bonferroni_correction(p_values)
    # print(f"Corrected P-values using Bonferroni: {corrected_p_values}")

    # Sensitivity Analysis Example
    # filtered_data = sensitivity_analysis("Tumor_Volume", z_threshold=2)
    # print(f"Data after excluding outliers based on Z-score: {filtered_data}")

    # Propensity Score Matching Example
    # matched_data = propensity_score_matching(
    #    "Treatment_Type", ["Age", "Sex", "Mutation_Type"]
    # )
    # print(f"Data after Propensity Score Matching: {matched_data}")
