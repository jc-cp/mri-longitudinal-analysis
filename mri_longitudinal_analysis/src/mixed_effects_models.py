"""
Script to run mixed-effects models on longitudinal data.
"""
# from utils.helper_functions import calculate_vif
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class MixedEffectsModel:
    """Class to run mixed-effects models on longitudinal data."""

    def __init__(self, mg_data):
        self.merged_data = mg_data
        self.merged_data.rename(
            columns={
                "Volume Change": "Volume_Change",
                "Volume Change Rate": "Volume_Change_Rate",
                "Days Between Scans": "Days_Between_Scans",
                "Follow-Up Time": "Follow_Up_Time",
                "Patient Classification Binary": "Patient_Classification_Binary",
            },
            inplace=True,
        )
        self.models = {}
        self.formula = (
            "Patient_Classification_Binary ~ Age + Volume + Volume_Change_Rate"
        )
        self.re_formula = " ~ Age + Volume + Volume_Change_Rate"

        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    ######################
    # DATA PREPROCESSING #
    ######################
    def prepare_data_for_mixed_model(self, variables, outcome_var, group_var):
        """
        Prepare the data for mixed-effects modeling.

        Args:
            variables (list): List of variable names to include as predictors.
            outcome_var (str): The name of the outcome variable.
            group_var (str): The name of the grouping variable, typically subject ID.

        Returns:
            pd.DataFrame: A DataFrame ready for mixed-effects modeling.
        """
        # Filter out necessary columns and drop rows where any of these are NaN
        data = self.merged_data[variables + [outcome_var, group_var]].dropna()

        # Optionally, implement more sophisticated imputation if necessary
        # Here's a simple imputation example replacing NaNs with the median of the column
        for var in variables:
            if data[var].isna().any():
                median_value = data[var].median()
                data[var].fillna(median_value, inplace=True)

        # Check and ensure no NaN values are present in the outcome or group variables
        if data[outcome_var].isna().any() or data[group_var].isna().any():
            raise ValueError(
                "Outcome or grouping variable cannot contain NaN values after preparation."
            )

        # data['Patient_ID'] = data['Patient_ID'].astype('category')
        data = data.rename(
            columns={
                "Volume Change": "Volume_Change",
                "Volume Change Rate": "Volume_Change_Rate",
                "Days Between Scans": "Days_Between_Scans",
                "Follow-Up Time": "Follow_Up_Time",
                "Patient Classification Binary": "Patient_Classification_Binary",
            }
        )
        data["Age_scaled"] = (data["Age"] - data["Age"].mean()) / data["Age"].std()
        data["Volume_scaled"] = (data["Volume"] - data["Volume"].mean()) / data[
            "Volume"
        ].std()
        data["Volume_Change_scaled"] = (
            data["Volume_Change"] - data["Volume_Change"].mean()
        ) / data["Volume_Change"].std()
        data["Volume_Change_Rate_scaled"] = (
            data["Volume_Change_Rate"] - data["Volume_Change_Rate"].mean()
        ) / data["Volume_Change_Rate"].std()

        X = data[["Age", "Volume", "Volume_Change", "Volume_Change_Rate"]]
        # calculate_vif(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        data[["Age", "Volume", "Volume_Change", "Volume_Change_Rate"]] = X_scaled
        X = data[["Age", "Volume", "Volume_Change", "Volume_Change_Rate"]]
        # calculate_vif(X)

        # Optionally: filter out extreme values or outliers
        threshold = 3  # 0 = no filtering
        for var in ["Volume_Change", "Volume_Change_Rate"]:
            data = self.filter_outliers(data, var, threshold)

        # Returning the cleaned DataFrame
        return data

    def filter_outliers(self, data, variable, threshold):
        """
        Basic outlier filtering based on mean and standard deviation.
        """
        if threshold == 0:
            return data
        mean = data[variable].mean()
        std = data[variable].std()
        data_filtered = data[
            (data[variable] >= mean - threshold * std)
            & (data[variable] <= mean + threshold * std)
        ]
        return data_filtered

    def preprocess_data_for_training(self, data, outcome_var):
        """
        Data preparation for comparison of models.
        """
        train_data, test_data = train_test_split(
            data, test_size=0.1, random_state=42, stratify=data[outcome_var]
        )
        train_data_index = train_data.index

        # Oversampling using Random Oversampling (ROS)
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(
            train_data[["Age", "Volume", "Volume_Change_Rate"]], train_data[outcome_var]
        )
        train_data_oversampled = pd.DataFrame(
            X_train_resampled, columns=["Age", "Volume", "Volume_Change_Rate"]
        )
        train_data_oversampled["Patient_Classification_Binary"] = y_train_resampled
        train_data_oversampled.index = train_data_index[ros.sample_indices_]
        train_data_oversampled["Patient_ID"] = train_data.loc[
            train_data_oversampled.index, "Patient_ID"
        ].values

        # Undersampling using Random Undersampling (RUS)
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(
            train_data[["Age", "Volume", "Volume_Change_Rate"]], train_data[outcome_var]
        )
        train_data_undersampled = pd.DataFrame(
            X_train_resampled, columns=["Age", "Volume", "Volume_Change_Rate"]
        )
        train_data_undersampled["Patient_Classification_Binary"] = y_train_resampled
        train_data_undersampled.index = train_data_index[rus.sample_indices_]
        train_data_undersampled["Patient_ID"] = train_data.loc[
            train_data_undersampled.index, "Patient_ID"
        ].values

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            train_data[["Age", "Volume", "Volume_Change_Rate"]], train_data[outcome_var]
        )
        train_data_smote = pd.DataFrame(
            X_train_resampled, columns=["Age", "Volume", "Volume_Change_Rate"]
        )
        train_data_smote["Patient_Classification_Binary"] = y_train_resampled
        train_data_smote["Patient_ID"] = np.random.choice(
            train_data["Patient_ID"].unique(), size=len(train_data_smote)
        )

        return (
            train_data,
            test_data,
            train_data_oversampled,
            train_data_undersampled,
            train_data_smote,
        )

    #################
    # MODEL FITTING #
    #################
    def run_mixed_effects_model(
        self, data, grouping_var, longi_vars, individual_model_dir, mixed_model_dir
    ):
        """
        Run mixed-effects logistic regression on longitudinal variables.
        """
        print("\t\tRunning mixed-effects logistic regression model:")
        # Individual models for each variable
        individual_models = {}
        for var in longi_vars:
            formula = f"Patient_Classification_Binary ~ {var}"
            model = smf.mixedlm(
                formula, data=data, groups=data[grouping_var], re_formula=f"~{var}"
            )
            try:
                ind_result = model.fit()
                individual_models[var] = ind_result
                print(f"\t\t\tIndividual model for {var} fitted succesfully.")
                ind_summary_file = os.path.join(
                    individual_model_dir, f"{var}_summary.txt"
                )
                with open(ind_summary_file, "w", encoding="utf-8") as file:
                    file.write(ind_result.summary().as_text())
            except ExceptionGroup as e:
                print(f"Error fitting individual model for {var}: {e}")

        # Combined model with interaction term if needed
        # no grouping variable need in the formula since we have a single level of grouping
        formula = self.formula  # + Volume_Change"
        model = smf.mixedlm(
            formula,
            data=data,
            groups=data[grouping_var],
            re_formula=self.re_formula  # + Volume_Change",
            # re_formula = '~1'
        )
        # pylint: disable=unexpected-keyword-arg
        combined_result = model.fit(reml=False)
        summary_file = os.path.join(mixed_model_dir, "combined_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as file:
            file.write(combined_result.summary().as_text())
        print("\t\t\tCombined model fitted successfully.")

        # Best model selection based on interaction terms
        # add an interaction term by multiplying the two variables instead of adding them
        interaction_terms = [
            ["Age", "Volume"],
            ["Age", "Volume_Change"],
            ["Age", "Volume_Change_Rate"],
            ["Volume", "Volume_Change"],
            ["Volume", "Volume_Change_Rate"],
            ["Volume_Change", "Volume_Change_Rate"],
        ]

        all_models = self.interaction_models(
            data, grouping_var, longi_vars, interaction_terms
        )
        for name, model in all_models.items():
            model_summary_file = os.path.join(
                mixed_model_dir, f"{str(name)}_summary.txt"
            )
            with open(model_summary_file, "w", encoding="utf-8") as file:
                file.write(all_models[name].summary().as_text())

        all_models["combined"] = combined_result
        return individual_models, combined_result, all_models

    def interaction_models(self, data, grouping_var, variables, interaction_terms=None):
        """
        Automatically fit mixed-effects models for each variable and interaction term.
        """
        models = {}
        formula_og = "Patient_Classification_Binary ~ " + " + ".join(variables)
        if interaction_terms:
            for interaction in interaction_terms:
                formula = formula_og + " + " + " * ".join(interaction)
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data[grouping_var],
                    # re_formula=f'~{" + ".join(interaction)}')
                    # re_formula='~1')
                    re_formula="~ Age + Volume + Volume_Change + Volume_Change_Rate",
                )
                # pylint: disable=unexpected-keyword-arg
                model_fit = model.fit(reml=False)
                models[" * ".join(interaction)] = model_fit
        return models

    def compare_models(self, models):
        """
        Function to compare models based on AIC and BIC.
        """
        best_model = None
        best_aic = float("inf")
        best_bic = float("inf")
        best_name = None
        for name, model in models.items():
            if model is None:
                continue
            if not model.converged:
                continue

            aic = model.aic
            bic = model.bic
            print(f"Model: {name}")
            print(f"\tAIC: {aic}")
            print(f"\tBIC: {bic}")

            if np.isnan(model.params).any() or np.isinf(model.params).any():
                print(
                    f"Skipping model '{name}' due to NaN or infinite parameter estimates."
                )
                continue

            if (model.pvalues < 0.05).any():
                if aic < best_aic:
                    best_model = model
                    best_aic = aic
                    best_bic = bic
                    best_name = name
                elif aic == best_aic and bic < best_bic:
                    best_model = model
                    best_bic = bic
                    best_name = name

        if best_model is not None:
            print(f"Best model: {best_name}")
            print(f"\tAIC: {best_aic}")
            print(f"\tBIC: {best_bic}")
            print(best_model.summary())
            return best_model
        else:
            print("No best model found.")
            return None

    def train_models(
        self,
        train_data,
        train_data_oversampled,
        train_data_undersampled,
        train_data_smote,
        outcome_var,
        trained_model,
    ):
        """
        Train and fit models on the data.
        """
        # Vanilla Model
        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(train_data[outcome_var]),
            y=train_data[outcome_var],
        )
        class_weights_dict = dict(enumerate(class_weights))
        train_data["class_weights"] = train_data[outcome_var].map(class_weights_dict)
        model = smf.mixedlm(
            formula=self.formula,
            data=train_data,
            groups=train_data["Patient_ID"],
            re_formula=self.re_formula,
        )
        # pylint: disable=unexpected-keyword-arg
        self.models["Vanilla"] = model.fit(reml=False)
        self.models["Vanilla Weights"] = model.fit(
            reml=False, freq_weights=train_data["class_weights"]
        )

        # Pretrained Model
        self.models["Pretrained"] = trained_model

        # Model with oversampling
        model_oversampling = smf.mixedlm(
            formula=self.formula,
            data=train_data_oversampled,
            groups=train_data_oversampled["Patient_ID"],
            re_formula=self.re_formula,
        )
        self.models["Oversampling"] = model_oversampling.fit(reml=False)

        # Model with undersampling
        model_undersampling = smf.mixedlm(
            formula=self.formula,
            data=train_data_undersampled,
            groups=train_data_undersampled["Patient_ID"],
            re_formula=self.re_formula,
        )
        self.models["Undersampling"] = model_undersampling.fit(reml=False)

        # Model with SMOTE
        model_smote = smf.mixedlm(
            formula=self.formula,
            data=train_data_smote,
            groups=train_data_smote["Patient_ID"],
            re_formula=self.re_formula,
        )
        self.models["SMOTE"] = model_smote.fit(reml=False)

        # Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=None,
        )
        rf_model.fit(
            train_data[["Age", "Volume", "Volume_Change_Rate"]], train_data[outcome_var]
        )
        self.models["Random Forest"] = rf_model

        rf_model_smote = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=None,
        )
        rf_model_smote.fit(
            train_data_smote[["Age", "Volume", "Volume_Change_Rate"]],
            train_data_smote[outcome_var],
        )
        self.models["RF SMOTE"] = rf_model_smote

    def evaluate_models(self, test_data, outcome_var, debug=False):
        """
        Evaluation loop for the models.
        """
        best_threshold = None
        best_model_threshold = None
        best_model_score = 0
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for model_name, model in self.models.items():
            if model_name == "Random Forest" or model_name == "RF SMOTE":
                test_data["pred_prob"] = model.predict_proba(
                    test_data[["Age", "Volume", "Volume_Change_Rate"]]
                )[:, 1]
            else:
                test_data["pred_prob"] = model.predict(test_data)
            best_threshold = None
            best_score = 0
            for threshold in thresholds:
                test_data[f"pred_class_{threshold}"] = np.where(
                    test_data["pred_prob"] >= threshold, 1, 0
                )
                if debug:
                    print(f"Model: {model_name}")
                    print(f"Threshold: {threshold}")
                    print(
                        classification_report(
                            test_data[outcome_var], test_data[f"pred_class_{threshold}"]
                        )
                    )
                    print()
                score = f1_score(
                    test_data[outcome_var], test_data[f"pred_class_{threshold}"]
                )
                score_name = "F1 Score"
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            print(f"Model: {model_name}")
            print(f"Best Threshold: {best_threshold}")
            print(f"Score: {score_name}")
            print(f"Best score: {best_score}")
            print()

            if best_score > best_model_score:
                best_model = model_name
                best_model_score = best_score
                best_model_threshold = best_threshold

        print("Best Model Overall:")
        print(f"Model: {best_model}")
        print(f"Best Threshold: {best_model_threshold}")
        print(f"Best {score_name}-score: {best_model_score}")

        # Simple Evaluation at given threshold for the best model
        if best_model == "Random Forest" or best_model == "RF SMOTE":
            test_data["pred_prob"] = self.models[best_model].predict_proba(
                test_data[["Age", "Volume", "Volume_Change_Rate"]]
            )[:, 1]
        else:
            test_data["pred_prob"] = self.models[best_model].predict(
                test_data[["Age", "Volume", "Volume_Change_Rate"]]
            )
        test_data["pred_class"] = np.where(
            test_data["pred_prob"] >= best_model_threshold, 1, 0
        )
        _ = self.custom_scorer(test_data[outcome_var], test_data["pred_class"])

        return best_model

    def standard_scorer(self, y_true, y_pred):
        """Standard scoring metrics for the model evaluation."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print()
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        return accuracy, precision, recall, f1

    def custom_scorer(self, y_true, y_pred):
        """
        Custom scorer for the model evaluation.
        """
        accuracy, precision, recall, f1 = self.standard_scorer(y_true, y_pred)

        # Adjust the weights based on your preference
        weighted_score = (accuracy + precision + recall + f1) / 4
        print(f"Weighted Score: {weighted_score}")
        return weighted_score

    def grid_search(self, train_data, outcome_var):
        """
        Grid search for Random Forest model.
        """
        # Create the Random Forest model
        rf_model = RandomForestClassifier(random_state=42)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=self.param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
        )
        grid_search.fit(
            train_data[["Age", "Volume", "Volume_Change_Rate"]], train_data[outcome_var]
        )

        # Get the best model and parameters
        best_rf_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("\nBest Random Forest Parameters:")
        print(best_params)

        return best_rf_model

    #################
    # VISUALIZATION #
    #################
    def visualize_mixed_model_effects(self, data, model_result, variable, output_dir):
        """
        Visualize the distribution of the variable, fixed effects, and random effects.
        """
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Distribution of the variable
        sns.histplot(data[variable], kde=True, ax=ax1, color="gray")
        ax1.set_title(f"Distribution of {variable}")
        ax1.set_ylabel("Density")
        # x1.set_yscale("log")

        # Fixed effects (coefficients and CIs)
        coef = model_result.fe_params[variable]
        ci_lower, ci_upper = model_result.conf_int().loc[variable]
        ax2.errorbar(
            x=0,
            y=coef,
            xerr=[[coef - ci_lower], [ci_upper - coef]],
            fmt="o",
            color="blue",
            label="Fixed Effect",
        )
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_title("Fixed Effect Coefficient")
        ax2.set_xlabel("Coefficient")
        ax2.set_ylabel("")
        ax2.legend()
        ax2.set_xlim(ci_lower - 0.5, ci_upper + 0.5)

        # Random effects plot
        re = model_result.random_effects
        random_effects_vals = [re[group][variable] for group in re]
        ax3.scatter(
            random_effects_vals,
            np.arange(len(random_effects_vals)),
            color="green",
            alpha=0.5,
        )
        ax3.axvline(0, color="red", linestyle="--")
        ax3.set_title(f"Random Effects Distribution for {variable}")
        ax3.set_xlabel("Random Effect Value")
        ax3.set_ylabel("Group Index")

        plt.tight_layout()
        file_name = f"{variable}_effect_size_distribution.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
        plt.close()
        print(f"\t\t\tVisualizations saved for {variable}.")

    def perform_model_diagnostics(self, var, model_result, output_dir):
        """
        Diagnostic plots for the mixed-effects model.
        """
        if var is None:
            var = "combined"
        # Residual plot, it is used for the checking the linearity assumption
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        residuals = model_result.resid
        ax1.scatter(model_result.fittedvalues, residuals)
        ax1.axhline(0, color="red", linestyle="--")
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residual Plot")

        # QQ plot of residuals, it is used for checking the normality assumption
        sm.qqplot(residuals, line="s", ax=ax2)
        ax2.set_title("QQ Plot of Residuals")

        # Scale-location plot: asseses homoscedasticity asssumption
        ax3.scatter(model_result.fittedvalues, np.sqrt(np.abs(model_result.resid)))
        ax3.set_xlabel("Fitted Values")
        ax3.set_ylabel("Square Root of Absolute Residuals")
        ax3.set_title("Scale-Location Plot")
        infl_file = os.path.join(output_dir, f"{var}_diagnostics.png")
        plt.savefig(infl_file, dpi=300)

    #########
    # MAINs #
    #########
    def main(self, output_dir):
        """Main function to run mixed-effects models on longitudinal data."""
        # variables and dirs
        longitudinal_vars = [
            "Age",
            "Volume",
            "Volume_Change",
            "Volume_Change_Rate",
        ]
        outcome_var = "Patient_Classification_Binary"
        grouping_var = "Patient_ID"
        mixed_model_dir = os.path.join(output_dir, "mixed_model")
        individual_model_dir = os.path.join(output_dir, "individual_model")
        os.makedirs(mixed_model_dir, exist_ok=True)
        os.makedirs(individual_model_dir, exist_ok=True)

        # data preparation
        data = self.prepare_data_for_mixed_model(
            variables=longitudinal_vars,
            outcome_var=outcome_var,
            group_var=grouping_var,
        )

        # runs individual models for each variable and a combined model
        individual_models, combined_model, all_models = self.run_mixed_effects_model(
            data, grouping_var, longitudinal_vars, individual_model_dir, mixed_model_dir
        )

        # runs individual diagnostic plots for each variable and a combined model
        for var, ind_result in individual_models.items():
            self.perform_model_diagnostics(var, ind_result, individual_model_dir)
        self.perform_model_diagnostics("combined", combined_model, mixed_model_dir)

        # visualize the effects of each variable in the combined model
        for var in longitudinal_vars:
            if var == "Volume_Change":
                continue
            self.visualize_mixed_model_effects(
                data, combined_model, var, individual_model_dir
            )

        # compare the mixed-effects models and select the best one
        best_mixed_effects_model = self.compare_models(all_models)

        # train the best model on different data sets
        (
            train_data,
            test_data,
            train_data_oversampled,
            train_data_undersampled,
            train_data_smote,
        ) = self.preprocess_data_for_training(data, outcome_var)
        self.train_models(
            train_data,
            train_data_oversampled,
            train_data_undersampled,
            train_data_smote,
            outcome_var,
            best_mixed_effects_model,
        )
        _ = self.evaluate_models(test_data, outcome_var)

        # finally do a grid search
        _ = self.grid_search(train_data, outcome_var)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Example usage of the MixedEffectsModel class
    CSV_FILE_PATH = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats_joint/pre-treatment_dl_features.csv"
    OUTPUT_DIR = (
        "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_plots_joint"
    )
    merged_data = pd.read_csv(CSV_FILE_PATH)
    mixed_model = MixedEffectsModel(merged_data)
    mixed_model.main(OUTPUT_DIR)
