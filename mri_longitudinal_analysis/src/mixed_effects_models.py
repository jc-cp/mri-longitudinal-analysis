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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix, 
    roc_curve,
    auc, 
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class ProgressionPrediction:
    """Class to run mixed-effects models on longitudinal data."""

    def __init__(self, mg_data):
        print("Initializing ProgressionPrediction...")
        self.merged_data = mg_data
        self.merged_data.rename(
            columns={
                "Normalized Volume": "Normalized_Volume",
                "Volume Change Rate": "Volume_Change_Rate",
                "Patient Classification Binary Composite": "Patient_Classification_Binary",
            },
            inplace=True,
        )
        self.models = {}
        self.formula = (
            "Patient_Classification_Binary ~ Age + Normalized_Volume + Volume_Change_Rate"
        )
        self.re_formula = " ~ Age + Normalized_Volume + Volume_Change_Rate"

        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None],
        }
        self.current_model_name = None

    ######################
    # DATA PREPROCESSING #
    ######################
    def prepare_data(self, variables, outcome_var, group_var):
        """
        Prepare the data for mixed-effects modeling.

        Args:
            variables (list): List of variable names to include as predictors.
            outcome_var (str): The name of the outcome variable.
            group_var (str): The name of the grouping variable, typically subject ID.

        Returns:
            pd.DataFrame: A DataFrame ready for mixed-effects modeling.
        """
        print("Preparing data...")
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

        data = data.rename(
            columns={
                "Normalized Volume": "Normalized_Volume",
                "Volume Change Rate": "Volume_Change_Rate",
                "Patient Classification Binary Composite": "Patient_Classification_Binary",
            }
        )
        X = data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        data[["Age", "Normalized_Volume", "Volume_Change_Rate"]] = X_scaled
        X = data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        # calculate_vif(X)

        # Optionally: filter out extreme values or outliers
        threshold = 3  # 0 = no filtering
        for var in ["Volume_Change_Rate"]:
            data = self.filter_outliers(data, var, threshold)

        # Returning the cleaned DataFrame
        return data

    def split_data(self, data, train_size=0.7, val_size=0.15, test_size=0.15):
        print("Splitting data into train, validation, and test sets...")
        unique_patients = data['Patient_ID'].unique()
        train_patients, temp_patients = train_test_split(unique_patients, test_size=(1-train_size), random_state=42)
        val_patients, test_patients = train_test_split(temp_patients, test_size=(test_size/(val_size+test_size)), random_state=42)

        train_data = data[data['Patient_ID'].isin(train_patients)]
        val_data = data[data['Patient_ID'].isin(val_patients)]
        test_data = data[data['Patient_ID'].isin(test_patients)]

        return train_data, val_data, test_data

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

    def prepare_sampled_data(self, train_data, outcome_var, id_var):
        """
        Data preparation for comparison of models.
        """
        print("Preparing sampled data...")
        X = train_data[['Age', 'Normalized_Volume', 'Volume_Change_Rate']]
        y = train_data[outcome_var]
        
        ros = RandomOverSampler(random_state=42)
        X_over, y_over = ros.fit_resample(X, y)
        train_data_oversampled = pd.DataFrame(X_over, columns=X.columns)
        train_data_oversampled[outcome_var] = y_over
        train_data_oversampled[id_var] = train_data[id_var].iloc[ros.sample_indices_].values

        rus = RandomUnderSampler(random_state=42)
        X_under, y_under = rus.fit_resample(X, y)
        train_data_undersampled = pd.DataFrame(X_under, columns=X.columns)
        train_data_undersampled[outcome_var] = y_under
        train_data_undersampled[id_var] = train_data[id_var].iloc[rus.sample_indices_].values

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        train_data_smote = pd.DataFrame(X_smote, columns=X.columns)
        train_data_smote[outcome_var] = y_smote
        train_data_smote[id_var] = np.random.choice(train_data[id_var].unique(), size=len(train_data_smote))

        return train_data_oversampled, train_data_undersampled, train_data_smote

    def cross_validate(self, train_data, outcome_var, id_var, n_splits=5):
        print("Starting cross-validation...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {model_name: [] for model_name in ["Vanilla", "Vanilla Weights", "Interaction_Age_Normalized_Volume", "Interaction_Age_Volume_Change_Rate", "Interaction_Normalized_Volume_Volume_Change_Rate", "Oversampling", "Undersampling", "SMOTE", "Random Forest", "RF Oversampling", "RF Undersampling", "RF SMOTE"]}

        for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
            print(f"Cross-validation fold {fold + 1}")
            cv_train, cv_val = train_data.iloc[train_index], train_data.iloc[val_index]
            
            cv_train_over, cv_train_under, cv_train_smote = self.prepare_sampled_data(cv_train, outcome_var, id_var)
            
            # Train models for this fold
            fold_models = self.train_models(cv_train, cv_train_over, cv_train_under, cv_train_smote, outcome_var)
            for model_name, model in fold_models.items():
                self.current_model_name = model_name
                _, f1_sc, _, _ = self.evaluate_model(model, cv_val, outcome_var)
                cv_scores[model_name].append(f1_sc)
        
        for model_name, scores in cv_scores.items():
            print(f"Average {model_name} CV score: {np.mean(scores)}")

        return cv_scores

    #################
    # MODEL FITTING #
    #################
    def evaluate_model(self, model, data, outcome_var, threshold=0.5):
        print(f"Evaluating model: {self.current_model_name}")
        X = data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        y_true = data[outcome_var]
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X)
            y_pred_proba = y_pred
            y_pred = (y_pred >= threshold).astype(int)

        weighted_score, f1_sc = self.custom_scorer(y_true, y_pred, y_pred_proba)
        print() 
        return weighted_score, f1_sc, y_pred, y_pred_proba

    def custom_scorer(self, y_true, y_pred, y_pred_proba):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")

        weighted_score = (accuracy + precision + recall + f1 + roc_auc) / 5
        print(f"Weighted Score: {weighted_score}")
        return weighted_score, f1

    def train_models(self, train_data, train_data_oversampled, train_data_undersampled, train_data_smote, outcome_var):
        print("Training models...")
        self.models = {}

        X_train = train_data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        y_train = train_data[outcome_var]

        # Vanilla Model
        model = smf.mixedlm(
            formula=self.formula,
            data=train_data,
            groups=train_data["Patient_ID"],
            re_formula=self.re_formula,
        )
        # pylint: disable=unexpected-keyword-arg
        self.models["Vanilla"] = model.fit(reml=False)

        # Weighted Vanilla Model
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        train_data['class_weights'] = y_train.map(class_weights_dict)
        self.models["Vanilla Weights"] = model.fit(reml=False, freq_weights=train_data['class_weights'])

        # Models with interaction terms
        interaction_terms = [
            ["Age", "Normalized_Volume"],
            ["Age", "Volume_Change_Rate"],
            ["Normalized_Volume", "Volume_Change_Rate"],
        ]
        for term1, term2 in interaction_terms:
            formula_with_interaction = f"{self.formula} + {term1}:{term2}"
            model_interaction = smf.mixedlm(
                formula=formula_with_interaction,
                data=train_data,
                groups=train_data["Patient_ID"],
                re_formula=self.re_formula,
            )
            self.models[f"Interaction_{term1}_{term2}"] = model_interaction.fit(reml=False)

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
            max_depth=5,
        )
        rf_model.fit(
            X_train, y_train
        )
        self.models["Random Forest"] = rf_model

        # Random Forest with oversampling
        rf_over = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=None,
        )
        rf_over.fit(
            train_data_oversampled[["Age", "Normalized_Volume", "Volume_Change_Rate"]], 
            train_data_oversampled[outcome_var]
        )
        self.models["RF Oversampling"] = rf_over

        # Random Forest with undersampling
        rf_under = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=None,
        )
        rf_under.fit(
            train_data_undersampled[["Age", "Normalized_Volume", "Volume_Change_Rate"]], 
            train_data_undersampled[outcome_var]
        )
        self.models["RF Undersampling"] = rf_under

        # Random Forest with SMOTE
        rf_smote = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=None,
        )
        rf_smote.fit(
            train_data_smote[["Age", "Normalized_Volume", "Volume_Change_Rate"]], 
            train_data_smote[outcome_var]
        )
        self.models["RF SMOTE"] = rf_smote

        return self.models
    
    def grid_search(self, train_data, outcome_var):
        """
        Grid search for Random Forest model.
        """
        print("Performing grid search for Random Forest...")
        X = train_data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        y = train_data[outcome_var]
        # Create the Random Forest model
        rf_model = RandomForestClassifier(random_state=42)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=self.param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)

        # Get the best model and parameters
        best_rf_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("\nBest Random Forest Parameters:")
        print(best_params)

        return best_rf_model

    def optimize_threshold(self, model, X, y):
        print("Optimizing classification threshold...")
        thresholds = np.arange(0.1, 1.0, 0.1)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            if hasattr(model, 'predict_proba'):
                y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
            else:
                y_pred = (model.predict(X) >= threshold).astype(int)
            f1 = f1_score(y, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Best classification threshold: {best_threshold}")
        return best_threshold
    
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

    def plot_confusion_matrix_and_roc(self, y_true, y_pred, y_pred_proba, output_dir):
        """
        Confusion matrix and ROC curve for the model evaluation.
        """
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontdict={'size': 20})
        plt.ylabel('True label', fontdict={'size': 15})
        plt.xlabel('Predicted label', fontdict={'size': 15})

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontdict={'size': 15})
        plt.ylabel('True Positive Rate', fontdict={'size': 15})
        plt.title('Receiver Operating Characteristic Curve', fontdict={'size': 18})
        plt.legend(loc="lower right")

        plt.tight_layout()
        file = os.path.join(output_dir, 'confusion_matrix_and_roc.png')
        plt.savefig(file, dpi=300)
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, output_dir):
        """
        Plot the precision-recall curve.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure()
        plt.plot(recall, precision, color='b', label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall', fontdict={'size': 15})
        plt.ylabel('Precision', fontdict={'size': 15})
        plt.title('Precision-Recall Curve', fontdict={'size': 20})
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()

    #########
    # MAINs #
    #########
    def main(self, output_dir):
        """Main function to run mixed-effects models on longitudinal data."""
        # variables and dirs
        longitudinal_vars = [
            "Age",
            "Normalized_Volume",
            "Volume_Change_Rate",
        ]
        outcome_var = "Patient_Classification_Binary"
        grouping_var = "Patient_ID"
        mixed_model_dir = os.path.join(output_dir, "mixed_model")
        individual_model_dir = os.path.join(output_dir, "individual_model")
        os.makedirs(mixed_model_dir, exist_ok=True)
        os.makedirs(individual_model_dir, exist_ok=True)

        # data preparation
        data = self.prepare_data(
            variables=longitudinal_vars,
            outcome_var=outcome_var,
            group_var=grouping_var,
        )

        train_data, val_data, test_data = self.split_data(data)
        # Perform cross-validation to select best model type
        cv_scores = self.cross_validate(train_data, outcome_var, grouping_var)
        best_model_type = max(cv_scores, key=lambda x: np.mean(cv_scores[x]))
        print(f"Best model type from cross-validation: {best_model_type}")

        # Train best model on full training data
        print(f"Training best model ({best_model_type}) on full training data...")
        train_data_oversampled, train_data_undersampled, train_data_smote = self.prepare_sampled_data(train_data, outcome_var, grouping_var)
        models = self.train_models(
            train_data,
            train_data_oversampled,
            train_data_undersampled,
            train_data_smote,
            outcome_var
        )
        best_model = models[best_model_type]
                
        if isinstance(best_model, RandomForestClassifier):
            print("Random Forest Classifier is the best model. Performing grid search.")
            #best_model = self.grid_search(train_data, outcome_var)
            best_model_type = "Optimized Random Forest"
            models[best_model_type] = best_model
            
        if not isinstance(best_model, RandomForestClassifier):
            print("Mixed-effects model is the best model. Performing diagnostics and visualizations of effects.")
            self.perform_model_diagnostics(None, best_model, mixed_model_dir)
            for var in longitudinal_vars:
                self.visualize_mixed_model_effects(data, best_model, var, mixed_model_dir)

        X_train = train_data[["Age", "Normalized_Volume", "Volume_Change_Rate"]]
        y_train = train_data[outcome_var]
        best_threshold = self.optimize_threshold(best_model, X_train, y_train)
        self.current_model_name = best_model_type

        # Evaluate on validation set
        weighted_val_score, f1_val_score, _, _ = self.evaluate_model(best_model, val_data, outcome_var, best_threshold)
        print(f"F1 Validation score for {best_model_type}: {f1_val_score}")
        print(f"Weighted Validation score for {best_model_type}: {weighted_val_score}")


        # Final evaluation on test set
        weighted_test_score, f1_test_score, y_pred, y_pred_proba = self.evaluate_model(best_model, test_data, outcome_var, best_threshold)
        print(f"F1 test score for {best_model_type}: {f1_test_score}")
        print(f"Weighted test score for {best_model_type}: {weighted_test_score}")

        # finally do a grid search if needed and plot the confusion matrix and ROC curve
        #_ = self.grid_search(train_data, outcome_var)
        self.plot_confusion_matrix_and_roc(test_data[outcome_var], y_pred, y_pred_proba, mixed_model_dir)
        self.plot_precision_recall_curve(test_data[outcome_var], y_pred_proba, mixed_model_dir)
        
        
        # Save model summary if it's a mixed-effects model
        if not isinstance(best_model, RandomForestClassifier):
            summary_file = os.path.join(mixed_model_dir, f"{best_model_type}_summary.txt")
            with open(summary_file, "w", encoding='utf-8') as f:
                f.write(best_model.summary().as_text())

        # Print feature importances if it's a Random Forest model
        if isinstance(best_model, RandomForestClassifier):
            importances = best_model.feature_importances_
            feature_names = ['Age', 'Normalized_Volume', 'Volume_Change_Rate']
            for feature, importance in zip(feature_names, importances):
                print(f"Importance of {feature}: {importance}")

        print(f"Results saved in {mixed_model_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Example usage of the MixedEffectsModel class
    CSV_FILE_PATH = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats_joint/pre-treatment_dl_features.csv"
    OUTPUT_DIR = (
        "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_plots_joint"
    )
    merged_data = pd.read_csv(CSV_FILE_PATH)
    mixed_model = ProgressionPrediction(merged_data)
    mixed_model.main(OUTPUT_DIR)
