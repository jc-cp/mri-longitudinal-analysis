"""
Script with univariate and multivariate logistic regression analysis and correlation analysis.
"""
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from utils.helper_functions import logistic_regression_analysis,calculate_vif, check_assumptions
from scipy import stats

from cfg.src import lr_and_correlations_cfg

class CorrelationAnalysis:
    def __init__(self, data, cohort):
        self.merged_data = data
        self.cohort = cohort
        self.results = {}
 
    def analyze_correlation(self, x_val, y_val, data, prefix, output_dir, test_type):
        """
        Perform and print the results of a statistical test to analyze the correlation
        between two variables.

        Parameters:
        - x_val (str): The name of the first variable.
        - y_val (str): The name of the second variable.
        - data (DataFrame): The data containing the variables.
        - prefix (str): The prefix to be used for naming visualizations.
        - test_type (str): The statistical method to be used.

        Updates the class attributes with the results of the test and prints the outcome.
        """
        test_result, coef, p_val = None, None, None  # Initialize test_result

        # Convert data types
        try:
            if pd.api.types.is_numeric_dtype(data[x_val]):
                data[x_val] = data[x_val].astype('float64')
            elif not isinstance(data[x_val].dtype, pd.CategoricalDtype):
                data[x_val] = data[x_val].astype('category')

            if pd.api.types.is_numeric_dtype(data[y_val]):
                data[y_val] = data[y_val].astype('float64')
            elif not isinstance(data[y_val].dtype, pd.CategoricalDtype):
                data[y_val] = data[y_val].astype('category')
        except Exception as e:
            print(f"\t\tError converting data types for {x_val} and {y_val}: {str(e)}")
            return

        x_dtype = data[x_val].dtype
        y_dtype = data[y_val].dtype

        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            if test_type == "Spearman":
                coef, p_val = stats.spearmanr(data[x_val], data[y_val])
                test_result = (coef, p_val)
            elif test_type == "Pearson":
                coef, p_val = stats.pearsonr(data[x_val], data[y_val])
                test_result = (coef, p_val)
        elif isinstance(x_dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(y_dtype):
            categories = data[x_val].nunique()
            if categories == 2:
                if test_type == "t-test":
                    if check_assumptions(x_val, y_val, data, "t-test"):
                        t_stat, p_val = stats.ttest_ind(data[data[x_val] == data[x_val].cat.categories[0]][y_val], 
                                                        data[data[x_val] == data[x_val].cat.categories[1]][y_val])
                        test_result = (t_stat, p_val)
                    else:
                        print(f"\t\tCould not perform t-test on {x_val} and {y_val} due to unmet assumptions.")
                if test_type == "point-biserial":
                    coef, p_val = stats.pointbiserialr(data[x_val].cat.codes, data[y_val])
                    test_result = (coef, p_val)
            else:
                # Use observed=True to address the FutureWarning
                groups = [group for _, group in data.groupby(x_val, observed=True)[y_val]]
                if len(groups) < 2:
                    print(f"\t\tCannot perform ANOVA or Kruskal-Wallis test on {x_val} and {y_val}: fewer than two groups.")
                    return
                if check_assumptions(x_val, y_val, data, "ANOVA"):
                    f_stat, p_val = stats.f_oneway(*groups)
                    test_result = (f_stat, p_val)
                    test_type = "ANOVA"
                else:
                    test_stat, p_val = stats.kruskal(*groups)
                    test_result = (test_stat, p_val)
                    test_type = "Kruskal-Wallis"
        elif isinstance(x_dtype, pd.CategoricalDtype) and isinstance(y_dtype, pd.CategoricalDtype):
            contingency_table = pd.crosstab(data[x_val], data[y_val])
            if data[x_val].nunique() == 2 and data[y_val].nunique() == 2:
                odds_ratio, p_val = stats.fisher_exact(contingency_table)
                test_result = (odds_ratio, p_val)
                test_type = "Fisher's Exact"
            else:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                test_result = (chi2, p_val)
                test_type = "Chi-squared"
        else:
            print(f"\t\tIncompatible data types for {x_val} ({x_dtype}) and {y_val} ({y_dtype})")
            return

        if test_result and test_result[1] < 0.05:
            if test_result[1] < 0.001:
                p_value_str = "<0.001"
            else:
                p_value_str = f"{test_result[1]:.3f}"
            print(
                f"\t\t{x_val} and {y_val} - {test_type.title()} Test: Statistic={test_result[0]:.4f},"
                f" P-value={p_value_str}"
            )
            self.visualize_statistical_test(
                x_val,
                y_val,
                data,
                test_result,
                prefix,
                output_dir,
                test_type,
            )
        elif test_result and not test_result[1] < 0.05:
            print(
                f"\t\t{x_val} and {y_val} - {test_type.title()} -> (Not significant)"
            )
        else:
            print(
                f"\t\tCould not perform analysis on {x_val} and {y_val} due to incompatible data"
                " types or insufficient data."
            )

        # save all of the p-values and coefficients in a dictionary call results
        self.results[(x_val, y_val)] = test_result           
    
    def visualize_statistical_test(
        self,
        x_val,
        y_val,
        data,
        test_result,
        prefix,
        output_dir,
        test_type,
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
        """
        stat, p_val = test_result
        title = f"{x_val} vs {y_val} ({test_type.capitalize()}) \n"
        num_patients = data["Patient_ID"].nunique()

        units = {
            "Age": "days",
            "Date": "date",
            "Volume": "mm³",
            "Baseline Volume": "mm³",
        }

        x_unit = units.get(x_val, "")
        y_unit = units.get(y_val, "")

        # Plot based on test type
        if test_type in ["ANOVA", "Kruskal-Wallis"]:
            sns.boxplot(x=x_val, y=y_val, data=data, width=0.5)
            title += f"Statistic: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type in ["point-biserial", "t-test"]:
            sns.boxplot(x=x_val, y=y_val, data=data, width=0.5)
            title += f"Correlation Coefficient: {stat:.2f}, P-value: {p_val:.3e} (N={num_patients})"
        elif test_type in ["Spearman", "Pearson"]:
            sns.scatterplot(x=x_val, y=y_val, data=data)
            sns.regplot(x=x_val, y=y_val, data=data, scatter=False, color="blue")
            title += (
                f"{test_type} correlation coefficient: {stat:.2f}, P-value:"
                f" {p_val:.3e} (N={num_patients})"
            )
        elif test_type in ["Chi-squared", "Fisher's Exact"]:
            contingency_table = pd.crosstab(data[y_val], data[x_val])
            sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="g")
            if test_type == "Chi-squared":
                title += f"Chi2: {stat:.2f}, P-value: {p_val:.3e}, (N={num_patients})"
            else:
                title += f"Odds Ratio: {stat:.2f}, P-value: {p_val:.3e}, (N={num_patients})"
        
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
            output_dir, f"{prefix}_{x_val}_vs_{y_val}_{test_type}.png"
        )
        plt.savefig(save_file)
        plt.close()

    def correlation_analysis(self, prefix, output_dir, categorical_vars, numerical_vars):
        """
        Analyze the correlation between variables in the dataset.
        """
        print("\tCorrelations:")

        correlation_dir = os.path.join(output_dir, "correlations")
        os.makedirs(correlation_dir, exist_ok=True)
        for num_var in numerical_vars:
            for cat_var in categorical_vars:
                if self.merged_data[cat_var].nunique() == 2:
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.merged_data,
                        prefix,
                        correlation_dir,
                        test_type="t-test",
                    )
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.merged_data,
                        prefix,
                        correlation_dir,
                        test_type="point-biserial",
                    )
                else:
                    self.analyze_correlation(
                        cat_var,
                        num_var,
                        self.merged_data,
                        prefix,
                        correlation_dir,
                        test_type=None,
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
                        self.merged_data,
                        prefix,
                        correlation_dir,
                        test_type="Spearman",
                    )
                    self.analyze_correlation(
                        num_var,
                        other_num_var,
                        self.merged_data,
                        prefix,
                        correlation_dir,
                        test_type="Pearson",
                    )
        
        aggregated_data = (
            self.merged_data.sort_values("Age").groupby("Patient_ID", as_index=False).last()
        )
        for cat_var in categorical_vars:
            for other_cat_var in categorical_vars:
                if cat_var != other_cat_var:
                    self.analyze_correlation(
                        cat_var,
                        other_cat_var,
                        aggregated_data,
                        prefix,
                        correlation_dir,
                        test_type=None
                    )

class LogisticRegressionAnalysis:
    def __init__(self, data, cohort, output_dir):
        self.merged_data = data
        self.cohort = cohort
        self.reference_categories = {}

        os.makedirs(output_dir, exist_ok=True)

    def lr_analysis(self, output_dir, lr_vars, lr_combinations, outcome_var, categorical_vars):
        
        ####################################################################
        ##### Univariate analysis, logistic regression and forest plot #####
        ####################################################################

        pooled_results_uni = pooled_results_multi = pd.DataFrame(
            columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
        )
        print("\t\tUnivariate Analysis:")
        for variable in lr_vars:
            print(f"\t\tAnalyzing {variable}...")
            pooled_results_uni = self.univariate_analysis(
                variable,
                outcome_var,
                pooled_results_uni,
                categorical_vars,
            )
        self.plot_forest_plot(pooled_results_uni, output_dir, categorical_vars)
        print("\t\tUnivariate Analysis done! Forest Plot saved.")

        #############################################
        ##### Multi-variate logistic regression #####
        #############################################
        print("\t\tMultivariate Analysis:")

        pooled_results_multi = pd.DataFrame(
            columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
        )
        for combo in lr_combinations:
            try:
                pooled_results_multi = self.multivariate_analysis(
                    combo, outcome_var, pooled_results_multi, categorical_vars
                )
                self.plot_forest_plot(
                    pooled_results_multi,
                    output_dir,
                    categorical_vars,
                    analysis_type="Multivariate",
                    combo=combo,
                )
                pooled_results_multi.drop(pooled_results_multi.index, inplace=True) # enable to clear up th epooled results and not have a cumulative forest plot
            except KeyError as e:
                print(f"\t\tError fitting model with {combo}: {e}")
                pass
        print("\t\tMulti-variate Analysis done! Forest Plots saved.")

    def univariate_analysis(self, variable, outcome_var, pooled_results_uni, cat_vars):
        """
        Perform univariate logistic regression analysis for a given variable.
        """
        X, y = self.prepare_data_for_analysis(variable, outcome_var, cat_vars)

        if X is not None and not X.empty:
            try:
                # calculate_vif(X)
                result = logistic_regression_analysis(y, X)
                # print(result.summary2())
                pooled_results_uni = self.pool_results(
                    result, variable, pooled_results_uni, cat_vars
                )
                print(f"\t\t\tModel fitted successfully with {variable}.")
            except ExceptionGroup as e:
                print(f"\t\tError fitting model with {variable}: {e}")
        else:
            print(f"\t\tNo data available for {variable}.")

        return pooled_results_uni

    def prepare_data_for_analysis(self, variables, outcome_var, cat_vars):
        """
        Prepare the data for univariate logistic regression analysis.
        This function handles patient-constant and time-varying variables differently.
        """
        if isinstance(
            variables, str
        ):  # For univariate case where a single variable string is passed
            variables = [variables]

        if len(variables) == 1:
            # Univariate case
            variable = variables[0]
            data_agg = (
                self.merged_data.groupby("Patient_ID")
                .agg({variable: "first", outcome_var: "first"})
                .reset_index()
            )
        else:
            # Multivariate case
            data_agg = self.merged_data[
                ["Patient_ID"] + variables + [outcome_var]
            ].copy()

            # Separate categorical and numerical variables
            cat_vars_subset = [var for var in variables if var in cat_vars]
            num_vars_subset = [var for var in variables if var not in cat_vars]

            # Aggregate categorical variables by taking the mode, numerical the mean and outcome the first value per patient
            agg_dict = {}
            for var in cat_vars_subset:
                agg_dict[var] = lambda x: x.value_counts().index[0]  # Mode for categorical
            for var in num_vars_subset:
                agg_dict[var] = 'mean'  # Mean for numerical
            agg_dict[outcome_var] = 'first'  # First value for outcome

            data_agg = data_agg.groupby("Patient_ID").agg(agg_dict).reset_index()
        

        for variable in variables:
            # For categorical variables, convert them to dummy variables
            if variable in cat_vars:
                reference_category = data_agg[variable].mode()[0]
                ref_count = (data_agg[variable] == reference_category).sum()
                print("\t\t\tReference category: ", reference_category)
                self.reference_categories[variable] = (reference_category, ref_count)
                data_agg[variable] = data_agg[variable].astype(str)
                dummies = pd.get_dummies(
                    data_agg[variable], prefix=variable, drop_first=False
                )
                if f"{variable}_{reference_category}" in dummies.columns:
                    dummies.drop(
                        columns=[f"{variable}_{reference_category}"], inplace=True
                    )
                data_agg = pd.concat(
                    [data_agg.drop(columns=[variable]), dummies], axis=1
                )
                for col in dummies.columns:
                    data_agg[col] = data_agg[col].astype(int)
            else:
                data_agg[variable] = pd.to_numeric(data_agg[variable], errors="coerce")
                if (data_agg[variable] <= 0).any():
                    # Handle zeros or negative values if necessary, e.g., by adding a small constant
                    # data_agg[variable] += 1
                    data_agg[variable] = data_agg[variable].replace(0, 0.1)
                    data_agg[variable] = data_agg[variable].clip(lower=0.1)
                # Apply log transformation
                data_agg[variable] = np.log(data_agg[variable])

        # Ensure outcome_var is binary numeric, reduce to relevant columns, check for missing values
        data_agg[outcome_var] = (
            pd.to_numeric(data_agg[outcome_var], errors="coerce").fillna(0).astype(int)
        )

        # drop patient ID and assign constant for regression
        data_agg = data_agg.drop(columns=["Patient_ID"], errors="ignore")
        data_agg = data_agg[
            [outcome_var] + [col for col in data_agg.columns if col != outcome_var]
        ]
        data_agg.dropna(inplace=True)
        if "const" not in data_agg.columns:
            data_agg = sm.add_constant(data_agg)

        if data_agg.empty:
            print("\t\tWarning: No data available. Recheck code and data structure.")
            return None, None
        else:
            y = data_agg[outcome_var]
            X = data_agg.drop(columns=[outcome_var], errors="ignore")
            return X, y

    def plot_forest_plot(
        self,
        pooled_results,
        output_dir,
        cat_vars,
        analysis_type="Univariate",
        combo=None,
    ):
        """
        Create a forest plot from the pooled results of univariate analyses.

        Args:
            pooled_results: DataFrame with columns 'Variable', 'OR', 'Lower', 'Upper', and 'p'.
            output_file: File path to save the forest plot image.
        """
        # print(pooled_results)
        expected_columns = {"MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"}
        if not expected_columns.issubset(pooled_results.columns):
            missing_cols = expected_columns - set(pooled_results.columns)
            raise ValueError(
                f"The DataFrame is missing the following required columns: {missing_cols}"
            )
        # Exclude 'Reference' entries from calculations
        reference_mask = pooled_results["Subcategory"].str.contains("Reference")
        references = pooled_results[reference_mask]
        filtered_results = pooled_results[~reference_mask]

        # sort pooled results alphabetically, then clear out non-positive and infinite values
        filtered_results = filtered_results[
            (filtered_results["OR"] > 0)
            & (filtered_results["Lower"] > 0)
            & (filtered_results["Upper"] > 0)
        ]
        filtered_results.replace([np.inf, -np.inf], np.nan, inplace=True)
        filtered_results.dropna(subset=["OR", "Lower", "Upper", "p"], inplace=True)
        if not filtered_results.empty:
            max_hr = 100  # remove outliers
            filtered_results = filtered_results[filtered_results["Upper"] <= max_hr]

        age_groups = ["Infant", "Preschool", "School Age", "Adolescent"]
        # Modify the sorting logic
        if 'Age Group at Diagnosis' in pooled_results['MainCategory'].unique():
            # Create a custom sorting key
            def custom_sort(row):
                if row['MainCategory'] == 'Age Group at Diagnosis':
                    return (0, age_groups.index(row['Subcategory']) if row['Subcategory'] in age_groups else len(age_groups))
                else:
                    return (1, row['MainCategory'], row['Subcategory'])

            final_results = pd.concat([filtered_results, references], ignore_index=True)
            final_results['sort_key'] = final_results.apply(custom_sort, axis=1)
            final_results.sort_values(by='sort_key', ascending=False, inplace=True)
            final_results.drop('sort_key', axis=1, inplace=True)
            final_results.reset_index(drop=True, inplace=True)
        else:
            # If 'Age Group at Diagnosis' is not present, use the original sorting
            final_results = pd.concat([filtered_results, references], ignore_index=True)
            final_results.sort_values(
                by=["MainCategory", "Subcategory"], ascending=[False, False], inplace=True
            )
            final_results.reset_index(drop=True, inplace=True)


        # General plot settings + x parameters
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.3, right=0.7)
        ax.set_xscale("log")
        ax.set_xlim(left=0.01, right=100)
        ax.set_xlabel("<-- Lower Risk of Progression | Higher Risk of Progression -->", fontdict={"fontsize": 15})
        ax.axvline(x=1, linestyle="--", color="blue", lw=1)

        # Categories handling and colors
        unique_main_categories = final_results["MainCategory"].unique()
        colormap = plt.get_cmap("tab20")
        colors = [colormap(i) for i in range(len(unique_main_categories))]
        category_colors = {
            cat: color for cat, color in zip(unique_main_categories, colors)
        }

        # annotations on the right
        ax.margins(x=1)
        fig.canvas.draw()  # Need to draw the canvas to update axes positions

        # Get the bounds of the axes in figure space
        ax_bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Calculate the figure and axes widths in inches
        fig_width_inches = fig.get_size_inches()[0]
        axes_width_inches = ax_bounds.width
        annotation_x_position = ax_bounds.x1 + 0.01 * fig_width_inches

        # Annotations on the left
        copy_df = self.merged_data.copy()
        unique_pat = copy_df.drop_duplicates(subset=["Patient_ID"])
        y_labels = []
        for i, row in enumerate(final_results.itertuples()):
            main_category = row.MainCategory
            subcategory = row.Subcategory
            if "(Reference)" in subcategory:
                _, count = self.reference_categories.get(main_category, (None, 0))
            else:
                if main_category in cat_vars:
                    count = unique_pat[main_category].value_counts().get(subcategory, 0)
                else:
                    count = len(unique_pat)
            label = f"{main_category} - {subcategory} - {count}"
            y_labels.append(label)

            # plotting
            if "(Reference)" not in subcategory:
                ax.errorbar(
                    row.OR,
                    i,
                    xerr=[[row.OR - row.Lower], [row.Upper - row.OR]],
                    fmt="o",
                    color=category_colors[main_category],
                    ecolor=category_colors[main_category],
                    elinewidth=1,
                    capsize=3,
                )
                ax.text(
                    annotation_x_position + (40 * axes_width_inches),
                    i,
                    f"{row.OR:.2f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
                ax.text(
                    annotation_x_position + (100 * axes_width_inches),
                    i,
                    f"({row.Lower:.2f}-{row.Upper:.2f})",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
                ax.text(
                    annotation_x_position + (600 * axes_width_inches),
                    i,
                    f"{row.p:.3f}" if row.p >= 0.01 else "<0.01",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
            else:
                ax.errorbar(
                    1.0,
                    i,
                    fmt="^",
                    color=category_colors[main_category],
                    capsize=3,
                )
                ax.text(
                    annotation_x_position + (40 * axes_width_inches),
                    i,
                    "Reference",
                    ha="left",
                    va="center",
                    fontsize=8,
                    transform=ax.transData,
                )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, ha="right")

        # titles on the plot
        ax.text(
            -0.35,
            1.01,
            "Variables and \n Subgroups",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            -0.2,
            1.01,
            "Count (n)",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.05,
            1.01,
            "OR",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.15,
            1.01,
            "95% CI",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            1.35,
            1.01,
            "P-val",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Add title, grid, and layout
        ax.set_title(f"{analysis_type} Analysis Forest Plot", fontdict={"fontsize": 18})
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

        if analysis_type == "Multivariate":
            combo_str = "_".join(combo)
            combo_len = len(combo_str)
            output_file = os.path.join(
                output_dir, f"{analysis_type}_{combo_len}_forest_plot.png"
            )
        else:
            output_file = os.path.join(output_dir, f"{analysis_type}_forest_plot.png")
        plt.savefig(output_file, dpi=300)
        plt.close()

    @staticmethod
    def safe_exp(x):
        try:
            return np.exp(x)
        except OverflowError or RuntimeWarning:
            return np.nan

    def pool_results(self, result, variables, pooled_results, cat_vars):
        """
        Pool the results of univariate analysis to create a forest plot.

        Args:
            result: Result object from univariate analysis.
            pooled_results: DataFrame to store pooled results.
        """
        if result is None:
            raise ValueError("No result object provided for pooling.")

        if not isinstance(pooled_results, pd.DataFrame) or pooled_results is None:
            pooled_results = pd.DataFrame(
                columns=["MainCategory", "Subcategory", "OR", "Lower", "Upper", "p"]
            )

        if not isinstance(variables, list):
            variables = [variables]

        for variable in variables:
            if variable in cat_vars:
                reference_category = self.reference_categories.get(variable, None)
                if reference_category is None:
                    raise ValueError(f"No reference category set for {variable}")
                ref_row = pd.DataFrame(
                    {
                        "MainCategory": variable,
                        "Subcategory": f"{reference_category} (Reference)",
                        "OR": 1.0,
                        "Lower": np.nan,
                        "Upper": np.nan,
                        "p": np.nan,
                    },
                    index=[0],
                )
                ref_row = ref_row.dropna(axis=1, how='all')
                missing_cols = set(pooled_results.columns) - set(ref_row.columns)
                for col in missing_cols:
                    ref_row[col] = np.nan
                pooled_results = pd.concat([pooled_results, ref_row], ignore_index=True)
        for idx in result.params.index:
                if idx != "const":
                    parts = idx.split("_")
                    main_category = parts[0]
                    subcategory = " ".join(parts[1:]) if len(parts) > 1 else "Continuous"

                    coef = result.params[idx]
                    conf = result.conf_int().loc[idx].values
                    p_val = result.pvalues[idx]

                    new_row = pd.DataFrame(
                        {
                            "MainCategory": main_category,
                            "Subcategory": subcategory,
                            "OR": self.safe_exp(coef),
                            "Lower": self.safe_exp(conf[0]),
                            "Upper": self.safe_exp(conf[1]),
                            "p": p_val,
                        },
                        index=[0],
                    )
                    pooled_results = pd.concat([pooled_results, new_row], ignore_index=True)
                    print(f"\t\t\tPooled results updated with {idx}.")

        return pooled_results

    def multivariate_analysis(
        self, variables, outcome_var, pooled_results_multi, cat_vars
    ):
        """
        Perform multivariate logistic regression analysis for a given set of variables.
        """
        X, y = self.prepare_data_for_analysis(variables, outcome_var, cat_vars)
        calculate_vif(X, variables)

        if X is not None and not X.empty:
            try:
                result = logistic_regression_analysis(y, X)
                # print(result.summary2())
                print(f"\t\t\tModel fitted successfully with {variables}.")
                pooled_results_multi = self.pool_results(
                    result, variables, pooled_results_multi, cat_vars
                )

            except ExceptionGroup as e:
                print(f"\t\tError fitting model with {variables}: {e}")
        else:
            print(f"\t\tNo data available for {variables}.")

        return pooled_results_multi



if __name__ == '__main__':
    cohort = lr_and_correlations_cfg.COHORT
    data_path = lr_and_correlations_cfg.COHORT_DATAFRAME
    output_dir = lr_and_correlations_cfg.OUTPUT_DIR
    outcome_var = lr_and_correlations_cfg.OUTCOME_VAR
    categorical_vars = lr_and_correlations_cfg.CATEGORICAL_VARS
    numerical_vars = lr_and_correlations_cfg.NUMERICAL_VARS
    lr_vars = lr_and_correlations_cfg.LR_VARS
    lr_combinations = lr_and_correlations_cfg.LR_COMBINATIONS
    
    # Load the data
    cohort_data = pd.read_csv(data_path)
    # Initialize the logistics regression analysis
    lr = LogisticRegressionAnalysis(cohort_data, cohort, output_dir)
    lr.lr_analysis(output_dir, lr_vars, lr_combinations, outcome_var, categorical_vars)
    # Initialize the correlation analysis
    corr = CorrelationAnalysis(cohort_data, cohort)
    #corr.correlation_analysis(cohort, output_dir, categorical_vars, numerical_vars)
    