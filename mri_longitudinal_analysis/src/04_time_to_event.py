"""
Script to run the time to event analysis.
"""
from typing import Optional, Union, Iterable 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cfg.src import time2event_cfg
from cfg.utils.helper_functions_cfg import NORD_PALETTE
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import remove_ticks, remove_spines, move_spines # add_at_risk_counts ; just in case default should be used
from itertools import combinations
from utils.helper_functions import calculate_vif, calculate_progression


class Time2Event:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
    
    def time_to_event_analysis(self, prefix, output_dir, variables, duration_col, event_col, stratify_by=None, progression_type="composite"):
        """
        Perform a Kaplan-Meier survival analysis on time-to-event data for tumor progression.

        Parameters:
        - prefix (str): Prefix used for naming the output file.

        The method fits the survival curve using the KaplanMeierFitter on the pre-treatment data,
        saves the plot image, and prints a confirmation message.
        """
        # pre_treatment_data as df copy for KM curves
        print("\tPerforming time-to-event analysis: Kaplan-Meier Survival Analysis.")        
        analysis_data_pre = self.data.copy()
        
        if progression_type == "volumetric":
            analysis_data_pre[event_col] = ~analysis_data_pre[
                "Age at First Progression"
            ].isna()
            # Compare the results of both approaches and check if they match
            analysis_data_pre[duration_col] = np.where(
                analysis_data_pre[event_col],
                analysis_data_pre["Time to Progression"],
                analysis_data_pre["Follow-Up Time"],
            )
            output_dir = os.path.join(output_dir, "volumetric")
        elif progression_type == "composite":            
            # Define Event_Occurred and Duration for composite progression
            analysis_data_pre[event_col] = (
                ~analysis_data_pre["Age at First Progression"].isna() | 
                (analysis_data_pre["Received Treatment"] == "Yes")
            )
            
            analysis_data_pre[duration_col] = np.where(
                analysis_data_pre[event_col],
                np.where(
                    ~analysis_data_pre["Age at First Progression"].isna(),
                    analysis_data_pre["Time to Progression"],
                    analysis_data_pre["Follow-Up Time"]
                ),
                analysis_data_pre["Follow-Up Time"]
            )
            output_dir = os.path.join(output_dir, "composite")
        
        os.makedirs(output_dir, exist_ok=True) 
        analysis_data_pre = analysis_data_pre.dropna(
            subset=[duration_col, event_col]
        )

        for element in stratify_by:
            if element is not None:
                self.kaplan_meier_analysis(
                    analysis_data_pre, output_dir, duration_col, event_col, element, prefix
                )
            else:
                self.kaplan_meier_analysis(
                    analysis_data_pre, output_dir, duration_col, event_col, prefix=prefix,
                )
                self.cox_proportional_hazards_analysis(analysis_data_pre, output_dir, variables, duration_col, event_col)

    ####################
    ### KM Estimator ###
    ####################
    def kaplan_meier_analysis(self, data, output_dir, duration_col, event_col, stratify_by=None, prefix=""):
        """
        Kaplan-Meier survival analysis for time-to-event data.
        """
        surv_dir = os.path.join(output_dir, "KM_survival")
        os.makedirs(surv_dir, exist_ok=True)
        colors = sns.color_palette(NORD_PALETTE, n_colors=len(NORD_PALETTE))
        analysis_data_pre = data.copy()
        analysis_data_pre = analysis_data_pre.drop_duplicates(subset=['Patient_ID'], keep='first')
        kmf = KaplanMeierFitter()
        
        analysis_data_pre.loc[:,"Duration_Months"] = analysis_data_pre[duration_col] / 30.44  # Average days in a month

        if stratify_by and stratify_by in analysis_data_pre.columns:
            unique_categories = analysis_data_pre[stratify_by].unique()
            if len(unique_categories) > 1:
                groups = []
                kmfs = []
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, category in enumerate(unique_categories):
                    category_data = analysis_data_pre[analysis_data_pre[stratify_by] == category]
                    if category_data.empty:
                        continue
                    kmf = KaplanMeierFitter()

                    kmf.fit(
                        category_data["Duration_Months"],
                        event_observed=category_data[event_col],
                        label=str(category),
                    )
                    kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=colors[i % len(colors)])
                    kmfs.append(kmf)
                    groups.append((category, category_data))
                
                plt.title(f"Stratified Survival Function by {stratify_by}", fontsize=20)
                plt.xlabel("Months since Diagnosis", fontsize=15)
                plt.ylabel("PFS Probability", fontsize=15)
                plt.legend(title=stratify_by, loc="best", fontsize=12)
                if kmfs:
                    #add_at_risk_counts(*kmfs, ax=ax)
                    self.add_at_risk_counts_monthly(*kmfs, ax=ax)
                max_months = int(analysis_data_pre["Duration_Months"].max())
                xticks = range(0, max_months + 13, 12)  # +13 to ensure the last tick is included
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks)
                ax.set_xlim(0, max_months)
                    
                # Save the combined plot
                survival_plot = os.path.join(surv_dir, f"{prefix}_survival_plot_category_{stratify_by}.png")
                plt.tight_layout()
                plt.savefig(survival_plot, dpi=300)
                plt.close(fig)
                print(f"\t\tSaved survival KaplanMeier curve for {stratify_by}.")

                # Perform pairwise log-rank tests
                pairwise_results = []
                for (cat1, group1), (cat2, group2) in combinations(groups, 2):
                    result = logrank_test(
                        group1["Duration_Months"], group2["Duration_Months"],
                        event_observed_A=group1[event_col], event_observed_B=group2[event_col]
                    )
                    p_value = result.p_value
                    display_p_value = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
                    pairwise_results.append((cat1, cat2, display_p_value))
                
                # Save pairwise log-rank test results
                results_file = os.path.join(surv_dir, f"{prefix}_pairwise_logrank_results_{stratify_by}.txt")
                with open(results_file, "w", encoding='utf-8') as f:
                    f.write("Combination\tp-value\n")
                    for cat1, cat2, display_p_value in pairwise_results:
                        f.write(f"{cat1} vs {cat2}\t{display_p_value}\n")
        else:
            fig, ax = plt.subplots(figsize=(9, 8))
            kmf.fit(
                analysis_data_pre["Duration_Months"],
                event_observed=analysis_data_pre[event_col],
            )
            kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True)
            ax.set_title("Survival function of Tumor Progression", fontdict={"fontsize": 20})
            ax.set_xlabel("Months since Diagnosis", fontdict={"fontsize": 15})
            ax.set_ylabel("PFS Probability", fontdict={"fontsize": 15})
            #add_at_risk_counts(kmf, ax=ax)
            self.add_at_risk_counts_monthly(kmf, ax=ax)
            
            # Save the plot
            survival_plot = os.path.join(surv_dir, f"{prefix}_survival_plot.png")
            plt.tight_layout()
            plt.savefig(survival_plot, dpi=300)
            plt.close(fig)
            print("\t\tSaved survival KaplanMeier curve.")

    def add_at_risk_counts_monthly(
        self,
        *fitters,
        labels: Optional[Union[Iterable, bool]] = None,
        rows_to_show=None,
        ypos=-0.6,
        xticks=None,
        ax=None,
        at_risk_count_from_start_of_period=False,
        **kwargs
    ):
        """
        Add counts showing how many individuals were at risk, censored, and observed, at each time point in
        survival/hazard plots. Adjusted for monthly intervals.
        """
        if ax is None:
            ax = plt.gca()
        fig = kwargs.pop("fig", None)
        if fig is None:
            fig = plt.gcf()
        if labels is None:
            labels = [f.label for f in fitters]
        elif labels is False:
            labels = [None] * len(fitters)
        if rows_to_show is None:
            rows_to_show = ["At risk", "Censored", "Events"]
        else:
            assert all(
                row in ["At risk", "Censored", "Events"] for row in rows_to_show
            ), 'must be one of ["At risk", "Censored", "Events"]'
        n_rows = len(rows_to_show)

        # Create another axes where we can put size ticks
        ax2 = plt.twiny(ax=ax)
        # Move the ticks below existing axes
        ax_height = (
            ax.get_position().y1 - ax.get_position().y0
        ) * fig.get_figheight()  # axis height
        ax2_ypos = ypos / ax_height

        move_spines(ax2, ["bottom"], [ax2_ypos])
        remove_spines(ax2, ["top", "right", "bottom", "left"])
        ax2.xaxis.tick_bottom()
        min_time, max_time = ax.get_xlim()
        ax2.set_xlim(min_time, max_time)
        
        # Adjust xticks for monthly intervals if not provided
        if xticks is None:
            xticks = np.arange(0, int(max_time) + 24, 24)  # Bi-Yearly intervals
        ax2.set_xticks(xticks)
        remove_ticks(ax2, x=True, y=True)

        ticklabels = []

        for tick in ax2.get_xticks():
            lbl = ""
            counts = []
            for f in fitters:
                if at_risk_count_from_start_of_period:
                    event_table_slice = f.event_table.assign(at_risk=lambda x: x.at_risk)
                else:
                    event_table_slice = f.event_table.assign(
                        at_risk=lambda x: x.at_risk - x.removed
                    )
                if not event_table_slice.loc[:tick].empty:
                    event_table_slice = (
                        event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]]
                        .agg(
                            {
                                "at_risk": lambda x: x.tail(1).values,
                                "censored": "sum",
                                "observed": "sum",
                            }
                        )
                        .rename(
                            {
                                "at_risk": "At risk",
                                "censored": "Censored",
                                "observed": "Events",
                            }
                        )
                        .fillna(0)
                    )
                    counts.extend([int(c) for c in event_table_slice.loc[rows_to_show]])
                else:
                    counts.extend([0 for _ in range(n_rows)])
            
            # Format the label
            if n_rows > 1:
                if tick == ax2.get_xticks()[0]:
                    max_length = len(str(max(counts)))
                    for i, c in enumerate(counts):
                        if i % n_rows == 0:
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"%s" % labels[int(i / n_rows)]
                                + "\n"
                            )
                        l = rows_to_show[i % n_rows]
                        s = (
                            "{}".format(l.rjust(10, " "))
                            + (" " * (max_length - len(str(c)) + 3))
                            + "{{:>{}d}}\n".format(max_length)
                        )
                        lbl += s.format(c)
                else:
                    for i, c in enumerate(counts):
                        if i % n_rows == 0 and i > 0:
                            lbl += "\n\n"
                        s = "\n{}"
                        lbl += s.format(c)
            else:
                if tick == ax2.get_xticks()[0]:
                    max_length = len(str(max(counts)))
                    lbl += rows_to_show[0] + "\n"
                    for i, c in enumerate(counts):
                        s = (
                            "{}".format(labels[i].rjust(10, " "))
                            + (" " * (max_length - len(str(c)) + 3))
                            + "{{:>{}d}}\n".format(max_length)
                        )
                        lbl += s.format(c)
                else:
                    for i, c in enumerate(counts):
                        s = "\n{}"
                        lbl += s.format(c)
            ticklabels.append(lbl)
        
        ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

        return ax

    #####################
    ### PCH Estimator ###
    #####################
    def cox_proportional_hazards_analysis(self, data, output_dir, variables, duration_col, event_col):
        """
        Cox proportional hazards analysis for time-to-event data.
        """
        analysis_data, variables, cat_vars, _ = self.preprocess_data_hz_model(data, variables, duration_col, event_col)
        analysis_data[duration_col] = analysis_data[duration_col] / 365.25  # Convert days to years
        surv_dir = os.path.join(output_dir, "PCH_survival")
        os.makedirs(surv_dir, exist_ok=True)

        print("\t\tPerforming univariate Cox proportional hazards analysis.")
        results_uni, pooled_results = self.univariate_analysis(analysis_data, duration_col, event_col)
        
        # 3. Plot the results
        self.plot_cox_proportional_hazards(results_uni, surv_dir, suffix="univariate")
        
        # 4. Feature selection
        selected_features = self.feature_selection(results_uni, p_value_threshold=0.05)
        print(f"\t\tSelected features: {selected_features}")
        
        # 5. VIF Caclulation
        calculate_vif(analysis_data[selected_features], cat_vars)
        
        # 6. Multivariate analysis
        print("\n\t\tPerforming multivariate Cox proportional hazards analysis.")
        results_multi, multivariate_model = self.multivariate_analysis(analysis_data, duration_col, event_col, selected_features)
        print("\nMultivariate Cox Proportional Hazards Results:")
        multivariate_model.print_summary()

        # 7. Plot the results of multivariate analysis
        self.plot_cox_proportional_hazards(results_multi, surv_dir, suffix="multivariate")
        self.plot_survival_curves(multivariate_model, analysis_data, surv_dir, duration_col, event_col, plot_type="survival")
        self.plot_survival_curves(multivariate_model, analysis_data, surv_dir, duration_col, event_col, plot_type="log_log")

        # 8. C-Index
        c_index, ci = self.calculate_c_index_with_ci(multivariate_model, analysis_data, duration_col, event_col)
        print(f"\t\tC-index: {c_index:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")    

    def preprocess_data_hz_model(self, data, variables, duration_col, event_col):
        """
        Preprocess the data for Cox proportional hazards analysis.
        """
        print("\tPerforming time-to-event analysis: Cox-Hazard Survival Analysis.")        
        list_of_columns = variables + [duration_col, event_col]
        analysis_data = data[list_of_columns].copy()
        
        # Identify column types
        categorical_columns = []
        continuous_columns = []
        for col in list_of_columns:
            if col in [duration_col, event_col]:
                continue
            elif analysis_data[col].dtype == 'object' or analysis_data[col].nunique() < 10:
                categorical_columns.append(col)
            else:
                continuous_columns.append(col)
        
        for var in categorical_columns:
            analysis_data[var] = analysis_data[var].astype('category')
            dummies = pd.get_dummies(analysis_data[var], prefix=var, drop_first=True)
            analysis_data = pd.concat([analysis_data, dummies], axis=1)
            analysis_data.drop(var, axis=1, inplace=True)

        #scaler = StandardScaler()
        #analysis_data[continuous_columns] = scaler.fit_transform(analysis_data[continuous_columns])         
        for col in continuous_columns:
            analysis_data[col] = pd.to_numeric(analysis_data[col], errors='coerce')
        
        # Handle missing and infinite values
        analysis_data = analysis_data.replace([np.inf, -np.inf], np.nan).dropna()
        analysis_data = analysis_data.reset_index(drop=True)

        return analysis_data, variables, categorical_columns, continuous_columns

    def univariate_analysis(self, data, duration_col, event_col):
        """Perform univariate Cox proportional hazards analysis."""
        results = {}
        pooled_results = pd.DataFrame(columns=["Variable", "Subcategory", "HR", "Lower", "Upper", "p", "C-index"])

        for column in data.columns:
            if column not in [duration_col, event_col]:
                try:
                    model = CoxPHFitter(baseline_estimation_method="breslow")
                    model.fit(data[[column, duration_col, event_col]],
                            duration_col=duration_col,
                            event_col=event_col)
                    
                    summary = model.summary
                    
                    #c_index, ci = self.calculate_c_index_with_ci(model, data, duration_col, event_col)
                    
                    results[column] = {
                        'p_value': summary.loc[column, 'p'],
                        'hazard_ratio': summary.loc[column, 'exp(coef)'],
                        'confidence_interval': (
                            summary.loc[column, 'exp(coef) lower 95%'],
                            summary.loc[column, 'exp(coef) upper 95%']
                        ), 
                        #'c_index': c_index, 
                        #'c_index_ci_lower': ci[0],
                        #'c_index_ci_upper': ci[1]
                        
                    }
                              
                except Exception as e:
                    print(f"Error in univariate analysis for {column}: {str(e)}")

        return pd.DataFrame(results).T, pooled_results

    def feature_selection(self, univariate_results, p_value_threshold=0.05):
        """Select features based on univariate analysis results."""
        return univariate_results[univariate_results['p_value'] < p_value_threshold].index.tolist()

    def multivariate_analysis(self, data, duration_col, event_col, features):
        """Perform multivariate Cox proportional hazards analysis."""
        results = {}
        model = CoxPHFitter(baseline_estimation_method="breslow")
        model.fit(data[features + [duration_col, event_col]], 
                             duration_col=duration_col, 
                             event_col=event_col)
        summary = model.summary 
        results = pd.DataFrame({
            'hazard_ratio': summary['exp(coef)'],
            'confidence_interval': list(zip(summary['exp(coef) lower 95%'], summary['exp(coef) upper 95%'])),
            'p_value': summary['p']
        })
        return results, model

    def calculate_c_index_with_ci(self, cph, data, duration_col, event_col, n_bootstraps=800):
        """
        Calculate the concordance index (C-index) with confidence intervals.
        """
        # Calculate the main C-index
        data = data.reset_index(drop=True)
        c_index = concordance_index(
            data[duration_col],
            -cph.predict_partial_hazard(data),
            data[event_col]
        )

        # Bootstrap to calculate confidence interval
        c_indices = []
        n_samples = len(data)
        for _ in range(n_bootstraps):
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_sample = data.iloc[boot_indices].reset_index(drop=True)
            try:
                c_ind = concordance_index(
                    boot_sample[duration_col],
                    -cph.predict_partial_hazard(boot_sample),
                    boot_sample[event_col]
                )
                c_indices.append(c_ind)
            except ZeroDivisionError:
                # Skip this bootstrap sample if there are no admissible pairs
                continue

        if len(c_indices) < n_bootstraps * 0.9:  # If we lost more than 10% of our bootstrap samples
            print(f"Warning: Only {len(c_indices)} out of {n_bootstraps} bootstrap samples were valid.")
        
        if len(c_indices) == 0:
            print("Error: No valid bootstrap samples. Unable to calculate confidence interval.")
            return c_index, (None, None)
        
        ci_lower = np.percentile(c_indices, 2.5)
        ci_upper = np.percentile(c_indices, 97.5)

        return c_index, (ci_lower, ci_upper)    

    ################
    ### PLOTTING ###
    ################
    def plot_cox_proportional_hazards(self, results, output_dir, suffix=""):
        """Forest plot of the Cox proportional hazards model."""
        
        hazard_ratios = results['hazard_ratio']
        confidence_intervals = results['confidence_interval']
        p_values = results['p_value']
        # New sorting logic
        sorted_vars = [(var.split('_')[0], var) for var in hazard_ratios.index]
        sorted_vars.sort(key=lambda x: (x[0], x[1]))
        sorted_order = [var[1] for var in sorted_vars]
        sorted_order = sorted_order[::-1]

        # Reorder the data based on the new sorting
        hazard_ratios = hazard_ratios[sorted_order]
        confidence_intervals = confidence_intervals.loc[sorted_order]
        p_values = p_values.loc[sorted_order]

        # Extract main categories from variable names
        main_categories = [var.split('_')[0] for var in hazard_ratios.index]
        unique_main_categories = sorted(set(main_categories))

        # Create color map
        colormap = plt.get_cmap("tab20")
        colors = [colormap(i) for i in range(len(unique_main_categories))]
        category_colors = {cat: color for cat, color in zip(unique_main_categories, colors)}

        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.3, right=0.7)
        y_pos = range(len(hazard_ratios))
        
        # Plot points for hazard ratios and error bars with colors
        for i, (var, hr) in enumerate(hazard_ratios.items()):
            category = var.split('_')[0]
            color = category_colors[category]
            
            ci_lower, ci_upper = confidence_intervals.loc[var]

            ax.errorbar(
                1 if pd.isna(hr) else hr, i, 
                xerr=[[0, 0]] if pd.isna(hr) else [[hr - ci_lower], [ci_upper - hr]],
                fmt='^' if pd.isna(hr) else 'o',
                color=color,
                ecolor=color,
                capsize=5,
                capthick=2,
                markersize=6,
                elinewidth=2,
                zorder=2
            )

        ax.axvline(x=1, color='r', linestyle='--', zorder=0)
        ax.set_xscale('log')
        ax.set_xlabel('<-- Lower Risk of Progression | Higher Risk of Progression -->', fontdict={'fontsize': 15})
        ax.set_ylabel('Variables', fontdict={'fontsize': 15})
        ax.set_title('Proportional Cox-Hazards Model', fontsize=20, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hazard_ratios.index, ha='right')
        
        ax.set_xlim(0.1, 10)

        fig.canvas.draw()
        ax_bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_width_inches = fig.get_size_inches()[0]
        axes_width_inches = ax_bounds.width
        annotation_x_position = ax_bounds.x1 + 0.01 * fig_width_inches

        for i, (var, hr) in enumerate(hazard_ratios.items()):
            ci_lower, ci_upper = confidence_intervals.loc[var]
            p_value = p_values.loc[var]
            
            ax.text(
                annotation_x_position + (0.75 * axes_width_inches),
                i,
                f"{hr:.2f}",
                ha="left",
                va="center",
                fontsize=10,
                transform=ax.transData,
            )
            ax.text(
                annotation_x_position + (2.25 * axes_width_inches),
                i,
                f"({ci_lower:.2f}-{ci_upper:.2f})",
                ha="left",
                va="center",
                fontsize=10,
                transform=ax.transData,
            )
            ax.text(
                annotation_x_position + (5.75 * axes_width_inches),
                i,
                f"{p_value:.3f}" if p_value >= 0.01 else "<0.01",
                ha="left",
                va="center",
                fontsize=10,
                transform=ax.transData,
            )

        ax.text(1.05, 1.0, "HR", ha="left", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(1.15, 1.0, "95% CI", ha="left", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(1.28, 1.0, "P-val", ha="left", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        
        cox_plot = os.path.join(output_dir, f"cox_proportional_hazards_plot_{suffix}.png")
        plt.savefig(cox_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\t\tSaved {suffix} Cox proportional hazards plot.")

    def plot_survival_curves(self, cph, analysis_data, output_dir, duration_col, event_col, plot_type="survival"):
        """
        Plot either survival curves or log-log curves for each covariate.
        
        Parameters:
        - cph: Fitted CoxPHFitter model
        - analysis_data: DataFrame containing the analysis data
        - output_dir: Directory to save the plots
        - plot_type: 'survival' for survival curves, 'log_log' for log-log plots
        """
        if plot_type not in ['survival', 'log_log']:
            raise ValueError("plot_type must be either 'survival' or 'log_log'")
        
        log_log_dir = os.path.join(output_dir, "log_log")
        survival_dir = os.path.join(output_dir, "survival")
        os.makedirs(log_log_dir, exist_ok=True)
        os.makedirs(survival_dir, exist_ok=True)
        
        analysis_data = analysis_data.reset_index(drop=True)
        categorical_vars = [col for col in analysis_data.columns if '_' in col]
        continuous_vars = [col for col in analysis_data.columns if col not in categorical_vars + [duration_col, event_col]]

        for var in categorical_vars + continuous_vars:
            plt.figure(figsize=(10, 6))
            
            if var in categorical_vars:
                unique_values = analysis_data[var].unique()
                if len(unique_values) <= 1:
                    print(f"Skipping {var} as it has only one unique value.")
                    plt.close()
                    continue
                values = [[unique_values[0]], [unique_values[1]]]
            else:  # continuous variable
                if analysis_data[var].dtype not in ['int64', 'float64']:
                    print(f"Skipping {var} as it is not a numeric type.")
                    plt.close()
                    continue
                q1, q3 = analysis_data[var].quantile([0.25, 0.75])
                values = [[q1], [q3]]

            try:   
                if plot_type == 'survival':
                    cph.plot_partial_effects_on_outcome(
                        covariates=var,
                        values=values,
                        plot_baseline=False
                    )
                    plt.title(f"Survival Curves for {var}")
                    plt.xlabel("Time")
                    plt.ylabel("Survival Probability")
                    plot_dir = survival_dir
                else:  # log_log
                    cph.plot_partial_effects_on_outcome(
                        covariates=var,
                        values=values,
                        plot_baseline=False,
                        y="cumulative_hazard"
                    )
                    plt.title(f"Log-Log Plot for {var}")
                    plt.xlabel("Time [years]")
                    plt.ylabel("Log(-Log(Survival Probability))")
                    plot_dir = log_log_dir
            except ExceptionGroup as e:
                print(f"Error: {e}")
            finally:
                # Save the plot
                var_name = var.replace(" / ", "_") if "/" in var else var
                plot_filename = os.path.join(plot_dir, f"{var_name}.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close()

        print(f"\t\tSaved {plot_type} plots for covariates.")
        # Ensure all figures are closed at the end of the analysis
        plt.close('all')
            
        
        
if __name__ == "__main__":
    cohort = time2event_cfg.COHORT
    output_dir = time2event_cfg.OUTPUT_DIR
    stratification_vars = time2event_cfg.STRATIFICATION_VARS
    data_path = time2event_cfg.COHORT_DATAFRAME
    baseline_vars = time2event_cfg.BASELINE_VARS
    event_col = time2event_cfg.EVENT_COL
    duration_col = time2event_cfg.DURATION_COL
    
    time2event = Time2Event(data_path)
    time2event.time_to_event_analysis(prefix=cohort, output_dir=output_dir, variables=baseline_vars, duration_col=duration_col, event_col=event_col, stratify_by=stratification_vars, progression_type="composite")
    #time2event.time_to_event_analysis(prefix=cohort, output_dir=output_dir, variables=baseline_vars, duration_col=duration_col, event_col=event_col, stratify_by=stratification_vars, progression_type="volumetric")