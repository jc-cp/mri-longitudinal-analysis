#%%
"""
Trajectory Plotting and Classifcation Analysis
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from cfg.src import trajectories_cfg
from cfg.utils import helper_functions_cfg
from utils.helper_functions import classify_patient_volumetric, classify_patient_composite, calculate_progression, plot_histo_distributions

class TrajectoryClassification:
    def __init__(self, path_to_data, variables, cohort, sample_size):
        df = pd.read_csv(path_to_data)
        self.data = df   
        self.category_list = variables     
        self.cohort = cohort
        self.sample_size_plots = sample_size
 
    def trajectories(self, output_dir):
        """
        Plot trajectories of patients.

        Parameters:
        - output_dir (str): Directory to save the output plots.
        """
        print("Step 0: Modeling growth trajectories:")
        # Data preparation for modeling
        self.data.sort_values(by=["Patient_ID", "Age"], inplace=True)
        self.data["Time since First Scan"] = self.data.groupby(
            "Patient_ID"
        )["Age"].transform(lambda x: (x - x.iloc[0]))
        
        self.data.reset_index(drop=True, inplace=True)

        # Error handling for sample size
        sample_size = self.sample_size_plots
        if sample_size:
            # Sample a subset of patients if sample_size is provided
            unique_patient_count = self.data["Patient_ID"].nunique()
            if sample_size > unique_patient_count:
                print(
                    f"\t\tSample size {sample_size} is greater than the number of unique patients"
                    f" {unique_patient_count}. Using {unique_patient_count} instead."
                )
                sample_size = unique_patient_count

            sample_ids = (
                self.data["Patient_ID"].drop_duplicates().sample(n=sample_size)
            )
            self.data = self.data[
                self.data["Patient_ID"].isin(sample_ids)
            ]

        dir_name = os.path.join(output_dir, "base_trajectories")
        os.makedirs(dir_name, exist_ok=True)
        # Plot the overlaying curves plots
        volume_change_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_volume_change_trajectories_plot.png"
        )
        self.plot_individual_trajectories(
            volume_change_trajectories_plot,
            plot_data=self.data,
            column="Volume Change",
            unit="mm^3",
        )
        volume_change_pct_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_volume_change_pct_trajectories_plot.png")
        self.plot_individual_trajectories(
            volume_change_pct_trajectories_plot,
            plot_data=self.data,
            column="Volume Change Pct",
            unit="%",)
        volume_change_rate_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_volume_change_rate_trajectories_plot.png"
        )
        self.plot_individual_trajectories(
            volume_change_rate_trajectories_plot,
            plot_data=self.data,
            column="Volume Change Rate",
            unit="mm^3 / day",
        )
        volume_change_rate_pct_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_volume_change_rate_pct_trajectories_plot.png")
        self.plot_individual_trajectories(
            volume_change_rate_pct_trajectories_plot,
            plot_data=self.data,
            column="Volume Change Rate Pct",
            unit="% / day")
        normalized_volume_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_normalized_volume_trajectories_plot.png"
        )
        self.plot_individual_trajectories(
            normalized_volume_trajectories_plot,
            plot_data=self.data,
            column="Normalized Volume",
            unit="mm^3",
        )
        volume_trajectories_plot = os.path.join(
            dir_name, f"{self.cohort}_volume_trajectories_plot.png"
        )
        self.plot_individual_trajectories(
            volume_trajectories_plot,
            plot_data=self.data,
            column="Volume",
            unit="mm^3",
        )

        category_out = os.path.join(output_dir, "stratified_trajectories")
        os.makedirs(category_out, exist_ok=True)

        for cat in self.category_list:
            cat_volume_change_name = os.path.join(
                category_out, f"{self.cohort}_{cat}_volume_change_trajectories_plot.png"
            )
            self.plot_individual_trajectories(
                cat_volume_change_name,
                plot_data=self.data,
                column="Volume Change",
                category_column=cat,
                unit="%",
            )
            cat_normalized_volume_name = os.path.join(
                category_out, f"{self.cohort}_{cat}_normalized_volume_trajectories_plot.png"
            )
            self.plot_individual_trajectories(
                cat_normalized_volume_name,
                plot_data=self.data,
                column="Normalized Volume",
                category_column=cat,
                unit="mm^3",
            )
            cat_volume_change_rate_trajectories_plot = os.path.join(
                category_out, f"{self.cohort}_{cat}_volume_change_rate_trajectories_plot.png"
            )
            self.plot_individual_trajectories(
                cat_volume_change_rate_trajectories_plot,
                plot_data=self.data,
                column="Volume Change Rate",
                category_column=cat,
                unit="% / day",
            )

        return self.data 

    def get_progression_data(self, progression_threshold, regression_threshold, volume_change_threshold, time_period_mapping):
        self.data.sort_values(by=["Patient_ID", "Age"], inplace=True)
        progression_data = self.data.groupby("Patient_ID").apply(lambda x:
            calculate_progression(x, progression_threshold, regression_threshold, volume_change_threshold, time_period_mapping)
        )
        progression_data = progression_data.reset_index(drop=False)
        # consider on the one side the merging back to the first data frame and continue with the analysis dataframe at the same time
        self.data = pd.merge(
            self.data, progression_data, on="Patient_ID", how="left"
        )
        self.data["Time Since Diagnosis"] = self.data["Time Since Diagnosis"].astype("category") # catefory version of Time Since First Scan
        
    def classification_analysis(self, data, output_dir, column_name="Normalized Volume"):
        """
        Classify patients based on their tumor growth trajectories
        into progressors, stable or regressors.
        """
        print("Step 1: Starting Classification Analysis:")
        patients_ids = data["Patient_ID"].unique()
        patient_classifications_volumetric = {
            patient_id: classify_patient_volumetric(
                data,
                patient_id,
            )
            for patient_id in patients_ids
        }
        patient_classifications_composite = {
            patient_id: classify_patient_composite(
                data,
                patient_id,
            )
            for patient_id in patients_ids
        }
        
        data["Classification Volumetric"] = data["Patient_ID"].map(patient_classifications_volumetric)
        data["Classification Composite"] = data["Patient_ID"].map(patient_classifications_composite)

        # Add classifications to the data DataFrame
        data["Classification Volumetric"] = data["Patient_ID"].map(patient_classifications_volumetric)
        data["Classification Composite"] = data["Patient_ID"].map(patient_classifications_composite)
        
        # Create binary classifications
        data["Patient Classification Binary Volumetric"] = (
            data["Classification Volumetric"].apply(lambda x: 1 if x == "Progressor" else 0)
        )
        data["Patient Classification Binary Composite"] = (
            data["Classification Composite"].apply(lambda x: 1 if x == "Progressor" else 0)
        )
        
        # Save classifications to self.data
        self.data["Patient Classification Volumetric"] = (
            self.data["Patient_ID"].map(patient_classifications_volumetric).astype("category")
        )
        self.data["Patient Classification Composite"] = (
            self.data["Patient_ID"].map(patient_classifications_composite).astype("category")
        )
        self.data["Patient Classification Binary Volumetric"] = (
            self.data["Patient_ID"].map(data.groupby("Patient_ID")["Patient Classification Binary Volumetric"].first())
        )
        self.data["Patient Classification Binary Composite"] = (
            self.data["Patient_ID"].map(data.groupby("Patient_ID")["Patient Classification Binary Composite"].first())
        )
        # Plots
        dir_name = os.path.join(output_dir, "classification")
        os.makedirs(dir_name, exist_ok=True)
        self.plot_classification_trajectories(data, dir_name, self.cohort, column_name, progression_type="volumetric", unit="mm^3")
        self.plot_classification_trajectories(data, dir_name, self.cohort, column_name, progression_type="composite", unit="mm^3")
        unique_pat = self.data.drop_duplicates(subset=["Patient_ID"])

        print(unique_pat["Patient Classification Volumetric"].value_counts())
        print(unique_pat["Patient Classification Composite"].value_counts())
        print("\tSaved classification analysis plot.")
        self.plot_classification_bars(data, dir_name)
        print("\tSaved classification bars plot.")

    ############################## Plotting Functions ##############################
    
    def plot_individual_trajectories(self, name, plot_data, column, category_column=None, unit=None, time_limit=4000, median_freq=273
    ):
        """
        Plot the individual volume trajectories for a sample of patients.

        Parameters:
        - name (str): The filename for the saved plot image.
        - plot_data (DataFrame): The data to be plotted.
        - column (str): The name of the column representing volume to be plotted.
        - output_dir (str): Directory where the plot image will be saved.
        - time_limit (int): Cutoff time in days for plotting data.
        - freq_days (int): Frequency in days for calculating median trajectories.
        """
        plt.figure(figsize=(10, 8))
        
        if column in ["Normalized Volume","Volume Change", "Volume Change Rate", "Volume Change Pct"]:
            mean = np.mean(plot_data[column])
            std = np.std(plot_data[column])
            if column in ["Normalized Volume", "Volume Change", "Volume Change Pct"]:
                factor = 2.5
            elif column in ["Volume Change Rate"]:
                factor = 0.25
            threshold = mean + factor * std
            plot_data = plot_data[plot_data[column] <= threshold]
            plot_data = plot_data[plot_data[column] >= -threshold]

        plot_data = plot_data[plot_data["Time since First Scan"] <= time_limit]
        num_patients = plot_data["Patient_ID"].nunique()    
        max_time = plot_data["Time since First Scan"].max()    
        # Get the median every 3 months
        median_data = (
            plot_data.groupby(
                pd.cut(
                    plot_data["Time since First Scan"],
                    pd.interval_range(
                        start=0, end=max_time, freq=median_freq,
                    ),
                ),
                observed=True,
            )[column]
            .median()
            .reset_index()
        )

        if category_column:
            categories = plot_data[category_column].unique()
            patient_palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(categories))
            median_palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, len(categories))
            legend_handles = []

            median_lines = False
            for (category, patient_color), median_color in zip(
                zip(categories, patient_palette), median_palette
            ):
                category_data = plot_data[plot_data[category_column] == category]
                if median_lines:
                    median_data_category = (
                        category_data.groupby(
                            pd.cut(
                                category_data["Time since First Scan"],
                                pd.interval_range(
                                    start=0,
                                    end=max_time,
                                    freq=median_freq,
                                ),
                            ), 
                            observed=True,
                        )[column]
                        .median()
                        .reset_index()
                    )
                legend_handles.append(
                    lines.Line2D([], [], color=patient_color, label=f"{category_column} {category}")
                )
                for patient_id in category_data["Patient_ID"].unique():
                    patient_data = category_data[category_data["Patient_ID"] == patient_id]
                    plt.plot(
                        patient_data["Time since First Scan"],
                        patient_data[column],
                        color=patient_color,
                        alpha=0.5,
                        linewidth=1,
                    )
                if median_lines:
                    sns.lineplot(
                        x=median_data_category["Time since First Scan"].apply(lambda x: x.mid),
                        y=column,
                        data=median_data_category,
                        color=median_color,
                        linestyle="--",
                        label=f"{category_column} {category} Median Trajectory",
                    )

            sns.lineplot(
                x=median_data["Time since First Scan"].apply(lambda x: x.mid),
                y=column,
                data=median_data,
                color="blue",
                linestyle="--",
                label="Cohort Median Trajectory",
            )
            # Retrieve the handles and labels from the current plot
            handles, _ = plt.gca().get_legend_handles_labels()
            # Combine custom category handles with the median trajectory handles
            combined_handles = legend_handles + handles[-(len(categories) + 1) :]

            plt.title(f"{column} Trajectories by {category_column} (N={num_patients})", fontdict={"size": 18})
            plt.legend(handles=combined_handles)

        else:
            # Plot each patient's data
            for patient_id in plot_data["Patient_ID"].unique():
                patient_data = plot_data[plot_data["Patient_ID"] == patient_id]
                sns.set_palette(helper_functions_cfg.NORD_PALETTE)
                plt.plot(
                    patient_data["Time since First Scan"],
                    patient_data[column],
                    alpha=0.5,
                    linewidth=1,
                )

            sns.lineplot(
                x=median_data["Time since First Scan"].apply(lambda x: x.mid),
                y=column,
                data=median_data,
                color="blue",
                linestyle="--",
                label="Median Trajectory",
            )
            plt.title(f"Individual Tumor {column} Trajectories (N={num_patients})")
            plt.legend()

        plt.xlabel("Days Since First Scan", fontdict={"size": 15})
        plt.ylabel(f"Tumor {column} [{unit}]", fontdict={"size": 15})
        plt.savefig(name, dpi=300)
        plt.close()
        if category_column:
            print(f"\t\tSaved tumor {column} trajectories plot by category: {category_column}.")
        else:
            print(f"\t\tSaved tumor {column} trajectories plot for all patients.")

    def plot_classification_trajectories(self, data, output_dir, prefix, column_name, progression_type, unit=None):
        """
        Plot the growth trajectories of patients with classifications.

        Parameters:
        - data: DataFrame containing patient growth data and classifications.
        - output_filename: Name of the file to save the plot.
        """
        results = {}
        plt.figure(figsize=(10, 8))
        if column_name == "Normalized Volume":
            mean = np.mean(data[column_name])
            std = np.std(data[column_name])
            factor = 2.5
            threshold = mean + factor * std
            data = data[data[column_name] <= threshold]
            data = data[data[column_name] >= -threshold]
        # Unique classifications & palette
        data = data[data["Time since First Scan"] <= 4000]
        palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE)
        if progression_type == "volumetric":
            classification_type = "Classification Volumetric"
            colors = [palette[0],palette[1], "green"]
        elif progression_type == "composite":
            classification_type = "Classification Composite"
            colors = [palette[0],palette[1]]
        results[classification_type] = {}
        classifications = data[classification_type].unique()
        for classification, color in zip(classifications, colors):
            class_data = data[data[classification_type] == classification]
            first_patient_plotted = False

            # Plot individual trajectories
            for patient_id in class_data["Patient_ID"].unique():
                patient_data = class_data[class_data["Patient_ID"] == patient_id]

                if classification is not None:
                    plt.plot(
                        patient_data["Time since First Scan"],
                        patient_data[column_name],
                        color=color,
                        alpha=0.5,
                        linewidth=1,
                        label=classification if not first_patient_plotted else "",
                    )
                first_patient_plotted = True

            if classification is not None:
                # Plot median trajectory for each classification
                median_data = (
                    class_data.groupby(
                        pd.cut(
                            class_data["Time since First Scan"],
                            pd.interval_range(
                                start=0, end=class_data["Time since First Scan"].max(), freq=365, 
                            ),
                        ), 
                        observed=True,
                    )[column_name]
                    .median()
                    .reset_index()
                )
                sns.lineplot(
                    x=median_data["Time since First Scan"].apply(lambda x: x.mid),
                    y=column_name,
                    data=median_data,
                    color=color,
                    linestyle="--",
                    label=f"{classification} Median",
                    linewidth=1.5,
                )

            volume_changes = class_data['Volume Change']
            volume_changes_pct = class_data['Volume Change Pct']
            
            results[classification_type][classification] = {
                'median_volume_change': volume_changes.median(),
                'mean_volume_change': volume_changes.mean(),
                'std_volume_change': volume_changes.std(),
                'median_volume_change_pct': volume_changes_pct.median(),
                'mean_volume_change_pct': volume_changes_pct.mean(),
                'std_volume_change_pct': volume_changes_pct.std(),
                'n_measurements': len(class_data),
                'n_patients': class_data['Patient_ID'].nunique(),
                'n_zero_changes': (volume_changes == 0).sum(),
                'n_non_zero_changes': (volume_changes != 0).sum(),
                'min_non_zero_change': volume_changes[volume_changes != 0].min() if (volume_changes != 0).any() else None,
                'max_non_zero_change': volume_changes[volume_changes != 0].max() if (volume_changes != 0).any() else None,
            }
        print(f"Detailed results for {classification_type}:")
        for classification, stats in results[classification_type].items():
            print(f"\n{classification}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        num_patients = data["Patient_ID"].nunique()
        plt.axhline(y=0.75, color='blue', linestyle="-", label="-25% Volume Change")
        plt.axhline(y=1.25, color='red', linestyle="-", label="+25% Volume Change")
        plt.xlabel("Days Since First Scan", fontdict={"size": 15})
        plt.ylabel(f"Tumor {column_name} [{unit}]", fontdict={"size": 15})
        plt.title(f"Patient Classification Trajectories (N={num_patients})", fontdict={"size": 20})
        plt.legend()
        output_filename = os.path.join(output_dir, f"{prefix}_classification_analysis_{progression_type}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close()

    def plot_classification_bars(self, data, output_dir):
        fig, ax = plt.subplots(figsize=(9, 6))
        palette = sns.color_palette(helper_functions_cfg.NORD_PALETTE, 3)
        
        # Prepare data for the horizontal bar plot
        classifications = [
            ("Patient Classification Volumetric", "Volumetric"),
            ("Patient Classification Composite", "Clinical"),
            ("Received Treatment", "Treatment"),
        ]
        
        y_positions = [2, 1, 0]  # Positions for the bars
        colors = [[palette[1], palette[2], palette[0]], [palette[1], palette[0]], [palette[0], palette[1]]]  # Colors for each bar
        
        for (col, label), y_pos, color_set in zip(classifications, y_positions, colors):
            counts = data.drop_duplicates('Patient_ID')[col].value_counts()
            total_pats = len(data['Patient_ID'].unique())
            percentages = counts / total_pats * 100
            
            left = 0
            for category, percentage in percentages.items():
                ax.barh(y_pos, percentage, left=left, height=0.5, 
                        color=color_set[counts.index.get_loc(category) % len(color_set)], 
                        label=f"{category} ({percentage:.1f}%)")
                ax.text(left + percentage/2, y_pos, f"{category}\n{percentage:.1f}%", 
                        ha='center', va='center', color='white', fontweight='bold', fontsize=14)
                left += percentage
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in classifications], fontsize=15)
        ax.set_xlabel("Percentage", fontsize=15)
        ax.set_title("Patient Classifications and Treatment Distribution", fontsize=20)
        
        plt.tight_layout()
        
        # Display the plot
        file_name = os.path.join(output_dir, "classification_bars.png")
        plt.savefig(file_name, dpi=300)
        plt.close(fig)   

    # TODO: Implement the following method
    def analyze_tumor_stability(
        self,
        data,
        output_dir,
        #volume_weight=0.5,
        #growth_weight=0.5,
        #change_threshold=20,
    ):
        pass
        # """
        # Analyze the stability of tumors based on their growth rates and volume changes.

        # Parameters:
        # - data (DataFrame): Data containing tumor growth and volume information.
        # - output_dir (str): Directory to save the output plots.
        # - volume_weight (float): Clinical significance weight for tumor volume stability.
        # - growth_weight (float): Clinical significance weight for tumor growth stability.

        # Returns:
        # - data (DataFrame): Data with added Stability Index and Tumor Classification.
        # """
        # print("\tAnalyzing tumor stability:")
        # data = data.copy()
        # volume_column = "Normalized Volume"
        # volume_change_column = "Volume Change"
        # data = calculate_group_norms_and_stability(
        #     data, volume_column, volume_change_column
        # )
        # # Calculate the overall volume change for each patient
        # data["Overall Volume Change"] = data["Patient_ID"].apply(
        #     lambda x: calculate_percentage_change(data, x, volume_column)
        # )
        # # Calculate the Stability Index using weighted scores
        # data["Stability Index"] = (
        #     volume_weight * data["Volume Stability Score"]
        #     + growth_weight * data["Change Stability Score"]
        # )

        # # Normalize the Stability Index to have a mean of 1
        # data["Stability Index"] /= np.mean(data["Stability Index"])

        # significant_volume_change = (
        #     abs(data["Overall Volume Change"]) >= change_threshold
        # )
        # stable_subset = data.loc[~significant_volume_change, "Stability Index"]
        # mean_stability_index = stable_subset.mean()
        # std_stability_index = stable_subset.std()
        # num_std_dev = 2
        # stability_threshold = mean_stability_index + (num_std_dev * std_stability_index)

        # data["Tumor Classification"] = data.apply(
        #     lambda row: "Unstable"
        #     if abs(row["Overall Volume Change"]) >= change_threshold
        #     or row["Stability Index"] > stability_threshold
        #     else "Stable",
        #     axis=1,
        # ).astype("category")

        # tumor_stability_out = os.path.join(output_dir, "tumor_stability_plots")
        # os.makedirs(tumor_stability_out, exist_ok=True)
        # data_n = calculate_stability_index(data)
        # visualize_individual_indexes(data_n, tumor_stability_out)
        # visualize_stability_index(data_n, tumor_stability_out)
        # visualize_volume_change(data_n, tumor_stability_out)
        # visualize_ind_indexes_distrib(data_n, tumor_stability_out)
        # roc_curve_and_auc(data_n, tumor_stability_out)
        #grid_search_weights(data_n)
        # Map the 'Stability Index' and 'Tumor Classification' to the
        # self.data using the maps
        # m_data = pd.merge(
        #     self.data,
        #     data[
        #         [
        #             "Patient_ID",
        #             "Age",
        #             "Stability Index",
        #             "Tumor Classification",
        #             "Overall Volume Change",
        #         ]
        #     ],
        #     on=["Patient_ID", "Age"],
        #     how="left",
        # )

        # self.data = m_data
        # self.data.reset_index(drop=True, inplace=True)
        # visualize_tumor_stability(
        #     data, output_dir, stability_threshold, change_threshold
        # )
        # print("\t\tSaved tumor stability plots.")


if __name__ == "__main__":
    path_to_data = trajectories_cfg.COHORT_DATAFRAME
    variables = trajectories_cfg.CURVE_VARS
    output_dir = trajectories_cfg.OUTPUT_DIR
    cohort = trajectories_cfg.COHORT
    sample_size = trajectories_cfg.SAMPLE_SIZE
    progression_threshold = trajectories_cfg.PROGRESSION_THRESHOLD
    regression_threshold = trajectories_cfg.REGRESSION_THRESHOLD
    volume_change_threshold = trajectories_cfg.CHANGE_THRESHOLD
    time_period_mapping = trajectories_cfg.TIME_PERIOD_MAPPING
    list_time_periods = list(time_period_mapping.keys())
    age_groups = trajectories_cfg.AGE_GROUPS
    
    os.makedirs(output_dir, exist_ok=True)
    traj = TrajectoryClassification(path_to_data, variables, cohort, sample_size)
    traj.trajectories(output_dir)
    traj.get_progression_data(progression_threshold, regression_threshold, volume_change_threshold, time_period_mapping)
    traj.classification_analysis(traj.data, output_dir)
    plot_histo_distributions(traj.data, output_dir, list_time_periods, age_groups, endpoint="volumetric")
    plot_histo_distributions(traj.data, output_dir, list_time_periods, age_groups, endpoint="composite")
# %%
