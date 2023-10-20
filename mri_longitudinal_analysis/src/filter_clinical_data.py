"""Module for processing and visualizing clinical data."""

import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cfg import filter_clinical_data_cfg


class ClinicalData:
    """Class to handle operations related to clinical data."""

    def __init__(self, file_path):
        """Initialize a new ClinicalData object.

        Args:
            file_path (str): The path to the clinical data CSV file.
        """
        self.file_path = file_path

        os.makedirs(filter_clinical_data_cfg.OUTPUT_DIR, exist_ok=True)

        self.treatment_plot = filter_clinical_data_cfg.TREATMENT_PLOT
        self.mutation_plot = filter_clinical_data_cfg.MUTATION_PLOT
        self.diagnosis_plot = filter_clinical_data_cfg.DIAGNOSIS_PLOT

        self.output_file = filter_clinical_data_cfg.OUTPUT_FILE
        self.delete_post_op_data = filter_clinical_data_cfg.REMOVING_SURGERY
        self.visualization = filter_clinical_data_cfg.VISUALIZE_DATA

    def load_file(self) -> pd.DataFrame:
        """Load clinical data from a CSV file.

        Returns:
            pd.DataFrame or None: The loaded DataFrame if successful, or None otherwise.
        """
        try:
            d_f = pd.read_csv(self.file_path, encoding="utf-8")
            # print(d_f.head())
            return d_f
        except FileNotFoundError:
            print("File not found. Please check the file path.")
            return sys.exit(1)
        except pd.errors.ParserError as excp:
            print("An unexpected error occurred.", excp)
            return sys.exit(1)

    def parse_data(self, d_f) -> dict:
        """Parse patient data from a DataFrame.

        Args:
            d_f (pd.DataFrame): The DataFrame containing the clinical data.

        Returns:
            dict: A dictionary containing the parsed data.
        """
        # create a dictionary to hold the parsed data
        patient_data = {}

        for index, row in d_f.iterrows():
            patient_id = row["BCH MRN"]
            dob = row["Date of Birth"]
            clinical_status = row["Clinical status at last follow-up"]
            pathologic_diagnosis = row["Pathologic diagnosis"]
            sex = row["Sex"]

            # Check if patient_id is valid
            if pd.isnull(patient_id):
                print(f"Patient with index {index} has no ID!")
                continue

            # Add leading 0 if patient_id length is only 6 digits
            if len(str(patient_id)) == 6:
                patient_id = "0" + str(patient_id)

            # Initialize patient data
            patient_data[patient_id] = {}

            # Determine BRAF mutation status
            braf_v600e = row["BRAF V600E mutation"]
            braf_fusion = row["BRAF fusion"]

            if braf_v600e == "Yes" or braf_fusion == "Yes":
                braf_mutation_status = "Mutated"
            else:
                braf_mutation_status = "Wildtype"

            patient_data[patient_id] = {
                "DOB": dob,
                "Sex": sex,
                "Clinical Status": clinical_status,
                "BRAF Mutation Status": braf_mutation_status,
            }

            # Handle 'Pathologic diagnosis'
            if "optic" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Optic Glioma"
            elif "astrocytoma" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Astrocytoma"
            elif "tectal" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Tectal Glioma"
            elif "DNET" in pathologic_diagnosis.upper():
                patient_data[patient_id]["Pathologic diagnosis"] = "DNET"
            elif "ganglioglioma" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Ganglioglioma"
            elif "glioneuronal" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Glioneuronal Neoplasm"
            elif "xanthoastrocytoma" in pathologic_diagnosis.lower():
                patient_data[patient_id]["Pathologic diagnosis"] = "Pleomorphic Xanthoastrocytoma"
            elif (
                "low grade glioma" in pathologic_diagnosis.lower()
                or "low-grade glioma" in pathologic_diagnosis.lower()
            ):
                patient_data[patient_id]["Pathologic diagnosis"] = "Plain Low Grade Glioma"
            else:
                patient_data[patient_id]["Pathologic diagnosis"] = "Other"

            if row["Surgical Resection"] == "Yes":
                patient_data[patient_id]["Surgery"] = "Yes"
                patient_data[patient_id]["Date of first surgery"] = row["Date of first surgery"]

            if row["Systemic therapy before radiation"] == "Yes":
                patient_data[patient_id]["Chemotherapy"] = "Yes"
                patient_data[patient_id]["Date of Systemic Therapy Start"] = row[
                    "Date of Systemic Therapy Start"
                ]

            if row["Radiation as part of initial treatment"] == "Yes":
                patient_data[patient_id]["Radiation"] = "Yes"
                patient_data[patient_id]["Start Date of Radiation"] = row["Start Date of Radiation"]

        return patient_data

    def visualize_data(self, data):
        """Visualize the clinical data.

        Args:
            data (dict): A dictionary containing the parsed data.
        """
        if self.treatment_plot:
            # Initialize counts
            counts = {
                "No Treatment": 0,
                "Surgery Only": 0,
                "Chemotherapy Only": 0,
                "Radiation Only": 0,
                "Surgery and Chemotherapy": 0,
                "Surgery and Radiation": 0,
                "Chemotherapy and Radiation": 0,
                "All Treatments": 0,
            }

            # Count occurrences
            for _, patient_info in data.items():  # omitted is the patient_id
                treatments = [
                    t
                    for t in ["Surgery", "Chemotherapy", "Radiation"]
                    if patient_info.get(t) == "Yes"
                ]
                if self.should_count_as_no_treatment(patient_info):
                    counts["No Treatment"] += 1
                elif len(treatments) == 1:
                    counts[f"{treatments[0]} Only"] += 1
                elif len(treatments) == 2:
                    counts[f"{treatments[0]} and {treatments[1]}"] += 1
                elif len(treatments) == 3:
                    counts["All Treatments"] += 1

            # Plot
            plt.figure(figsize=(14, 10))
            plt.grid(axis="y")
            plt.rcParams.update({"font.size": 12})

            bars = plt.bar(
                range(len(counts)),
                list(counts.values()),
                align="center",
                color="skyblue",
            )
            plt.xticks(range(len(counts)), list(counts.keys()), rotation="vertical")

            for bar_ in bars:
                yval = bar_.get_height()
                plt.text(
                    bar_.get_x() + bar_.get_width() / 2,
                    yval + 0.1,
                    yval,
                    ha="center",
                    va="bottom",
                )

            plt.title("Treatment Counts")
            plt.xlabel("Treatment Type")
            plt.ylabel("Number of Patients")

            plt.tight_layout()

            plt.savefig(filter_clinical_data_cfg.OUTPUT_TREATMENT)

        if self.mutation_plot:
            # Data for additional plots
            sexes = [info["Sex"] for info in data.values() if "Sex" in info]
            mutations = [
                info["BRAF Mutation Status"]
                for info in data.values()
                if "BRAF Mutation Status" in info
            ]

            # Plot Mutation Distribution by Sex
            plt.figure(figsize=(10, 6))
            plt.rcParams.update({"font.size": 10})
            a_x1 = sns.countplot(x=sexes, hue=mutations)
            plt.title("Mutation Distribution by Sex")
            plt.xlabel("Sex")
            plt.ylabel("Count")
            self.annotate_plot(a_x1)
            plt.legend(title="BRAF Mutation Status")
            plt.tight_layout()
            plt.savefig(filter_clinical_data_cfg.OUTPUT_MUTATION)

        if self.diagnosis_plot:
            diagnoses = [
                info["Pathologic diagnosis"]
                for info in data.values()
                if "Pathologic diagnosis" in info
            ]

            # Add a check here to see if the list is empty
            if not diagnoses:
                print("No diagnoses data found.")
            else:
                # Plot Pathologic Diagnosis Distribution
                plt.figure(figsize=(10, 6))
                plt.rcParams.update({"font.size": 10})
                a_x2 = sns.countplot(x=diagnoses)
                plt.title("Pathologic Diagnosis Distribution")
                plt.xlabel("Diagnosis")
                plt.ylabel("Count")
                plt.xticks(rotation=90)
                self.annotate_plot(a_x2)
                plt.tight_layout()
                plt.savefig(filter_clinical_data_cfg.OUTPUT_DIAGNOSIS)

    def annotate_plot(self, a_x):
        """Annotate the bar plot with the respective heights.

        Args:
            a_x (matplotlib.axis): The axis object to be annotated.
        """
        for patch in a_x.patches:
            height = patch.get_height()
            a_x.text(
                x=patch.get_x() + (patch.get_width() / 2),
                y=height,
                s=f"{height:.0f}",
                ha="center",
            )

    def write_dict_to_file(self, data_dict, filename):
        """Write a dictionary to a file.

        Args:
            data_dict (dict): The dictionary to write.
            filename (str): The name of the output file.
        """
        with open(filename, "w", encoding="utf-8") as file:
            for patient_id, patient_info in data_dict.items():
                file.write(f"Patient ID: {patient_id}\n")
                for key, value in patient_info.items():
                    file.write(f"\t{key}: {value}\n")
                file.write("\n")

    def move_post_surgery_files(self, patient_data, directory):
        """Moves the files related to post-surgery/ treatment cases to a folder for separate analysis.

        Args:
            patient_data (dict): A dictionary containing patient data.
            directory (str): The directory where the files are located.
        """
        post_treatment_folder = os.path.join(directory, "post_treatment_files")
        os.makedirs(post_treatment_folder, exist_ok=True)
        counter = 0
        for patient_id in patient_data:
            treatment_type = None
            treatment_date_str = None

            if "Surgery" in patient_data[patient_id]:
                treatment_type = "Surgery"
                treatment_date_str = patient_data[patient_id]["Date of first surgery"]

            elif "Chemotherapy" in patient_data[patient_id]:
                treatment_type = "Chemotherapy"
                treatment_date_str = patient_data[patient_id]["Date of Systemic Therapy Start"]

            elif "Radiation" in patient_data[patient_id]:
                treatment_type = "Radiation"
                treatment_date_str = patient_data[patient_id]["Start Date of Radiation"]

            if treatment_type and treatment_date_str and isinstance(treatment_date_str, str):
                try:
                    # Attempt to parse the date in the expected format
                    treatment_date = datetime.strptime(treatment_date_str, "%d/%m/%Y")
                except ValueError:
                    # If the above fails, try to parse in the 'day/month/year' format
                    treatment_date = datetime.strptime(treatment_date_str, "%Y-%m-%d")

                patient_folder = os.path.join(post_treatment_folder, str(patient_id))
                os.makedirs(patient_folder, exist_ok=True)

                for filename in os.listdir(directory):
                    if str(patient_id) in filename:
                        file_path = os.path.join(directory, filename)
                        file_date_str = filename.split("_")[1].split(".")[0]
                        file_date = datetime.strptime(file_date_str, "%Y%m%d")

                        if file_date > treatment_date:
                            print(filename)
                            # shutil.move(
                            #    file_path,
                            #    os.path.join(patient_folder, filename),
                            # )
            if self.should_count_as_no_treatment(patient_data[patient_id]):
                counter += 1
                print(counter)
                print(f"No treatment data found for patient {patient_id}.")

    def check_treatment_type(self, patient_info):
        return any(
            treatment in patient_info for treatment in ["Surgery", "Chemotherapy", "Radiation"]
        )

    def check_treatment_date(self, patient_info):
        return any(
            key in patient_info
            and patient_info[key] is not None
            and isinstance(patient_info[key], str)
            for key in [
                "Date of first surgery",
                "Date of Systemic Therapy Start",
                "Start Date of Radiation",
            ]
        )

    def should_count_as_no_treatment(self, patient_info):
        return not (
            self.check_treatment_type(patient_info) and self.check_treatment_date(patient_info)
        )

    def main(self):
        """Main function to handle loading, parsing, and possibly visualizing the clinical data."""
        d_f = self.load_file()
        patient_data = self.parse_data(d_f)

        if self.visualization:
            self.visualize_data(patient_data)

        if self.output_file:
            self.write_dict_to_file(patient_data, filter_clinical_data_cfg.OUTPUT_FILE_NAME)

        if self.delete_post_op_data:
            self.move_post_surgery_files(patient_data, filter_clinical_data_cfg.DATA_DIR)


if __name__ == "__main__":
    cd = ClinicalData(file_path=filter_clinical_data_cfg.CSV_FILE)
    cd.main()
