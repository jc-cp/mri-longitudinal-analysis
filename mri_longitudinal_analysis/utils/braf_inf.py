"""
Creates a basic script that outputs the necessary csv-file for the BRAF inference.
"""
import os
import csv

def part1(directory, csv_file, header):
    """
    Out of the directory, create a csv file with the patient_id, scan_id and label. Label is just a placeholder for the real inference pipeline.
    """
    # Open the CSV file for writing
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        # Iterate over files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".nii.gz") and "_mask" not in filename:
                # Extract patient ID and scan ID from the filename
                parts = filename.split("_")
                patient_id = parts[1]
                scan_id = parts[2].split(".")[0]

                # Write the information to the CSV file
                csvwriter.writerow([patient_id, scan_id, "3"])

    print(f"Data written to {csv}")


def part2(file_list):
    """
    Evaluate the results of the inference.
    """
    print("Evaluating the results of the inference...")
    
    for file in file_list:
        patient_data = {}
        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader) 

            for row in csvreader:
                patient_id, scan_id, label = row[0], row[1], int(row[2])
                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'scans': [],
                        'labels': []
                    }
                patient_data[patient_id]['scans'].append(scan_id)
                patient_data[patient_id]['labels'].append(label)
            
            for patient_id, data in patient_data.items():
                label_counts = {0: 0, 1: 0, 2: 0}
                for label in data['labels']:
                    label_counts[label] += 1
                final_label = max(label_counts, key=label_counts.get)
                data['final_label'] = final_label
                print(f"Patient ID: {patient_id}, Final Label: {final_label}")

            # Write the final results to new csv
            with open(file, "w", newline="", encoding="utf-8") as csvfile:
                csvwriter = csv.writer(csvfile)
                new_header = header + ['Final Label', 'Description']
                csvwriter.writerow(new_header)
                description = {0: "V600E", 1: "Fusion", 2: "Wildtype"}
                for patient_id, data in patient_data.items():
                    descrip = description[data['final_label']]
                    for scan_id, label in zip(data['scans'], data['labels']):
                        row = [patient_id, scan_id, label, data['final_label'], descrip]
                        csvwriter.writerow(row)

            print("Final results written to final_results.csv")


def calculate_metrics(file_list):
    """
    Calculate metrics for the predictions.
    """
    for file in file_list:
        if file.endswith("bch.csv"):
            cohort = "BCH"
        else:
            cohort = "CBTN"

        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0

            for row in csvreader:
                _, _, label, final_label = row[0], row[1], int(row[2]), int(row[3])
                if final_label == label:
                    if final_label == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if final_label == 1:
                        false_positives += 1
                    else:
                        false_negatives += 1

            # Calculate additional metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Print the metrics for the cohort
            print(f"Metrics for {cohort}:")
            print(f"True Positives: {true_positives}")
            print(f"True Negatives: {true_negatives}")
            print(f"False Positives: {false_positives}")
            print(f"False Negatives: {false_negatives}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"F1 Score: {f1_score:.2f}")


def main():
    """
    part1: creates the csv files for inference
    part2: evaluate the results after inference
    """
    PART1 = False
    PART2 = True

    # Directory containing the files and the CSV file output
    img_dir = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/final_dataset"
    csv_file = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/final_patients_all_scans.csv"
    header = ["pat_id", "scandate", "label"]
    if PART1:
        part1(img_dir, csv_file, header)


    bch_csv = "/home/juanqui55/git/mri-longitudinal-analysis/data/input/clinical/final_final_classification_bch.csv"
    cbtn_csv = "/home/juanqui55/git/mri-longitudinal-analysis/data/input/clinical/final_final_classification_cbtn.csv"
    file_list = [bch_csv, cbtn_csv]
    header = ["pat_id",	"scandate",	"FinalPrediction"]

    if PART2:
        part2(file_list)
        calculate_metrics(file_list)


if __name__ == "__main__":
    main()
