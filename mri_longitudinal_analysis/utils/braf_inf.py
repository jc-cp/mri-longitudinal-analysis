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

def part2(file_list, cohort_dict1, cohort_dict2, output_dir):
    """
    Evaluate the results of the inference.
    """
    print("Evaluating the results of the inference...")
    output_files = []
    for file in file_list:
        if file.endswith("bch.csv"):
            cohort_dict = cohort_dict1
        else:
            cohort_dict = cohort_dict2
        patient_data = {}
        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader) 
            for row in csvreader:
                if len(row) >=4:
                    patient_id, scan_id, label, max_model_output = row[0], row[1], int(row[2]), float(row[3])
                    if patient_id in cohort_dict and scan_id in cohort_dict[patient_id]:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {
                                    'scans': [],
                                    'labels': [],
                                    'max_model_outputs': [],
                                }
                        patient_data[patient_id]['scans'].append(scan_id)
                        patient_data[patient_id]['labels'].append(label)
                        patient_data[patient_id]['max_model_outputs'].append(max_model_output)
        
        if not patient_data:
            print(f"No patient data found in {file}. Skipping...")
            continue
                
        for patient_id, data in patient_data.items():
            label_counts = {0: 0, 1: 0, 2: 0}
            label_probabilities = {0: [], 1: [], 2: []}
            
            for label, max_model_output in zip(data['labels'], data['max_model_outputs']):
                label_counts[label] += 1
                label_probabilities[label].append(max_model_output)
            
            avg_probabilities = {label: sum(probabilities) / len(probabilities) if probabilities else 0.0
                         for label, probabilities in label_probabilities.items()}
            
            final_label = max(avg_probabilities, key=avg_probabilities.get)
            data['final_label'] = final_label
            print(f"Patient ID: {patient_id}, Final Label: {final_label}")

        output_file = os.path.join(output_dir, f"final_{os.path.basename(file)}")
        output_files.append(output_file)
        
        # Write the final results to new csv
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            new_header = header + ['Final Label', 'Description']
            csvwriter.writerow(new_header)
            description = {0: "V600E", 1: "Fusion", 2: "Wildtype"}
            for patient_id, data in patient_data.items():
                descrip = description[data['final_label']]
                for scan_id, label, maxmdlout in zip(data['scans'], data['labels'], data['max_model_outputs']):
                    row = [patient_id, scan_id, label, maxmdlout, data['final_label'], descrip]
                    csvwriter.writerow(row)

        print(f"Final results written to {output_file}.")

    return output_files

def calculate_metrics(file_list, cohort_dict1, cohort_dict2):
    """
    Calculate metrics for the predictions.
    """
    for file in file_list:
        if file.endswith("bch.csv"):
            cohort = "BCH"
            cohort_dict = cohort_dict1
        else:
            cohort = "CBTN"
            cohort_dict = cohort_dict2

        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            confusion_matrix = {
                0: {0: 0, 1: 0, 2: 0},
                1: {0: 0, 1: 0, 2: 0},
                2: {0: 0, 1: 0, 2: 0}
                }

            for row in csvreader:
                if len(row) >= 6:
                    patient_id, scan_id, label, _, final_label, _ = row
                    if patient_id in cohort_dict and scan_id in cohort_dict[patient_id]:
                        label = int(label)
                        final_label = int(final_label)
                        confusion_matrix[label][final_label] += 1

            metrics = {}
            for class_label in range(3):
                true_positives = confusion_matrix[class_label][class_label]
                false_positives = sum(confusion_matrix[other_label][class_label] for other_label in range(3) if other_label != class_label)
                false_negatives = sum(confusion_matrix[class_label][other_label] for other_label in range(3) if other_label != class_label)
                true_negatives = sum(confusion_matrix[other_label1][other_label2] for other_label1 in range(3) for other_label2 in range(3) if other_label1 != class_label and other_label2 != class_label)

                # Calculate additional metrics
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                metrics[class_label] = {
                    'True Positives': true_positives,
                    'False Positives': false_positives,
                    'False Negatives': false_negatives,
                    'True Negatives': true_negatives,
                    'Precision': precision,
                    'Recall': recall,
                    'Accuracy': accuracy,
                    'F1 Score': f1_score
                }
                
            total_samples = sum(sum(confusion_matrix[i].values()) for i in range(3))
            overall_accuracy = sum(confusion_matrix[i][i] for i in range(3)) / total_samples
            macro_precision = sum(metrics[i]['Precision'] for i in range(3)) / 3
            macro_recall = sum(metrics[i]['Recall'] for i in range(3)) / 3
            macro_f1_score = sum(metrics[i]['F1 Score'] for i in range(3)) / 3

            # Print the metrics for the cohort
            print(f"\nMetrics for {cohort}:")
            for class_label in range(3):
                print(f"\nClass {class_label}")
                for metric, value in metrics[class_label].items():
                    if isinstance(value, float):
                        print(f"{metric}: {value:.2f}")
                    else:
                        print(f"{metric}: {value}")
            
            print("\nOverall Metrics:")
            print(f"Accuracy: {overall_accuracy:.2f}")
            print(f"Macro-averaged Precision: {macro_precision:.2f}")
            print(f"Macro-averaged Recall: {macro_recall:.2f}")
            print(f"Macro-averaged F1 Score: {macro_f1_score:.2f}")

def create_patient_scan_sets(cohort_file, patient_list):
    """
    Create sets of patient IDs and their corresponding scan IDs for two cohorts.
    """
    cohort_dict = {}

    # Process cohort 1 file
    with open(cohort_file, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.DictReader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            patient_id = row['Patient_ID'] 
            patient_id = prefix_zeros_to_six_digit_ids(patient_id)
            scan_id = row['Scan_ID']
            
            if patient_id in patient_list:
                if patient_id not in cohort_dict:
                    cohort_dict[patient_id] = set()

                cohort_dict[patient_id].add(scan_id)
    
    return cohort_dict

def prefix_zeros_to_six_digit_ids(patient_id):
    """
    Adds 0 to the beginning of 6-digit patient IDs.
    """
    str_id = str(patient_id)
    if not patient_id.startswith("C"):
        if len(str_id) == 6:
            # print(f"Found a 6-digit ID: {str_id}. Prefixing a '0'.")
            patient_id = "0" + str_id

        else:
            patient_id = str_id
    else:
        str_id = str_id[1:]
        patient_id = str_id
    return patient_id

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

    cbtn_patients = ["1003434","1042056","1047222","1077234","1095315","1232829","123861","2334663","2354097","2380788","2568978","2855076","324351","324597","3396522","3399720","36654","3684711","3817551","3971547","4065273","4095900","41082","4309797","4312872","46863","52644","53013","62730","63099","719673","735540","744642","77490","83640","836892","854604","860508","883509","88929","916473","95079","972192"]
    bch_patients = ["0135939","0137476","1058916","1194890","2001398","2004560","2088116","2103993","2113964","2147101","2173072","2183847","2260520","2261605","2280828","2306428","2316922","4015437","4092758","4098993","4108745","4132691","4137900","4155943","4252068","4303399","4304956","4305171","4345209","4348109","4362479","4416410","4450936","4455045","4478592","4489651","4505982","4571440","4572857","4624899","4635148","4647390","4695947","4802764","4803246","4857369","4923951","4931993","4975776","5002720","5029974","5046466","5048067","5208771","5238412","5531498"]
    patient_list = cbtn_patients + bch_patients

    bch_csv = "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/final_final_classification_bch.csv"
    cbtn_csv = "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/final_final_classification_cbtn.csv"
    file_list = [bch_csv, cbtn_csv]
    output_dir = "/home/jc053/GIT/mri_longitudinal_analysis/data/output"

    bch_scans = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats_bch/pre-treatment_dl_features.csv"
    cbtn_scans = "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats_cbtn/pre-treatment_dl_features.csv"

    if PART2:
        cohort_dict1 = create_patient_scan_sets(bch_scans, patient_list)
        cohort_dict2 = create_patient_scan_sets(cbtn_scans, patient_list)
                    
        output_files = part2(file_list, cohort_dict1, cohort_dict2, output_dir)
        calculate_metrics(output_files, cohort_dict1, cohort_dict2)


if __name__ == "__main__":
    main()
