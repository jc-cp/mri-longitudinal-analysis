"""
Creates a basic script that outputs the necessary csv-file for the BRAF inference.
"""
import os
import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def part1(directory, csv_file, header):
    """
    Purpose:
        Create a CSV file with patient_id, scan_id, and a placeholder label for the BRAF inference pipeline.
    Process:
        Iterates through a directory containing .nii.gz files (excluding mask files).
        Extracts patient ID and scan ID from filenames.
        Writes this information to a CSV file, using "3" as a placeholder label.
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

def part2(file_list, cohort_dict1, cohort_dict2, output_dir, use_majority_vote=False):
    """
    Purpose: 
        Evaluate the results of the inference and create final output files.
    Process:
        Reads input CSV files (for BCH and CBTN cohorts).
        Processes each patient's data, calculating average probabilities for each label.
        Assigns a final label based on the highest average probability.
        Writes final results to new CSV files, including the final label and a description (V600E, Fusion, or Wildtype).
        Calculates metrics (confusion matrix, precision, recall, accuracy, F1 score) for each cohort.    
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
            if use_majority_vote:
                # Majority Voting
                label_counts = Counter(data['labels'])
                final_label = label_counts.most_common(1)[0][0]
                output_file = os.path.join(output_dir, f"majority_vote_{os.path.basename(file)}")

            else:
                label_probabilities = {0: [], 1: [], 2: []}
            
                for label, max_model_output in zip(data['labels'], data['max_model_outputs']):
                    label_probabilities[label].append(max_model_output)
            
                avg_probabilities = {label: sum(probabilities) / len(probabilities) if probabilities else 0.0 for label, probabilities in label_probabilities.items()}
            
                final_label = max(avg_probabilities, key=avg_probabilities.get)
                output_file = os.path.join(output_dir, f"avg_prob_{os.path.basename(file)}")

            
            data['final_label'] = final_label
            print(f"Patient ID: {patient_id}, Final Label: {final_label}")

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

def part3(file_list, output_dir, use_majority_vote=False):
    """
    Analyze the stability of predictions longitudinally for each patient.
    """
    print("Analyzing longitudinal stability of predictions...")

    all_stability_results = []
    all_patient_stats = {}

    for file in file_list:
        patient_data = {}
        
        # Read the CSV file
        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                patient_id = row['pat_id']
                scan_date = row['scandate']
                final_prediction = int(row['FinalPrediction']) # prediction of the model
                final_label = int(row['Final Label']) # assigned label according to employed methodology
                max_model_output = float(row['MaxModelOutput'])
                
                if patient_id not in patient_data:
                    patient_data[patient_id] = []
                
                patient_data[patient_id].append({
                    'scan_date': scan_date,
                    'final_prediction': final_prediction,
                    'final_label' : final_label, 
                    'max_model_output': max_model_output
                })
                
        # Analyze stability for each patient
        stability_results = []
        for patient_id, scans in patient_data.items():
            if len(scans) > 1:  # Only analyze patients with multiple scans
                predictions = [scan['final_prediction'] for scan in scans]
                final_labels = [scan['final_label'] for scan in scans]
                probabilities = [scan['max_model_output'] for scan in scans]
                
                # Calculate metrics
                prediction_consistency = sum(m == f for m, f in zip(predictions, final_labels)) / len(predictions)
                
                class_counts_original = {0: 0, 1: 0, 2: 0}
                class_counts_final = {0: 0, 1: 0, 2: 0}
                for o, f in zip(predictions, final_labels):
                    class_counts_original[o] += 1
                    class_counts_final[f] += 1
                
                majority_class_original = max(class_counts_original, key=class_counts_original.get)
                majority_class_final = max(class_counts_final, key=class_counts_final.get)  
                
                class_changes = sum(o != f for o, f in zip(predictions, final_labels))
                probability_mean = np.mean(probabilities)
                probability_std = np.std(probabilities)
                probability_range = max(probabilities) - min(probabilities)
                cv = probability_std / probability_mean if probability_mean != 0 else np.nan
                
                stability_results.append({
                    'patient_id': patient_id,
                    'num_scans': len(scans),
                    'consistency': prediction_consistency,
                    'majority_class_original': majority_class_original,
                    'majority_class_final': majority_class_final,
                    'class_changes' : class_changes,
                    'probability_mean': probability_mean,
                    'probability_std': probability_std,
                    'probability_range': probability_range,
                    'cv': cv
                })

        # Perform additional consistency checks
        all_stability_results.extend(stability_results)
        patient_stats = additional_consistency_checks(patient_data)
        all_patient_stats.update(patient_stats)

        # Write stability results to a new CSV file
        prefix = "majority_vote" if use_majority_vote else"avg_prob"
        output_file = f"{output_dir}/{prefix}_longitudinal_stability_{file.split('_')[-1]}"
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ['patient_id', 'num_scans', 'consistency', 'majority_class_original', 
                          'majority_class_final', 'class_changes', 'probability_mean', 
                          'probability_std', 'probability_range', 'cv']
            csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows(stability_results)

        print(f"Longitudinal stability results written to {output_file}")

    
    # Calculate and print summary statistics
    print_summary_statistics(all_stability_results, all_patient_stats, use_majority_voting=use_majority_vote)
    generate_visualizations(all_stability_results, all_patient_stats, output_dir, use_majority_voting=use_majority_vote)

    return all_stability_results

def calculate_metrics(file_list, cohort_dict1, cohort_dict2, output_dir, use_majority_vote):
    """
    Calculate & save metrics for the predictions.
    """
    method = "Majority Voting" if use_majority_vote else "Average Probability"
    output_file = f"{output_dir}/metrics_{method.lower().replace(' ', '_')}.txt"
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(f"Metrics for {method} method:\n\n")
    
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

            # Print & save the metrics for the cohort
            print(f"\nMetrics for {cohort}:")
            for class_label in range(3):
                f.write(f"\nClass {class_label}\n")
                print(f"\nClass {class_label}")
                for metric, value in metrics[class_label].items():
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.2f}\n")
                        print(f"{metric}: {value:.2f}")
                    else:
                        f.write(f"{metric}: {value}\n")
                        print(f"{metric}: {value}")
            
            f.write("\nOverall Metrics:\n")
            f.write(f"Accuracy: {overall_accuracy:.2f}\n")
            f.write(f"Macro-averaged Precision: {macro_precision:.2f}\n")
            f.write(f"Macro-averaged Recall: {macro_recall:.2f}\n")
            f.write(f"Macro-averaged F1 Score: {macro_f1_score:.2f}\n")
            print("\nOverall Metrics:")
            print(f"Accuracy: {overall_accuracy:.2f}")
            print(f"Macro-averaged Precision: {macro_precision:.2f}")
            print(f"Macro-averaged Recall: {macro_recall:.2f}")
            print(f"Macro-averaged F1 Score: {macro_f1_score:.2f}")
    print(f"Metrics saved to {output_file}")

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

def get_mode(data):
    """
    Calculate the mode of a list of data using a robust method.
    """
    # Use Counter to get the most common element
    return Counter(data).most_common(1)[0][0]

def print_summary_statistics(stability_results, patient_stats, use_majority_voting):
    """
    Print summary statistics for the longitudinal stability analysis.
    """
    if use_majority_voting:
        print("RESULTS FOR MAJORITY VOTING")
    else:
        print("RESULTS FOR AVG PROBABILITIES")
    
    num_patients = len(stability_results)
    avg_scans = np.mean([result['num_scans'] for result in stability_results])
    avg_consistency = np.mean([result['consistency'] for result in stability_results])
    avg_std = np.mean([result['probability_std'] for result in stability_results])
    avg_range = np.mean([result['probability_range'] for result in stability_results])
    avg_cv = np.mean([result['cv'] for result in stability_results])

    print("\nSummary Statistics:")
    print(f"Number of patients analyzed: {num_patients}")
    print(f"Average number of scans per patient: {avg_scans:.2f}")
    print(f"Average consistency: {avg_consistency:.2f}")
    print(f"Average probability standard deviation: {avg_std:.2f}")
    print(f"Average probability range: {avg_range:.2f}")
    print(f"Average coefficient of variation: {avg_cv:.2f}")

    # Additional statistics from patient_stats
    overall_mean = np.mean([stats['mean'] for stats in patient_stats.values()])
    overall_std = np.std([stats['mean'] for stats in patient_stats.values()])
    avg_patient_std = np.mean([stats['std'] for stats in patient_stats.values()])

    print("\nAdditional Statistics:")
    print(f"Overall mean of max_model_output: {overall_mean:.4f}")
    print(f"Overall std of patient means: {overall_std:.4f}")
    print(f"Average within-patient std: {avg_patient_std:.4f}")

def generate_visualizations(stability_results, patient_stats, output_dir, use_majority_voting=False):
    """
    Generate visualizations for the longitudinal stability analysis.
    """
    prefix = "Majority_Vote" if use_majority_voting else "Maximum_Average"
    
    # 1. Consistency Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot([r['consistency'] for r in stability_results], bins=20, kde=True)
    plt.xlabel('Consistency')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Consistency ({prefix})')
    plt.savefig(f"{output_dir}/{prefix.lower().replace(' ', '_')}_consistency_distribution.png")
    plt.close()

    # 2. Class Changes Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot([r['class_changes'] for r in stability_results], bins=range(0, max([r['num_scans'] for r in stability_results]) + 2), kde=False)
    plt.xlabel('Number of Class Changes')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Class Changes ({prefix})')
    plt.savefig(f"{output_dir}/{prefix.lower().replace(' ', '_')}_class_changes_distribution.png")
    plt.close()

    # 3. Consistency vs Number of Scans
    plt.figure(figsize=(10, 6))
    plt.scatter([r['num_scans'] for r in stability_results], [r['consistency'] for r in stability_results])
    plt.xlabel('Number of Scans')
    plt.ylabel('Consistency')
    plt.title(f'Consistency vs Number of Scans ({prefix})')
    plt.savefig(f"{output_dir}/{prefix.lower().replace(' ', '_')}_consistency_vs_scans.png")
    plt.close()

    # 4. Probability Mean vs Consistency
    plt.figure(figsize=(10, 6))
    plt.scatter([r['probability_mean'] for r in stability_results], [r['consistency'] for r in stability_results])
    plt.xlabel('Mean Probability')
    plt.ylabel('Consistency')
    plt.title(f'Mean Probability vs Consistency ({prefix})')
    plt.savefig(f"{output_dir}/{prefix.lower().replace(' ', '_')}_mean_probability_vs_consistency.png")
    plt.close()

def additional_consistency_checks(patient_data):
    """
    Perform additional checks on the consistency of predictions.
    """
    patient_stats = {}
    all_outputs = []

    for patient_id, scans in patient_data.items():
        if len(scans) > 1:
            outputs = [scan['max_model_output'] for scan in scans]
            patient_stats[patient_id] = {
                'mean': np.mean(outputs),
                'std': np.std(outputs)
            }
            all_outputs.extend(outputs)

    overall_mean = np.mean(all_outputs)
    overall_std = np.std(all_outputs)
    avg_patient_std = np.mean([stats['std'] for stats in patient_stats.values()])

    print(f"Overall mean of max_model_output: {overall_mean:.4f}")
    print(f"Overall std of max_model_output: {overall_std:.4f}")
    print(f"Average within-patient std: {avg_patient_std:.4f}")

    return patient_stats

def compare_methods_visualization(results_avg_prob, results_majority, output_dir):
    """
    Generate visualizations comparing the two methods directly.
    """
    # Consistency Comparison
    plt.figure(figsize=(12, 6))
    sns.histplot([r['consistency'] for r in results_avg_prob], bins=20, kde=True, label='Max Average')
    sns.histplot([r['consistency'] for r in results_majority], bins=20, kde=True, label='Majority Vote')
    plt.xlabel('Consistency')
    plt.ylabel('Frequency')
    plt.title('Consistency Distribution Comparison')
    plt.legend()
    plt.savefig(f"{output_dir}/method_comparison_consistency.png")
    plt.close()

    # Class Changes Comparison
    plt.figure(figsize=(12, 6))
    sns.histplot([r['class_changes'] for r in results_avg_prob], bins=range(0, 20), kde=False, label='Max Average')
    sns.histplot([r['class_changes'] for r in results_majority], bins=range(0, 20), kde=False, label='Majority Vote')
    plt.xlabel('Number of Class Changes')
    plt.ylabel('Frequency')
    plt.title('Class Changes Distribution Comparison')
    plt.legend()
    plt.savefig(f"{output_dir}/method_comparison_class_changes.png")
    plt.close()

    # Coefficient of Variation Distribution (combined for both methods)
    plt.figure(figsize=(12, 6))
    sns.histplot([r['cv'] for r in results_avg_prob if not np.isnan(r['cv'])], bins=20, kde=True, label='Combined')
    #sns.histplot([r['cv'] for r in results_majority if not np.isnan(r['cv'])], bins=20, kde=True, label='Majority Vote')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Coefficient of Variation (Comparison)')
    plt.legend()
    plt.savefig(f"{output_dir}/method_comparison_cv_distribution.png")
    plt.close()

    # Mean Probability vs Consistency Comparison
    plt.figure(figsize=(12, 6))
    plt.scatter([r['probability_mean'] for r in results_avg_prob], 
                [r['consistency'] for r in results_avg_prob], 
                label='Max Average', alpha=0.6)
    plt.scatter([r['probability_mean'] for r in results_majority], 
                [r['consistency'] for r in results_majority], 
                label='Majority Vote', alpha=0.6)
    plt.xlabel('Mean Probability')
    plt.ylabel('Consistency')
    plt.title('Mean Probability vs Consistency Comparison')
    plt.legend()
    plt.savefig(f"{output_dir}/method_comparison_mean_probability_vs_consistency.png")
    plt.close()

    # Consistency vs Number of Scans Comparison
    plt.figure(figsize=(12, 6))
    plt.scatter([r['num_scans'] for r in results_avg_prob], [r['consistency'] for r in results_avg_prob], label='Max Average', alpha=0.6)
    plt.scatter([r['num_scans'] for r in results_majority], [r['consistency'] for r in results_majority], label='Majority Vote', alpha=0.6)
    plt.xlabel('Number of Scans')
    plt.ylabel('Consistency')
    plt.title('Consistency vs Number of Scans Comparison')
    plt.legend()
    plt.savefig(f"{output_dir}/method_comparison_consistency_vs_scans.png")
    plt.close()

    # Combined Class Distribution Comparison
    plot_combined_class_distribution(results_avg_prob, results_majority, output_dir)
    
    print(f"Method comparison visualizations saved in {output_dir}")

def plot_combined_class_distribution(results_avg_prob, results_majority, output_dir):
    """
    Plot combined class distribution for both methods.
    """
    original_classes = [r['majority_class_original'] for r in results_avg_prob]
    final_classes_avg = [r['majority_class_final'] for r in results_avg_prob]
    final_classes_majority = [r['majority_class_final'] for r in results_majority]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original Class Distribution
    sns.countplot(x=original_classes, ax=ax1)
    ax1.set_title('Original Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    for i, count in enumerate(ax1.containers[0]):
        ax1.text(count.get_x() + count.get_width()/2, count.get_height(), 
                 str(int(count.get_height())), ha='center', va='bottom')

    # Final Class Distribution (Max Average)
    sns.countplot(x=final_classes_avg, ax=ax2)
    ax2.set_title('Final Class Distribution (Max Average)')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    for i, count in enumerate(ax2.containers[0]):
        ax2.text(count.get_x() + count.get_width()/2, count.get_height(), 
                 str(int(count.get_height())), ha='center', va='bottom')

    # Final Class Distribution (Majority Vote)
    sns.countplot(x=final_classes_majority, ax=ax3)
    ax3.set_title('Final Class Distribution (Majority Vote)')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    for i, count in enumerate(ax3.containers[0]):
        ax3.text(count.get_x() + count.get_width()/2, count.get_height(), 
                 str(int(count.get_height())), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_class_distribution_comparison.png")
    plt.close()

def main():
    """
    part1: creates the csv files and prepares them for inference
    part2: evaluate the results after inference
    part3: checks the consitency of the predictions
    """
    PART1 = False
    PART2 = False
    PART3 = True

    output_dir = "/home/jc053/GIT/mri_longitudinal_analysis/data/output"
    output_braf = f"{output_dir}/braf_inf"
    output_part1 = f"{output_braf}/part1"
    output_part2 = f"{output_braf}/part2"
    output_part3 = f"{output_braf}/part3"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_braf, exist_ok=True)
    os.makedirs(output_part1, exist_ok=True)
    os.makedirs(output_part2, exist_ok=True)
    os.makedirs(output_part3, exist_ok=True)

    ##########
    # PART 1 #
    ##########
    # Directory containing the files and the CSV file output
    img_dir = "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/final_dataset"
    # Template file in order to run BRAF inference
    csv_file = f"{output_part1}/final_patients_all_scans.csv"
    header = ["pat_id", "scandate", "label"]

    if PART1:
        part1(img_dir, csv_file, header)

    ##########
    # PART 2 #
    ##########
    # Just the IDs, no C's for CBTN
    cbtn_patients = ["1003434","1042056","1047222","1077234","1095315","1232829","123861","2334663","2354097","2380788","2568978","2855076","324351","324597","3396522","3399720","36654","3684711","3817551","3971547","4065273","4095900","41082","4309797","4312872","46863","52644","53013","62730","63099","719673","735540","744642","77490","83640","836892","854604","860508","883509","88929","916473","95079","972192"]
    bch_patients = ["0135939","0137476","1058916","1194890","2001398","2004560","2088116","2103993","2113964","2147101","2173072","2183847","2260520","2261605","2280828","2306428","2316922","4015437","4092758","4098993","4108745","4132691","4137900","4155943","4252068","4303399","4304956","4305171","4345209","4348109","4362479","4416410","4450936","4455045","4478592","4489651","4505982","4571440","4572857","4624899","4635148","4647390","4695947","4802764","4803246","4857369","4923951","4931993","4975776","5002720","5029974","5046466","5048067","5208771","5238412","5531498"]
    patient_list = cbtn_patients + bch_patients

    # Input files should be in the second folder after inference
    bch_csv = f"{output_part2}/final_classification_bch.csv"
    cbtn_csv = f"{output_part2}/final_classification_cbtn.csv"
    file_list = [bch_csv, cbtn_csv]

    # Clinical files
    file_name = "pre-treatment_dl_features.csv"
    bch_scans = f"{output_dir}/correlation_stats_bch/{file_name}"
    cbtn_scans = f"{output_dir}/correlation_stats_cbtn/{file_name}"

    if PART2:
        cohort_dict1 = create_patient_scan_sets(bch_scans, patient_list)
        cohort_dict2 = create_patient_scan_sets(cbtn_scans, patient_list)
                    
        output_files_avg_prob = part2(file_list, cohort_dict1, cohort_dict2, output_part2, use_majority_vote=False)
        calculate_metrics(output_files_avg_prob, cohort_dict1, cohort_dict2, output_part2, use_majority_vote=False)
        
        output_files_majority_vote = part2(file_list, cohort_dict1, cohort_dict2, output_part2, use_majority_vote=True)
        calculate_metrics(output_files_majority_vote, cohort_dict1, cohort_dict2, output_part2, use_majority_vote=True)

    ##########
    # PART 3 #
    ##########
    if PART3:
        input_avg_prob = [f"{output_part2}/avg_prob_final_classification_bch.csv", f"{output_part2}/avg_prob_final_classification_cbtn.csv"]
        input_majority = [f"{output_part2}/majority_vote_final_classification_bch.csv", f"{output_part2}/majority_vote_final_classification_cbtn.csv"]
        results_avg_prob = part3(input_avg_prob, output_part3, use_majority_vote=False)
        results_majority = part3(input_majority, output_part3, use_majority_vote=True)
        compare_methods_visualization(results_avg_prob, results_majority, output_part3)

if __name__ == "__main__":
    main()
