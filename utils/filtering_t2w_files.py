import os
import shutil
import pandas as pd

def filter_files(input_dir, csv_file, output_dir):
    # Define column names
    column_names = ["Image_Name", "Quality", "Comments"]

    # Load the data
    df = pd.read_csv(csv_file, names=column_names)
    df["Quality"] = df["Quality"].astype(int)

    # Get the image names of the files with Quality 1 and no comments
    image_names = df[(df["Quality"] == 1) & (df["Comments"].isna())]["Image_Name"]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Copy the files
    for name in image_names:
        # Create full file paths
        input_file = os.path.join(input_dir, name)
        output_file = os.path.join(output_dir, name)

        # Check if file exists
        if os.path.isfile(input_file):
            # Copy the file
            shutil.copy2(input_file, output_file)

# Define directories and file paths
input_dir = "/path/to/input_directory"
csv_file = "/path/to/annotations.csv"
output_dir = "/path/to/output_directory"

# Call the function
filter_files(input_dir, csv_file, output_dir)
