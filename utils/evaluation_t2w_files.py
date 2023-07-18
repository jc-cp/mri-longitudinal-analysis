import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(sys.path.append(str(Path(__file__).resolve().parent.parent)))
from tqdm import tqdm

from cfg.evaluation_cfg import (CSV_FILE, DATA_FOLDER, DIR1_NO_COMMENTS,
                                DIR1_WITH_COMMENTS, DIR5, MOVING, OUT_FILE)


# Define function to move files based on condition once review
def move_files(df, condition, source_dir, destination_folder):
    filenames = df[condition]["Image_Name"]
    for filename in tqdm(filenames, desc="Moving files"):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_folder, filename)
        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)


# START SCRIPT: Define csv_file
csv_file = CSV_FILE

# Define column names and load
column_names = ["Image_Name", "Quality", "Comments"]
df = pd.read_csv(csv_file, names=column_names)

# Compute some statistics
df["Quality"] = df["Quality"].astype(int)
num_images = df["Image_Name"].nunique()
avg_quality = df["Quality"].mean()
quality_dist = df["Quality"].value_counts()

print(f"Number of unique images: {num_images}")
print(f"Average quality: {avg_quality:.2f}")
print("\nQuality distribution:", quality_dist)

# Get all unique comments
all_comments = df["Comments"].unique()
print(f"All types of annotations: {all_comments}")


# Filter comments for quality 5
quality5_comments = df[df["Quality"] == 5]["Comments"]
quality1_comments = df[df["Quality"] == 1]["Comments"]
print("\nComments for quality rating of 5:", quality5_comments.value_counts())
print("\nComments for quality rating of 1:", quality1_comments.value_counts())

# Number of images with quality 1
quality1_no_comments = df[(df["Quality"] == 1) & (df["Comments"].isna())]
quality1_with_comments = df[(df["Quality"] == 1) & (df["Comments"].notna())]
print(
    f"\nNumber of images with quality rating of 1 and no comments: {len(quality1_no_comments)}"
)
print(
    f"Number of images with quality rating of 1 and with comments: {len(quality1_with_comments)}"
)

# Number of images with quality 5
quality5_no_comments = df[(df["Quality"] == 5) & (df["Comments"].isna())]
quality5_with_comments = df[(df["Quality"] == 5) & (df["Comments"].notna())]
print(
    f"\nNumber of images with quality rating of 5 and no comments: {len(quality5_no_comments)}"
)
assert len(quality5_no_comments) == 0
print(
    f"Number of images with quality rating of 5 and with comments: {len(quality5_with_comments)}"
)


# Compute some statistics
df["Quality"] = df["Quality"].astype(int)
df["Comments"] = df["Comments"].fillna("Valid Images")

# Define comment categories
comment_categories = [
    "tricky"
    "FLAIR",
    "T1",
    "T1c",
    "OTHER",
    "Valid Images",
    "other body part",
    "quality",
    "view",
    "cropped",
]

# New column to hold the category
df["Category"] = df["Quality"].astype(str) + "-" + df["Comments"].map(str)

# Only consider records with Quality as 1 or 5, and Comments in comment_categories
df_filtered = df[df["Quality"].isin([1, 5]) & df["Quality"]==1]# df["Comments"].isin(comment_categories)]

# Histogram of the quality
plt.figure(figsize=(12, 8))
plt.title("Distribution of Quality Ratings")
ax = sns.countplot(data=df_filtered, x="Quality", hue="Comments")

# Iterate over the bars, and add a label for each
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10), textcoords='offset points')
    

plt.tight_layout()
plt.savefig(OUT_FILE)


# Moving Files if needed
moving = MOVING
if moving:
    # Folder variables
    source_dir = DATA_FOLDER
    dir1_no_comments = DIR1_NO_COMMENTS
    dir1_with_comments = DIR1_WITH_COMMENTS
    dir5 = DIR5

    # Make dirs
    os.makedirs(dir1_no_comments, exist_ok=True)
    os.makedirs(dir1_with_comments, exist_ok=True)
    os.makedirs(dir5, exist_ok=True)

    # Move the files
    move_files(
        df,
        (df["Quality"] == 1) & (df["Comments"] == "None"),
        source_dir,
        dir1_no_comments,
    )
    move_files(
        df,
        (df["Quality"] == 1) & (df["Comments"] != "None"),
        source_dir,
        dir1_with_comments,
    )
    move_files(df, df["Quality"] == 5, source_dir, dir5)
