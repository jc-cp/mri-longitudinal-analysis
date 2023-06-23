import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# Define csv_file
csv_file = "/home/jc053/GIT/mri-longitudinal-segmentation/data/t2w/annotations.csv"

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
print(f"\nNumber of images with quality rating of 1 and no comments: {len(quality1_no_comments)}")
print(f"Number of images with quality rating of 1 and with comments: {len(quality1_with_comments)}")

# Number of images with quality 5
quality5_no_comments = df[(df["Quality"] == 5) & (df["Comments"].isna())]
quality5_with_comments = df[(df["Quality"] == 5) & (df["Comments"].notna())]
print(f"\nNumber of images with quality rating of 5 and no comments: {len(quality5_no_comments)}")
print(f"Number of images with quality rating of 5 and with comments: {len(quality5_with_comments)}")

# Histogram of the quality 
plt.figure(figsize=(10, 6))
quality_plot = sns.countplot(data=df, x="Quality")
plt.title("Distribution of Quality Ratings")

# Compute some statistics
df["Quality"] = df["Quality"].astype(int)
df["Comments"] = df["Comments"].fillna("None")

# Define comment categories
comment_categories = ["FLAIR", "T1", "T1c", "OTHER", "None", "artifact", "quality", "view", "cropped"]

quality_list = [int(label.get_text()) for label in quality_plot.get_xticklabels()]

# # Iterate over the patches (bars)
# for i, p in enumerate(quality_plot.patches):
#     # Get corresponding quality rating
#     quality = quality_list[i]
    
#     # Get comments for current quality rating
#     comments = df[df["Quality"] == quality]["Comments"]
    
#     # Count comment categories
#     comment_counts = Counter(comments)
    
#     # Create annotation text: each line will have format "Comment: Count"
#     annotation_text = "\n".join(f"{category}: {comment_counts.get(category, 0)}"
#                                 for category in comment_categories)
    
#     # Add annotation to the bar
#     quality_plot.text(
#         p.get_x() + p.get_width() / 2.0, p.get_height(),
#         annotation_text,
#         ha="center", va="center",
#         fontsize=8, color='black',
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
#     )
legend_patches = []
for i, p in enumerate(quality_plot.patches):
    # Get corresponding quality rating
    quality = quality_list[i]
    
    # Annotate total count
    quality_plot.annotate(
        format(p.get_height(), ".0f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )
    
    # Only calculate comment categories for quality ratings 1 and 5
    if quality in [1, 5]:
        # Get comments for current quality rating
        comments = df[df["Quality"] == quality]["Comments"]
        
        # Count comment categories
        comment_counts = Counter(comments)
        
        # Add to the legend patches list
        for category in comment_categories:
            count = comment_counts.get(category, 0)
            legend_patches.append(mpatches.Patch(
                color='none', label=f"Quality {quality} - {category}: {count}"))

# Create legend
plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc='upper left')

plt.tight_layout()
plt.savefig("output_evaluation_t2w.png")