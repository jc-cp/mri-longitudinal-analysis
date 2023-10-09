"""
This script provides an evaluation framework for analyzing image quality ratings.
It reads data from specified CSV files, processes and filters the data,
and creates a histogram to visualize the distribution of quality ratings.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cfg import evaluation_cfg


class Evaluation:
    """
    A class to encapsulate the evaluation process for image quality.
    """

    def __init__(self):
        """
        Initialize the evaluation object with files and other configurations.
        """
        self.files = {
            "60": evaluation_cfg.CSV_FILE_60,
            "29": evaluation_cfg.CSV_FILE_29,
            "18": evaluation_cfg.CSV_FILE_18,
        }
        self.out_file_prefix = evaluation_cfg.OUT_FILE_PREFIX

    def read_file(self, csv_file) -> pd.DataFrame:
        """
        Read a CSV file and return a DataFrame.

        Args:
            csv_file (str): Path to the CSV file to be read.

        Returns:
            DataFrame: DataFrame containing image names, quality ratings, and comments.
        """
        column_names = ["Image_Name", "Quality", "Comments"]
        d_f = pd.read_csv(csv_file, names=column_names)
        return d_f

    def parse_df(self, d_f) -> pd.DataFrame:
        """
        Parse and filter the DataFrame to compute various statistics.

        Args:
            d_f (DataFrame): Input DataFrame to be parsed.

        Returns:
            DataFrame: Filtered DataFrame containing only relevant data.
        """
        # Compute some statistics
        d_f["Quality"] = d_f["Quality"].astype(int)
        num_images = d_f["Image_Name"].nunique()
        avg_quality = d_f["Quality"].mean()
        quality_dist = d_f["Quality"].value_counts()

        print(f"Number of unique images: {num_images}")
        print(f"Average quality: {avg_quality:.2f}")
        print("\nQuality distribution:", quality_dist)

        # Get all unique comments
        all_comments = d_f["Comments"].unique()
        print(f"All types of annotations: {all_comments}")

        # Filter comments for quality 5
        quality5_comments = d_f[d_f["Quality"] == 5]["Comments"]
        quality1_comments = d_f[d_f["Quality"] == 1]["Comments"]
        print(
            "\nComments for quality rating of 5:",
            quality5_comments.value_counts(),
        )
        print(
            "\nComments for quality rating of 1:",
            quality1_comments.value_counts(),
        )
        # Number of images with quality 1
        quality1_no_comments = d_f[(d_f["Quality"] == 1) & (d_f["Comments"].isna())]
        quality1_with_comments = d_f[(d_f["Quality"] == 1) & (d_f["Comments"].notna())]
        print(f"\nImages with quality rating of 1 and no comments: {len(quality1_no_comments)}")

        print(f"\nImages with quality rating of 1 and with comments: {len(quality1_with_comments)}")
        # Number of images with quality 5
        quality5_no_comments = d_f[(d_f["Quality"] == 5) & (d_f["Comments"].isna())]
        quality5_with_comments = d_f[(d_f["Quality"] == 5) & (d_f["Comments"].notna())]
        print(f"\nImages with quality rating of 5 and no comments: {len(quality5_no_comments)}")

        assert len(quality5_no_comments) == 0
        print(f"\nImages with quality rating of 5 and with comments: {len(quality5_with_comments)}")

        # Compute some statistics
        d_f["Quality"] = d_f["Quality"].astype(int)
        d_f["Comments"] = d_f["Comments"].fillna("Valid Images")

        # New column to hold the category
        d_f["Category"] = d_f["Quality"].astype(str) + "-" + d_f["Comments"].map(str)

        # Only consider records with Quality as 1 or 5, and Comments in comment_categories
        df_filtered = d_f[d_f["Quality"].isin([1, 5]) & d_f["Quality"] == 1]
        return df_filtered

    def hist_plot(self, df_filtered, suffix):
        """
        Create and save a histogram plot based on the filtered DataFrame.

        Args:
            df_filtered (DataFrame): Filtered DataFrame for plotting.
            suffix (str): Suffix to be appended to the output file name.
        """
        prefix = str(self.out_file_prefix)

        # Histogram of the quality
        plt.figure(figsize=(12, 8))
        plt.title("Distribution of Quality Ratings")
        a_x = sns.countplot(data=df_filtered, x="Quality", hue="Comments")

        # Iterate over the bars, and add a label for each
        for patch in a_x.patches:
            a_x.annotate(
                format(patch.get_height(), ".0f"),
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        plt.tight_layout()
        plt.savefig(f"{prefix}_{suffix}_cohort.png")

    def main(self):
        """
        Main function to execute the evaluation workflow.
        """
        for suffix, csv_file in self.files.items():
            d_f = self.read_file(csv_file)
            filtered_df = self.parse_df(d_f)
            self.hist_plot(filtered_df, suffix)


if __name__ == "__main__":
    ev = Evaluation()
    ev.main()
