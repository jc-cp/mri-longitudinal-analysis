import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

from cfg.volume_est_cfg import SEG_DIR, PLOTS_DIR, REDCAP_FILE, LIMIT_LOADING

class VolumeEstimator:
    def __init__(self, path, dob_file):
        self.path = path

        # Process the .csv with clinical data
        self.dob_df = pd.read_csv(dob_file, sep=',', encoding='UTF-8') 
        self.dob_df = self.dob_df[self.dob_df['no ops cohort'] == 'NAN']
        print(f"The length of the filtered dataset is: {len(self.dob_df)}")
        if len(self.dob_df) != 60:
            print("Warning: The length of the filtered dataset is not 60")
        self.dob_df['Date of Birth'] = pd.to_datetime(self.dob_df['Date of Birth'], dayfirst=False)
        print(self.dob_df['Date of Birth'].unique())
        self.dob_df['BCH MRN'] = self.dob_df['BCH MRN'].astype(int)
        
        self.volumes = defaultdict(list)

    @staticmethod
    def estimate_volume(segmentation_path):
        segmentation = sitk.ReadImage(segmentation_path)
        voxel_spacing = segmentation.GetSpacing()
        voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
        segmentation_array = sitk.GetArrayFromImage(segmentation)
        num_voxels = (segmentation_array > 0).sum()
        total_volume = num_voxels * voxel_volume
        return total_volume

    def process_files(self, max_patients=None):
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        patient_ids = set()
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            patient_id, _ = file_name.split("_")[0], file_name.split("_")[1]
            patient_ids.add(patient_id)

        if max_patients is not None and max_patients < len(patient_ids):
            patient_ids = list(patient_ids)[:max_patients]

        filtered_file_paths = [fp for fp in file_paths if os.path.basename(fp).split("_")[0] in patient_ids]
        filtered_df = self.dob_df[self.dob_df['BCH MRN'].astype(str).isin(patient_ids)]

        print(len(filtered_df))

        with Pool(cpu_count()) as p:
            results = p.map(self.estimate_volume, filtered_file_paths)

        for file_path, volume in zip(filtered_file_paths, results):
            file_name = os.path.basename(file_path)
            patient_id, date_str = file_name.split("_")[0], file_name.split("_")[1]
            date_str = date_str.replace(".nii.gz", "")
            date = datetime.strptime(date_str, "%Y%m%d")

            if patient_id in filtered_df['BCH MRN'].astype(str).values:
                dob = self.dob_df.loc[self.dob_df['BCH MRN'] == int(patient_id), 'Date of Birth'].iloc[0]
                age = (date - dob).days / 365.25        
                self.volumes[patient_id].append((date, volume, age))

    def plot_volumes(self, output_path):
        for patient_id, volumes in self.volumes.items():
            volumes.sort(key=lambda x: x[0])  # sort by date
            dates, volumes, ages = zip(*volumes)  # unzip to two lists
            os.makedirs(output_path, exist_ok=True)

            #plt.figure(figsize=(12, 8))
            #plt.plot(dates, volumes, marker="o")
            fig, ax1 = plt.subplots(figsize=(12, 8))

            color = 'tab:blue'
            ax1.set_xlabel('Scan Date')
            ax1.set_ylabel('Volume (mm³)', color=color)
            ax1.plot(dates, volumes, color=color, marker="o")
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
            ax1.set_xticks(dates)
            ax1.set_xticklabels([dt.strftime('%m/%d/%Y') for dt in dates], rotation=90, fontsize=8)
            ax1.xaxis.set_tick_params(pad=5)

            ax2 = ax1.twiny()
            ax2.xaxis.set_ticks_position('top') 
            ax2.xaxis.set_label_position('top')
            ax2.set_xlabel('Patient Age (Years)')
            ax2.set_xlim(ax1.get_xlim())
            date_nums = mdates.date2num(dates)
            ax2.set_xticks(date_nums)
            ax2.set_xticklabels([f"{age:.1f}" for age in ages])
            ax2.xaxis.set_tick_params(labelsize=8)

            # volume changes
            volume_changes = [0]
            for i, v in enumerate(volumes[1:], 1):
                if volumes[i - 1] != 0:
                    volume_change = ((v - volumes[i - 1]) / volumes[i - 1]) * 100
                else:
                    volume_change = np.nan  # or another value of your choice
                volume_changes.append(volume_change)

            for i, (date, volume, volume_change, age) in enumerate(
                zip(dates, volumes, volume_changes, ages)
            ):
                ax1.text(
                    date,
                    volume,
                    f"{volume_change:.2f}%",
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )


            plt.title(f"Patient ID: {patient_id}")
            #plt.xlabel("Scan Date / Patient Age")
            #plt.ylabel("Volume (mm³)")

            # Format x-axis to show dates
            #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
            #plt.xticks(ticks=dates, rotation=90, fontsize=8)
            #plt.gca().xaxis.set_tick_params(pad=5)

            fig.tight_layout()

            date_range = (
                f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
            )
            plt.savefig(
                os.path.join(output_path, f"volume_{patient_id}_{date_range}.png")
            )
            plt.close()


if __name__ == "__main__":
    ve = VolumeEstimator(SEG_DIR, REDCAP_FILE)
    print("Getting prediction masks.")
    ve.process_files(max_patients=LIMIT_LOADING)
    print("Saving data.")
    ve.plot_volumes(output_path=PLOTS_DIR)