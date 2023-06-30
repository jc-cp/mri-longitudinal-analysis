import glob
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import SimpleITK as sitk


class VolumeEstimator:
    def __init__(self, path):
        self.path = path
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

    def process_files(self):
        file_paths = glob.glob(os.path.join(self.path, "*.nii.gz"))

        with Pool(cpu_count()) as p:
            results = p.map(self.estimate_volume, file_paths)

        for file_path, volume in zip(file_paths, results):
            file_name = os.path.basename(file_path)
            patient_id, date_str = file_name.split("_")[0], file_name.split("_")[1]
            date_str = date_str.replace(".nii.gz", "")
            date = datetime.strptime(date_str, "%Y%m%d")
            volume = self.estimate_volume(file_path)
            self.volumes[patient_id].append((date, volume))

    def plot_volumes(self, output_path):
        for patient_id, volumes in self.volumes.items():
            volumes.sort(key=lambda x: x[0])  # sort by date
            dates, volumes = zip(*volumes)  # unzip to two lists
            os.makedirs(output_path, exist_ok=True)

            plt.figure(figsize=(12, 8))

            plt.plot(dates, volumes, marker="o")

            # volume changes
            volume_changes = [0]
            for i, v in enumerate(volumes[1:], 1):
                if volumes[i - 1] != 0:
                    volume_change = ((v - volumes[i - 1]) / volumes[i - 1]) * 100
                else:
                    volume_change = np.nan  # or another value of your choice
                volume_changes.append(volume_change)

            for i, (date, volume, volume_change) in enumerate(
                zip(dates, volumes, volume_changes)
            ):
                plt.text(
                    date,
                    volume,
                    f"{volume_change:.2f}%",
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )

            plt.title(f"Patient ID: {patient_id}")
            plt.xlabel("Date")
            plt.ylabel("Volume (mm³)")

            # Format x-axis to show dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
            plt.xticks(ticks=dates, rotation=90, fontsize=8)
            plt.gca().xaxis.set_tick_params(pad=5)

            plt.tight_layout()

            date_range = (
                f"{min(dates).strftime('%Y%m%d')}_{max(dates).strftime('%Y%m%d')}"
            )
            plt.savefig(
                os.path.join(output_path, f"volume_{patient_id}_{date_range}.png")
            )
            plt.close()


if __name__ == "__main__":
    ve = VolumeEstimator(
        "/home/jc053/GIT/mri-longitudinal-segmentation/data/output/seg_predictions/"
    )
    print("Getting prediction masks.")
    ve.process_files()
    print("Saving data.")
    ve.plot_volumes(
        output_path="/home/jc053/GIT/mri-longitudinal-segmentation/data/output/plots/"
    )
