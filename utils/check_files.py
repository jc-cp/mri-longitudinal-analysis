import os
import re
import sys
import csv
from pathlib import Path

sys.path.append(sys.path.append(str(Path(__file__).resolve().parent.parent)))
from cfg.check_files_cfg import INFERRED_NO_OPS_DIR, ORIGINAL_NO_OPS_DIR, VOLUMES_29_DIR, VOLUMES_60_DIR, REDCAP_FILE


class CheckFiles():

    @staticmethod
    def extract_ids(filenames) -> list:
        id_list = []
        for filename in filenames:
            # Search for a sequence of digits between underscores and at the beginning of the filename.
            match = re.search(r'_(\d+)_', filename) or re.match(r"^(\d+)(?=[._])", filename)
            #print(f"Filename: {filename}, Match: {match}")
            if match:
                #print(f"Match group 1: {match.group(1)}")
                id = match.group(1)
                if len(id) == 6:
                    id = "0" + id
                id_list.append(id)
            else:
                id_list.append(None)
        return id_list

    @staticmethod
    def find_csv_filenames(path_to_dir, suffix=".csv") -> list:
        filenames = os.listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    @staticmethod
    def find_subfolders(path_to_dir) -> list:
        contents = os.listdir(path_to_dir)
        subfolders = [
            item if len(item) != 6 else "0" + item
            for item in contents
            if os.path.isdir(os.path.join(path_to_dir, item))
        ]
        return subfolders

    @staticmethod
    def compare_ids(list1, list2) -> list:
        return list(set(list1) ^ set(list2))

    def check_files_harddrive(self):
        curated_no_ops = ORIGINAL_NO_OPS_DIR
        csv_files_an225 = self.find_csv_filenames(curated_no_ops)
        subfolders_an225 = self.find_subfolders(curated_no_ops)
        ids_an225 = self.extract_ids(csv_files_an225)

        inference_path = INFERRED_NO_OPS_DIR
        csv_files_inf = self.find_csv_filenames(inference_path)
        ids_inf = self.extract_ids(csv_files_inf)

        diff_csv_files = self.compare_ids(ids_an225, ids_inf)
        diff_folders_csv_files = self.compare_ids(ids_an225, subfolders_an225)

        print("Number of csv files in an225 folder", len(csv_files_an225))
        print("Number of subfolders in an225 folder", len(subfolders_an225))
        print("Mismatching folders / csv's", diff_folders_csv_files)

        print("Number of parsed files", len(csv_files_inf))
        print("Number of different files", len(diff_csv_files))
        print("Mismatching csv's (aka not processed ones)", diff_csv_files)

    def check_files_segmentation(self):    
        files60 = [f for f in os.listdir(VOLUMES_60_DIR) if os.path.isfile(os.path.join(VOLUMES_60_DIR, f))]
        files29 = [f for f in os.listdir(VOLUMES_29_DIR) if os.path.isfile(os.path.join(VOLUMES_29_DIR, f))]

        all_files = files60 + files29
        print("Number of files with a plot: ", len(all_files))
        
        extracted_ids = self.extract_ids(all_files)
        print("Number of extracted IDs from plots: ", len(extracted_ids))
        #print("List of extracted IDs from plots:", extracted_ids)

        csv_ids = []
        csv_file_path = REDCAP_FILE
        with open(csv_file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                id = row['BCH MRN']
                if len(id) == 6:
                    id = "0" + id

                csv_ids.append(id)
        
        print("Number of csv IDs:", len(csv_ids))
        #print("List of extracted IDs from csv:", csv_ids)

        difference = self.compare_ids(extracted_ids, csv_ids)
        print(f"{len(difference)} files not having plots. Exact files:", difference)


if __name__ == "__main__":
    cf = CheckFiles()
    print("-----------------------------------------------------------------")
    print("Checking mismatching files between hard drive and processed ones!")
    print("-----------------------------------------------------------------")
    cf.check_files_harddrive()
    
    print("-----------------------------------------------------------------")
    print("Checking mismatching files between gen. plots and segmentations! ")
    print("-----------------------------------------------------------------")
    cf.check_files_segmentation()
