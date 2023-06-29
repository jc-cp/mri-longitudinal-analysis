import os
import re
import sys
from pathlib import Path

sys.path.append(sys.path.append(str(Path(__file__).resolve().parent.parent)))
from cfg.check_files_cfg import INFERRED_NO_OPS_DIR, ORIGINAL_NO_OPS_DIR


def extract_ids(filenames) -> list:
    id_list = []
    for filename in filenames:
        match = re.match(r"(\d+)(?=[._])", filename)
        if match:
            id = match.group(0)
            if len(id) == 6:
                id = "0" + id
            id_list.append(id)
        else:
            id_list.append(None)
    return id_list


def find_csv_filenames(path_to_dir, suffix=".csv") -> list:
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def find_subfolders(path_to_dir) -> list:
    contents = os.listdir(path_to_dir)
    subfolders = [
        item if len(item) != 6 else "0" + item
        for item in contents
        if os.path.isdir(os.path.join(path_to_dir, item))
    ]
    return subfolders


def compare_ids(list1, list2) -> list:
    return list(set(list1) ^ set(list2))


# an225 files
curated_no_ops = ORIGINAL_NO_OPS_DIR
csv_files_an225 = find_csv_filenames(curated_no_ops)
subfolders_an225 = find_subfolders(curated_no_ops)
ids_an225 = extract_ids(csv_files_an225)

inference_path = INFERRED_NO_OPS_DIR
csv_files_inf = find_csv_filenames(inference_path)
ids_inf = extract_ids(csv_files_inf)

diff_csv_files = compare_ids(ids_an225, ids_inf)
diff_folders_csv_files = compare_ids(ids_an225, subfolders_an225)

print("Number of csv files in an225 folder", len(csv_files_an225))
print("Number of subfolders in an225 folder", len(subfolders_an225))
print("Mismatching folders / csv's", diff_folders_csv_files)

print("Number of parsed files", len(csv_files_inf))
print("Number of different files", len(diff_csv_files))
print("Mismatching csv's (aka not processed ones)", diff_csv_files)
