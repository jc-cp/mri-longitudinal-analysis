import pandas as pd
import os 
import shutil
import math
from cfg.utils import cbtn_parsing_cfg
import numpy as np
import re

def read_csv() -> (list, list):
    path_cbtn = cbtn_parsing_cfg.PATH_CSV
    df_cbtn = pd.read_csv(path_cbtn)

    # Add here filtering and assertation criteria
    assert df_cbtn["Event Type"] == "Initial CNS Tumor"
    assert df_cbtn["Flywheel Imaging Available"] == "Yes"
    assert df_cbtn['Diagnoses'] == ('Low-grade glioma/astrocytoma (WHO grade I/II)'
                                    or 'Ganglioglioma' or 'Glial-neuronal tumor NOS'
                                    or 'Low-grade glioma/astrocytoma (WHO grade I/II)'
                                    or 'Ganglioglioma; Low-grade glioma/astrocytoma (WHO grade I/II)')
    assert df_cbtn['Follow Up Event'] == 'Initial Diagnosis'
    assert not df_cbtn['Age at Event Days'].any().isnan()
    assert df_cbtn['Specimen Collection Origin'] == ('Initial CNS Tumor Surgery' 
                                                     or 'Not Applicable')
    assert len(df_cbtn) == 536

    #df_cbtn = df_cbtn[df_cbtn['Follow Up Event'] == 'Initial Diagnosis']
    #df_cbtn = df_cbtn[df_cbtn['Surgery'] == 'Yes']
    #df_cbtn = df_cbtn[df_cbtn['Diagnoses'] == 'Low-grade glioma/astrocytoma (WHO grade I/II)'] 
    #df_cbtn = df_cbtn[df_cbtn['Age at Event Days'] != 'Unavailable']
    #df_cbtn = df_cbtn.reset_index()

    ids = sorted(df_cbtn['CBTN Subject ID'].tolist())
    # TODO: do check that ids are unique 

    ages= []
    for patient_id in ids:
        for _, patient_item in enumerate(df_cbtn['CBTN Subject ID']):
            if patient_id == patient_item:
                ages.append(df_cbtn.loc[i,'Age at Event Days'])



    path_mr_cbtn = []
    path_mrs = cbtn_parsing_cfg.PATH_IMAGES
    for patient_id in ids:
        for patient_item in os.listdir(path_mrs):
            if patient_id == patient_item:
                path_mr_cbtn.append(os.path.join(path_mrs, patient_item))
    return ages, path_mr_cbtn



# Function to extract the number before 'd' in the string
def extract_number(s):
    match = re.match(r"(\d+)d", s)
    if match:
        number = int(match.group(1))
        return number
    else:
        return None

list_ages = []
nmidonama = []
for i, item in enumerate(path_mr_cbtn):
    tempo = []
    
    for folder in os.listdir(item):
        if extract_number(folder) <= int(ages[i]):
            tempo.append(folder)


    list_ages.append([extract_number(temp) for temp in tempo]) 

    if tempo == []:
        nmidonama.append(item[27:])

nmidonama = pd.DataFrame(nmidonama)
nmidonama.to_csv('mraftsurg.csv', index=False)

#It seems you dont need this part, it is for choosing closest age 
def closest_number(target, num_list):
    return min(num_list, key=lambda x: abs(x - target))

choosen_ages = []
for i, sublist in enumerate(list_ages):
    if sublist == []:
        choosen_ages.append([])

    else:
        for x in sublist:
            choosen_ages.append(closest_number(int(ages[i]), sublist))
            break 


New_path_cbtn = cbtn_parsing_cfg.NEW_PATH_IMAGES
meta_path = cbtn_parsing_cfg.PATH_METADATA
metadata = pd.read_excel(meta_path, header = 1, engine='openpyxl')


#Getting T2 sequences

c1 = 0
c2 = 0 
seqs =[]
session_label = []
for i, item in enumerate(path_mr_cbtn):
    for j, row in enumerate(metadata['subject_label']):
        if choosen_ages[i]!= []:
            condition = (metadata['subject_label'] == item[27:]) & (metadata['age_at_imaging_in_days'] == choosen_ages[i])
            filtered_df = metadata[condition]
            filtered_df = filtered_df.reset_index()
            session_label.append(filtered_df.loc[0,'session_label'])
            if filtered_df.shape[0] > 0:
                # print(filtered_df)
                label_n = filtered_df['acquisition_label'].tolist()

                containing_1 = [seq for seq in label_n if 't2_tse_tra_p2' in seq]
                if containing_1:
                    seqs.append(containing_1)
                    break
                else:
                    containing_2 = [seq for seq in label_n if 't2' in seq and 'tra' in seq]
                    if containing_2:
                        seqs.append(containing_2)
                        break
                    else:
                        containing_3 = [seq for seq in label_n if 'T2' in seq and 'AXIAL' in seq]
                        if containing_3:
                            seqs.append(containing_3)
                            break
                        else:
                            containing_4 = [seq for seq in label_n if 'T2' in seq and 'Ax' in seq]
                            if containing_4:
                                seqs.append(containing_4)
                                break
                            else:
                                containing_5 = [seq for seq in label_n if 't2' in seq]
                                if containing_5:
                                    seqs.append(containing_5)
                                    break
                                else:
                                    containing_6 = [seq for seq in label_n if 'T2' in seq]
                                    if containing_6:
                                        seqs.append(containing_6)
                                        break
                                    else:
                                        seqs.append([])
                                        c2+=1
                                        break
                break
        else:
            seqs.append([])
            session_label.append([])
            c1+=1
            break


def count_nan(lst):
    count = 0
    for item in lst:
        if isinstance(item, list):  # Check if it is a list
            count += count_nan(item)  # Recursively count in the sublist
        elif item == 'nan':
            count += 1
    return count

nan_count = count_nan(seqs)
print(nan_count)

for i, item in enumerate(path_mr_cbtn):
    if len(seqs[i])!= 0:
        print(item, session_label[i],seqs[i][0])
        image = os.listdir(os.path.join(item,session_label[i],seqs[i][0]))
        shutil.copy(os.path.join(item,session_label[i],seqs[i][0],image[0]),os.path.join(New_path_cbtn,f'{item[27:]}.nii.gz'))
    else:
        pass

def main():
    read_csv()


if __name__=='__main__':
    main()