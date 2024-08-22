"""
Config file for the cohort creation script.
"""
from pathlib import Path

# Cohort definition
COHORT = "JOINT"    # "DF_BCH" or "CBTN" or "JOINT"
SAMPLE_SIZE = 99  # BCH: 56, CBTN: 43, JOINT: 99 -> for plotting trajectories, number of cohort

# Directories and files
CLINICAL_CSV_BCH = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/bch_filtering_68_.csv"
)
CLINICAL_CSV_CBTN = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_filtered_pruned_treatment_513.csv")

VOLUMES_BCH = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/final/pre_treatment/output/time_series/moving_average")
VOLUMES_CBTN = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/accepted/pre_treatment/output/time_series/moving_average")
VOLUMES_JOINT = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/final_dataset/output/time_series/moving_average")


CLINICAL_CSV_PATHS = {
    "df_bch": CLINICAL_CSV_BCH,
    "cbtn": CLINICAL_CSV_CBTN
}

VOLUMES_DATA_PATHS = {
    "df_bch": VOLUMES_BCH,
    "cbtn": VOLUMES_CBTN,
    "joint": VOLUMES_JOINT
}
OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/01_cohort_data")
OUTPUT_STATS_FILE = OUTPUT_DIR / f"{COHORT.lower()}_cohort_stats.txt"
COHORT_TABLE_FILE = OUTPUT_DIR / f"{COHORT.lower()}_cohort_table.csv" # can be a .csv or .txt

# DICTIONARIES: Symptoms, Locations, Glioma Types
BCH_SYMPTOMS = {
    "incidental": "Asymptomatic (Incidentally Found)",
    "headache": "Sympomatic",#"Headaches",
    "migraine": "Sympomatic",#"Headaches",
    "seizure": "Sympomatic",#"Seizures",
    "staring": "Sympomatic",#"Seizures",
    "syncopal": "Sympomatic",#"Neurological deficits",
    "vertigo": "Sympomatic",#"Neurological deficits",
    "scoliosis": "Sympomatic",#"Neurological deficits",
    "curve": "Sympomatic",#"Neurological deficits",
    "foot": "Sympomatic",#"Neurological deficits",
    "developmental": "Sympomatic",#"Developmental delay",
    "macrocephaly": "Sympomatic",#"Developmental delay",
    "hydrocephalus": "Sympomatic",#"Developmental delay",
    "circumference": "Sympomatic",#"Developmental delay",
    "motor": "Sympomatic",#"Developmental delay",
    "craniosynostosis": "Sympomatic",#"Developmental delay",
    "visual": "Sympomatic",#"Visual deficits",
    "diplopia": "Sympomatic",#"Visual deficits",
    "eye": "Sympomatic",#"Visual deficits",
    "optic": "Sympomatic",#"Visual deficits",
    "nystagmus": "Sympomatic",#"Visual deficits",
    "proptosis": "Sympomatic",#"Visual deficits",
    "vision": "Sympomatic",#"Visual deficits",
    "ADHD": "Sympomatic",#"Other",
    "vomitting": "Sympomatic",#"Other",
    "vomited":"Sympomatic", #"Other",
    "obesity": "Sympomatic",#"Other",
    "sinusitis": "Sympomatic",#"Other",
    "numbness": "Sympomatic",#"Other",
}

CBTN_SYMPTOMS = {
    "headaches": "Sympomatic",#"Headaches",
    "hydrocephalus": "Sympomatic",#"Headaches",
    "emesis" : "Sympomatic",#"Headaches",
    "seizure": "Sympomatic",#"Seizures",
    "neurological": "Sympomatic",#"Neurological deficits",
    "behavior": "Sympomatic",#"Neurological deficits",
    "developmental": "Sympomatic",#"Developmental delay",
    "endocrinopathy": "Sympomatic",#"Developmental delay",
    "visual": "Sympomatic",#"Visual deficits",
    "other": "Sympomatic",#"Other",
    "None": "Asymptomatic (Incidentally Found)",
    "Unavailable": "Asymptomatic (Incidentally Found)",
}

BCH_LOCATION = {
    "posterior fossa": "Cerebellum",
    "cerebel": "Cerebellum",
    "vermis": "Cerebellum",
    "temporal": "Cortical",
    "frontal": "Cortical",
    "parietal": "Cortical",
    "sylvian": "Cortical",
    "meninges": "Midline", #"Meninges / Suprasellar",
    "suprasellar": "Midline", #"Meninges / Suprasellar",
    "thalamic": "Basal Ganglia / Thalamus",
    "thalamus": "Basal Ganglia / Thalamus",
    "basal": "Basal Ganglia / Thalamus",
    "midbrain": "Midline", #"Brainstem",
    "centrum": "Midline", #"Brainstem",
    "tectum": "Midline", #"Brainstem",
    "tectal": "Midline", #"Brainstem",
    "cervicomedullary": "Midline", #"Brainstem",
    "stem": "Midline", #"Brainstem",
    "ventricle": "Midline", #"Ventricles",
    "ventricular":"Midline", #"Ventricles",
    "optic": "Midline", #"Meninges / Suprasellar",
    "spinal": "Other",
    "midline": "Other",
    "pineal": "Other",
}

CBTN_LOCATION = {
    "Ventricles": "Midline", #"Ventricles",
    "Basal": "Basal Ganglia / Thalamus",
    "Thalamus": "Basal Ganglia / Thalamus",
    "Stem": "Midline", #"Brainstem",
    "Cerebellum": "Cerebellum",
    "Parietal Lobe": "Cortical",
    "Frontal Lobe": "Cortical",
    "Temporal Lobe": "Cortical",
    "Occipital Lobe": "Cortical",
    "Meninges": "Midline", #"Meninges / Suprasellar",
    "Suprasellar": "Midline", #"Meninges / Suprasellar",
    "Optic": "Midline", #"Meninges / Suprasellar",
    "Spinal": "Other",
    "Other": "Other",
}

BCH_GLIOMA_TYPES = {
    "astro": "Astrocytoma",
    "JPA": "Astrocytoma",
    "gang": "Ganglioglioma",
    "glio": "Glial-neuronal glioma",
    "glial": "Glial-neuronal glioma",
    "neuro": "Glial-neuronal glioma",
    "DNET": "Glial-neuronal glioma",
    "neo": "Glial-neuronal glioma",
    "pseudo": "Other",
    "oligodendroglioma": "Other",
    # "low": "Plain Low Grade Glioma",
    # "tectal": "Plain Low Grade Glioma",
    "low": "Other",
    "tectal": "Other",

}

CBTN_GLIOMA_TYPES = {
    "neuro": "Glial-neuronal glioma",
    "gang": "Ganglioglioma",
    "astro": "Astrocytoma",
}

BCH_DTYPE_MAPPING = {
    "BCH MRN": "string",
    "Location": "category",
    "Symptoms": "category",
    "Sex": "category",
    "BRAF Status": "category",
    "Treatment Type": "category",
    #"Tumor Progression": "category",
    "Received Treatment": "category",
    "Follow-Up Time": "int",
    "Histology": "category",
    
}

BCH_DATETIME_COLUMNS = [
    "Age at First Diagnosis",
    "Age at First Treatment",
    #"Age at First Progression",
    "Age at Last Clinical Follow-Up",]

CBTN_DTYPE_MAPPING = {
    "CBTN Subject ID": "string",
    "Location": "category",
    "Symptoms": "category",
    "Sex": "category",
    "BRAF Status": "category",
    "Treatment Type": "category",
    #"Tumor Progression": "category",
    "Received Treatment": "category",
    # "Follow-up Time": "int",          # provided through the volume data csv's and the ages
    "Histology": "category",
}

CBTN_DATETIME_COLUMNS = [
    "Age at First Diagnosis",
    "Age at First Treatment",
    #"Age at First Progression",
    "Age at Last Clinical Follow-Up",
]

CATEGORICAL_VARS = [
            "Location",
            "Symptoms",
            "Histology",
            "Treatment Type",
            "Age Group at Diagnosis",
            "BRAF Status",
            "Sex",
            #"Tumor Classification",
            "Received Treatment",
            "Change Speed",
            "Change Type",
            "Change Trend",
            "Change Acceleration",
        ]

NUMERICAL_VARS = [
            "Age",
            "Age Median",
            'Age at First Diagnosis (Years)',
            "Age at Last Clinical Follow-Up", # should be in years
            "Days Between Scans",
            "Days Between Scans Median",
            "Follow-Up Time",
            "Follow-Up Time Median",
            "Volume",
            "Volume Median",
            "Normalized Volume",
            "Baseline Volume",
            "Volume Change",
            "Volume Change Median",
            "Volume Change Rate",
            "Volume Change Rate Median",
        ]