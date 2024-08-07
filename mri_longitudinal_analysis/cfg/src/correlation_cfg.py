"""Config file for the correlation analysis script."""
from pathlib import Path

# Cohort definition
COHORT = "JOINT"    # "BCH" or "CBTN" or "JOINT"
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
    "bch": CLINICAL_CSV_BCH,
    "cbtn": CLINICAL_CSV_CBTN
}

VOLUMES_DATA_PATHS = {
    "bch": VOLUMES_BCH,
    "cbtn": VOLUMES_CBTN,
    "joint": VOLUMES_JOINT
}
OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output")
OUTPUT_DIR_CORRELATIONS = OUTPUT_DIR / f"correlation_plots_{COHORT.lower()}"
OUTPUT_DIR_STATS = OUTPUT_DIR / f"correlation_stats_{COHORT.lower()}"


# Step 1
SENSITIVITY = False
SENSITIVITY_THRESHOLD = 2

# Step 2
PROPENSITY = False
CALIPER = 0.1  # usually between 0.05 and 0.2

# Step 3
ANALYSIS_PRE_TREATMENT = True

PROGRESSION_THRESHOLD = 1.25  # +25% volume change threshold that defines progression on normalized volume
REGRESSION_THRESHOLD = 0.75   # -25% volume change threshold that defines regression on normalized volume
CHANGE_THRESHOLD = 10  # % volume change threshold for stability index / time gap

# Stability Index Weights
VOLUME_WEIGHT = 0.25
GROWTH_WEIGHT = 0.75



# Step 4
CORRECTION = False
CORRECTION_ALPHA = 0.05

# Step 5
FEATURE_ENG = True


# DICTIONARIES: Symptoms, Locations, Glioma Types

BCH_SYMPTOMS = {
    "incidental": "Asymptomatic (Incidentally Found)",
    "headache": "Headaches",
    "migraine": "Headaches",
    "seizure": "Seizures",
    "staring": "Seizures",
    "syncopal": "Neurological deficits",
    "vertigo": "Neurological deficits",
    "scoliosis": "Neurological deficits",
    "curve": "Neurological deficits",
    "foot": "Neurological deficits",
    "developmental": "Developmental delay",
    "macrocephaly": "Developmental delay",
    "hydrocephalus": "Developmental delay",
    "circumference": "Developmental delay",
    "motor": "Developmental delay",
    "craniosynostosis": "Developmental delay",
    "visual": "Visual deficits",
    "diplopia": "Visual deficits",
    "neurofibromatosis": "Visual deficits",
    "eye": "Visual deficits",
    "optic": "Visual deficits",
    "nystagmus": "Visual deficits",
    "proptosis": "Visual deficits",
    "vision": "Visual deficits",
    "ADHD": "Other",
    "vomitting": "Other",
    "vomited": "Other",
    "obesity": "Other",
    "sinusitis": "Other",
    "numbness": "Other",
}

CBTN_SYMPTOMS = {
    "headaches": "Headaches",
    "hydrocephalus": "Headaches",
    "emesis" : "Headaches",
    "seizure": "Seizures",
    "neurological": "Neurological deficits",
    "behavior": "Neurological deficits",
    "developmental": "Developmental delay",
    "endocrinopathy": "Developmental delay",
    "visual": "Visual deficits",
    "other": "Other",
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
    "meninges": "Meninges / Suprasellar",
    "suprasellar": "Meninges / Suprasellar",
    "thalamic": "Basal Ganglia / Thalamus",
    "thalamus": "Basal Ganglia / Thalamus",
    "basal": "Basal Ganglia / Thalamus",
    "midbrain": "Brainstem",
    "centrum": "Brainstem",
    "tectum": "Brainstem",
    "tectal": "Brainstem",
    "cervicomedullary": "Brainstem",
    "stem": "Brainstem",
    "ventricle": "Ventricles",
    "ventricular": "Ventricles",
    "optic": "Meninges / Suprasellar",
    "spinal": "Other",
    "midline": "Other",
    "pineal": "Other",
}

CBTN_LOCATION = {
    "Ventricles": "Ventricles",
    "Basal": "Basal Ganglia / Thalamus",
    "Thalamus": "Basal Ganglia / Thalamus",
    "Stem": "Brainstem",
    "Cerebellum": "Cerebellum",
    "Parietal Lobe": "Cortical",
    "Frontal Lobe": "Cortical",
    "Temporal Lobe": "Cortical",
    "Occipital Lobe": "Cortical",
    "Meninges": "Meninges / Suprasellar",
    "Suprasellar": "Meninges / Suprasellar",
    "Optic": "Meninges / Suprasellar",
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
