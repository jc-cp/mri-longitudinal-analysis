"""Config file for the correlation analysis script."""
from pathlib import Path

CLINICAL_CSV = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_filtered_pruned_treatment_513.csv"
)

VOLUMES_CSV = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/accepted/pre_treatment/output/time_series_csv_kernel_smoothed"
)

COHORT = "CBTN"  # "BCH" or "CBTN"
SAMPLE_SIZE = 45  # BCH: 62, CBTN: 45 -> for plotting trajectories, number of cohort


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

ANGLE = False
PROGRESSION_THRESHOLD = 10  # angle value that defines progression
HIGH_RISK_THRESHOLD = 25  # angle value that defines high increase
STABILITY_THRESHOLD = 2  # angle value that defines stability

VOLUME_WEIGHT = 0.25
GROWTH_WEIGHT = 0.75
CHANGE_THRESHOLD = 25  # % volume change threshold for stability index


# Step 4
CORRECTION = False
CORRECTION_ALPHA = 0.05

# Step 5
FEATURE_ENG = True


# DICTIONARIES: Symptoms, Locations, Glioma Types

BCH_SYMPTOMS = {
    "incidental": "No symptoms (incident finding)",
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
    "None": "No symptoms (incident finding)",
    "Unavailable": "No symptoms (incident finding)",
    "headaches": "Headaches",
    "seizure": "Seizures",
    "neurological": "Neurological deficits",
    "developmental": "Developmental delay",
    "visual": "Visual deficits",
    "behavior": "Other",
    "endocrinopathy": "Other",
    "hydrocephalus": "Other",
    "other": "Other",
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
    # "suprasellar": "Meninges / Suprasellar", # grouping for regression
    # "suprasellar": "Other",
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
    "optic": "Other",
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
    "Optic": "Other",
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
    "neuro": "Glial-neuronal",
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
    "Time to Treatment": "int",
    "Histology": "category",
}

BCH_DATETIME_COLUMNS = [
    "Age at First Diagnosis",
    "Age at First Treatment",
    #"Age at First Progression",
    "Age at Last Clinical Follow-Up",
]

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
    "Time to Treatment": "int",
    "Histology": "category",
}

CBTN_DATETIME_COLUMNS = [
    "Age at First Diagnosis",
    "Age at First Treatment",
    #"Age at First Progression",
    "Age at Last Clinical Follow-Up",
]
