"""
Configuration file for the logistic regressions and the correlation analysis. 
"""
from pathlib import Path
from cfg.src.cohort_creation_cfg import NUMERICAL_VARS, CATEGORICAL_VARS

COHORT = "JOINT"    # DF_BCH or CBTN or JOINT
COHORT_DATAFRAME = Path("/home/juanqui55/git/mri-longitudinal-analysis/data/output/02_trajectories/JOINT_trajectories_cohort_data_features.csv")
OUTPUT_DIR = Path("/home/juanqui55/git/mri-longitudinal-analysis/data/output/03_lr_and_correlations")
ENDPOINT = "Volumetric" # Composite == Clinical 
OUTCOME_VAR = f"Patient Classification Binary {ENDPOINT}"

# Variable Types
CATEGORICAL_VARS = CATEGORICAL_VARS
NUMERICAL_VARS = NUMERICAL_VARS
# Only Relevant variables for LR; should be baseline variables
LR_VARS = [
            "Location",
            #"Symptoms",
            "BRAF Status",
            "Sex",
            "Age Group at Diagnosis",
            "Baseline Volume cm3",
            #"Age at First Diagnosis (Years)",
        ]

LR_COMBINATIONS = [LR_VARS] # add other combinations here in the list

