"""
Pipeline management for MRI Longitudinal Analysis.

This module defines the pipeline steps, their dependencies, and handles execution.
"""

import os
import subprocess
import time
import threading
import streamlit as st
import queue
from threading import Lock

# Define a global queue for thread-safe communication
output_queue = queue.Queue()
queue_lock = Lock()

app_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(app_dir, "images")

# Ensure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Define the pipeline steps with descriptions derived from script documentation
PIPELINE_STEPS = [
    {
        "name": "Step 0.1: Image Preprocessing",
        "script": "mri_preprocess_3d.py",
        "description": """
        **Image Preprocessing**
        
        This step preprocesses the images to prepare them for volume estimation.
        It performs brain extraction, normalization, and resampling to ensure consistent dimensions.
        
        **Inputs:**
        - MRI images through folder selection

        **Outputs:**
        - Preprocessed images
        - Normalized, resampled, registered and brain-extracted images
        """,
        "output_dir": "00_preprocessed_images",
        "illustration": os.path.join(images_dir, "step/image_preprocessing.png"),
        "illustration_width": 500,
    },
    {
        "name": "Step 0.2: Tumor Segmentation",
        "script": "00_tumor_segmentation.py",
        "description": """
        **Tumor Segmentation**
        
        This step performs tumor segmentation using a deep learning model. We have based this anylasis on the well established and validated model from the paper: Stepwise transfer learning for expert-level pediatric brain tumor MRI segmentation in a limited data scenario.
        This work is also from our work group at the AI in Medicine Lab at Harvard Medical School.
        
        """,
        "output_dir": "00_segmentation_plots",
        "illustration": os.path.join(images_dir, "step/segmentation.png"),
        "illustration_width": 500,
    },
    {
        "name": "Step 0.3: Volume Estimation",
        "script": "00_volume_estimation.py",
        "description": """
        **Volume Estimation**
        
        This step processes segmentation files to estimate tumor volumes over time. 
        It calculates volumetric measurements from the segmentation masks and 
        visualizes the volume changes for each patient, creating not only individual trayectories, but also a longitudinal database per patient.
        
        **Inputs:**
        - Segmentation files through folder selection

        **Outputs:**
        - Volume trajectory plots for each patient
        - Longitudinal database per patient
        - Volumetric statistics per patient
        """,
        "output_dir": "volume_plots",
        "illustration": os.path.join(images_dir, "volume_estimation.png"),
        "example_outputs": [
            os.path.join(images_dir, "output/trajectory_example.png"),
        ],
        "example_width": 800,
    },
    {
        "name": "Step 1: Joint Cohort Creation",
        "script": "01_cohort_creation.py",
        "description": """
        **Joint Cohort Creation**
        
        This step creates a joint cohort of patients with the selected clinical data. This is very important since the features between the cohorts may be differnt and need to be harmonized. We use a matching AI algorithm to find the relevant columns automatically and create the expected output format for further analysis.
        **Outputs:**
        - Cluster visualization plots (UMAP, t-SNE)
        - Heatmaps of cluster characteristics
        - Patient data EDA visualizations
        """,
        "output_dir": "clustering_plots",
        "illustration": os.path.join(images_dir, "cohort_creation.png"),
        "example_outputs": [
            os.path.join(images_dir, "umap_example.png"),
            os.path.join(images_dir, "heatmap_example.png")
        ],
        "example_width": 600,
        "output_width": 500,
    },
    {
        "name": "Step 2: Trajectory Classification",
        "script": "02_trajectories_classification.py",
        "description": """
        **Trajectory Classification**
        
        This step classifies tumor growth trajectories based on volumetric changes.
        It categorizes patients as progressors or non-progressors based on predefined
        criteria and visualizes the different trajectory patterns.
        
        **Outputs:**
        - Classification trajectory plots
        - Progression status distribution
        - Time-to-progression analysis
        """,
        "output_dir": "trajectory_plots",
        "illustration": os.path.join(images_dir, "trajectory_classification.png"),
        "example_outputs": [
            os.path.join(images_dir, "progression_example.png"),
            os.path.join(images_dir, "time_to_progression_example.png")
        ]
    },
    {
        "name": "Step 3: Logistic Regression & Correlations",
        "script": "03_lr_and_correlations.py",
        "description": """
        **Logistic Regression & Correlations Analysis**
        
        This step performs statistical analysis to identify correlations between 
        clinical variables and tumor progression. It includes univariate and 
        multivariate logistic regression to identify risk factors.
        
        **Outputs:**
        - Forest plots for univariate and multivariate analyses
        - Correlation heatmaps
        - Statistical test visualizations
        """,
        "output_dir": "correlation_plots",
        "illustration": os.path.join(images_dir, "correlation_analysis.png"),
        "example_outputs": [
            os.path.join(images_dir, "forest_plot_example.png"),
            os.path.join(images_dir, "correlation_heatmap_example.png")
        ]
    },
    {
        "name": "Step 4: Time-to-Event Analysis",
        "script": "04_time_to_event.py",
        "description": """
        **Time-to-Event Analysis**
        
        This step performs survival analysis using Kaplan-Meier curves and Cox 
        proportional hazards models to analyze time to progression or treatment.
        It identifies factors that influence progression timing.
        
        **Outputs:**
        - Kaplan-Meier survival curves
        - Cox hazard ratio forest plots
        - Time-to-event distribution plots
        """,
        "output_dir": "survival_plots",
        "illustration": os.path.join(images_dir, "survival_analysis.png"),
        "example_outputs": [
            os.path.join(images_dir, "kaplan_meier_example.png"),
            os.path.join(images_dir, "cox_hazard_example.png")
        ]
    },
    {
        "name": "Step 5: Volumetric Forecasting",
        "script": "05_volumetric_forecasting.py",
        "description": """
        **Volumetric Forecasting**
        
        This step performs a hybrid forecasting algorithm based on Autoregressive Integrated Moving Average (ARIMA) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models,
        to predict the future volume development of the tumor.
        
        **Inputs:**
        - Longitudinal database per patient

        **Outputs:**
        - Forecasted volume plots
        """,
        "output_dir": "forecasting_plots",
        "illustration": os.path.join(images_dir, "forecasting.png"),
        "example_outputs": [
            os.path.join(images_dir, "forecast_example.png"),
            os.path.join(images_dir, "prediction_accuracy_example.png")
        ]
    }
]

class Pipeline:
    """Manages the execution and state of the analysis pipeline."""
    
    def __init__(self):
        """Initialize the pipeline state."""
        self.steps_status = ["pending"] * len(PIPELINE_STEPS)
        self.current_step = None
        self.progress_percentage = 0
        
        # Get the correct paths
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(self.app_dir))
        self.src_dir = os.path.join(self.project_root, "src")
        
    def get_step_status(self, step_index):
        """Get the status of a specific pipeline step."""
        if step_index < 0 or step_index >= len(PIPELINE_STEPS):
            return None
        return self.steps_status[step_index]
    
    def can_run_step(self, step_index):
        """Check if a step can be run based on dependencies."""
        # First step can always run
        if step_index == 0:
            return True
        
        # Other steps require previous step to be completed
        return self.steps_status[step_index - 1] == "completed"
    
    def run_step(self, step_index):
        """Run a specific pipeline step."""
        if not self.can_run_step(step_index):
            st.error(f"Cannot run step {step_index}. Previous steps must be completed first.")
            return
        
        self.current_step = step_index
        self.steps_status[step_index] = "running"
        
        # Start the script execution in a separate thread
        thread = threading.Thread(
            target=self._execute_script,
            args=(step_index,)
        )
        thread.daemon = True  # Make thread a daemon so it exits when main thread exits
        thread.start()
    
    def _execute_script(self, step_index):
        """Execute the script for a specific step and update status."""
        try:
            step = PIPELINE_STEPS[step_index]
            script_path = os.path.join(self.src_dir, step["script"])
            
            # Use the global queue instead of session state
            with queue_lock:
                output_queue.put(f"\n🚀 Starting Step {step_index}: {step['name']}...\n")
            
            # Execute the script and capture output
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output to queue
            for line in process.stdout:
                with queue_lock:
                    output_queue.put(line)
            
            # Wait for process to complete
            process.wait()
            
            # Update status based on exit code
            if process.returncode == 0:
                self.steps_status[step_index] = "completed"
                self._update_progress()
                with queue_lock:
                    output_queue.put(f"\n✅ Step {step_index}: {step['name']} completed successfully!\n")
            else:
                self.steps_status[step_index] = "failed"
                with queue_lock:
                    output_queue.put(f"\n❌ Step {step_index}: {step['name']} failed with exit code {process.returncode}\n")
        
        except Exception as e:
            self.steps_status[step_index] = "failed"
            with queue_lock:
                output_queue.put(f"\n❌ Error executing step {step_index}: {str(e)}\n")
    
    def _update_progress(self):
        """Update the overall pipeline progress percentage."""
        completed_steps = self.steps_status.count("completed")
        self.progress_percentage = (completed_steps / len(PIPELINE_STEPS)) * 100

    def omit_step(self, step_index):
        """Skip a specific pipeline step by simulating successful execution."""
        if not self.can_run_step(step_index):
            print(f"Cannot omit step {step_index}. Previous steps must be completed first.")
            return
        
        self.current_step = step_index
        self.steps_status[step_index] = "running"
        
        # Start the mock execution in a separate thread
        thread = threading.Thread(
            target=self._mock_execute_script,
            args=(step_index,)
        )
        thread.daemon = True
        thread.start()

    def _mock_execute_script(self, step_index):
        """Simulate execution of a script with mock output."""
        try:
            step = PIPELINE_STEPS[step_index]
            
            # Use the global queue instead of session state
            with queue_lock:
                output_queue.put(f"\n🔄 Omitting Step {step_index}: {step['name']}...\n")
            
            # Generate mock output based on the step
            mock_output = self._generate_mock_output(step_index)
            
            # Stream mock output to queue with small delays to simulate processing
            for line in mock_output.split('\n'):
                with queue_lock:
                    output_queue.put(line + '\n')
                time.sleep(0.1)  # Small delay between lines
            
            # Mark step as completed
            self.steps_status[step_index] = "completed"
            
            # Update progress percentage
            self._update_progress()
            
        except Exception as e:
            with queue_lock:
                output_queue.put(f"Error in mock execution: {str(e)}\n")
            self.steps_status[step_index] = "pending"

    def _generate_mock_output(self, step_index):
        """Generate mock output for a specific step."""
        if step_index == 0:  # Image Preprocessing
            return """Input directory: /home/juanqui55/git/mri-longitudinal-analysis/data/input/raw_images
Output directory: /home/juanqui55/git/mri-longitudinal-analysis/data/output/00_preprocessed_images
Bias field correction...
Correction progress: 100%|██████████| 970/970 [01:23<00:00, 5.5s/it]
Bias field correction complete!
Resampling...
Resampling progress: 100%|██████████| 970/970 [01:23<00:00, 5.5s/it]
Resampling complete!
Registering test data...
Registration progress: 100%|██████████| 970/970 [01:23<00:00, 5.5s/it]
Registration complete!
Brain Extraction with HD-BET...
Brain Extraction progress: 100%|██████████| 970/970 [01:23<00:00, 5.5s/it]
Brain Extraction complete!
All files processed successfully.
Preprocessing complete!"""
        
        elif step_index == 1:  # Tumor Segmentation
            return """Loading segmentation model...
Model loaded successfully.
Processing files from: /home/juanqui55/git/mri-longitudinal-analysis/data/output/00_preprocessed_images
Segmentation progress: 100%|██████████| 970/970 [45:15<00:00, 9.0s/it]
Saving segmentation masks...
Post-processing segmentations...
All segmentations completed successfully.
Tumor segmentation complete!
Output directory: /home/juanqui55/git/mri-longitudinal-analysis/data/output/00_segmentation_plots"""
        
        elif step_index == 2:  # Volume Estimation
            return """Initializing Volume Estimator:
\tAdjusted the clinical data.
Processing files:
\tNumber of unique patient IDs in the original CSV: 56
\tNumber of unique patient IDs in the original CSV: 43
\tNumber of unique patient IDs in the directory: 99
\tNumber of unique patient IDs in the final CSV: 99
\tNumber of reduced patient IDs: 0
\tNumber of mismatched patient IDs: 0
\tAll files processed.
Generating plots:
\tPlotted raw data!
\tPlotted filtered data!
\tPlotted poly_smoothing data!
\tPlotted kernel_smoothing data!
\tPlotted window_smoothing data!
\tPlotted moving_average data!
\tSaved all plots.
Generating volume comparison:
\tSaved comparison.
Analyzing volume changes:
\t95% CI for volume change: (-1250.45, 1876.32)
\t95% CI for volume change rate per day: -0.87, 1.23
\tAnalyzed volume changes.
Generating time-series csv's.
\tSaved all csv's."""
        
        else:
            return f"Mock execution completed for step {step_index}."