"""
Utility functions for the MRI Longitudinal Analysis Pipeline GUI.

This module provides helper functions for displaying outputs and managing the GUI.
"""

import os
import glob
import streamlit as st
from pipeline import PIPELINE_STEPS
import tkinter as tk
from tkinter import filedialog
import tempfile

def display_step_output(step_index):
    """Display the output visualizations for a specific pipeline step."""
    step = PIPELINE_STEPS[step_index]
    step_status = st.session_state.pipeline.get_step_status(step_index)
    
    # Only show outputs if the step is completed or running
    if step_status not in ["completed", "running"]:
        return
    
    # Get the correct paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(app_dir))
    
    # Check if example_outputs is defined in the step
    if "example_outputs" in step:
        # Display the example outputs
        st.subheader("Visualization Outputs")
        
        # Get the width from the step parameters
        output_img_width = step.get("output_width", 400)
        
        # Filter to only include existing files
        image_files = [img_path for img_path in step["example_outputs"] if os.path.exists(img_path)]
        
        if not image_files:
            st.info("No visualization outputs found for this step.")
            return
        
        # Display the images
        display_image_grid(image_files, width=output_img_width)
    else:
        # If no example_outputs defined, look in the output directory
        output_dir = os.path.join(project_root, "data", "output", step["output_dir"])
        
        # Check if output directory exists
        if not os.path.exists(output_dir):
            st.warning(f"Output directory not found: {output_dir}")
            return
        
        # Find image files in the output directory
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(glob.glob(os.path.join(output_dir, f"*{ext}")))
        
        if not image_files:
            st.info("No visualization outputs found for this step yet.")
            return
        
        # Display actual output images
        st.subheader("Visualization Outputs")
        
        # Get the width from the step parameters
        output_img_width = step.get("output_width", 400)
        
        # Create tabs for different categories of images
        image_categories = categorize_images(image_files)
        
        # Center the output in the container
        with st.container():
            if len(image_categories) > 1:
                tabs = st.tabs(list(image_categories.keys()))
                
                for i, (category, files) in enumerate(image_categories.items()):
                    with tabs[i]:
                        display_image_grid(files, width=output_img_width)
            else:
                # If only one category, display without tabs
                display_image_grid(image_files, width=output_img_width)

def display_image_grid(image_files, width=400):
    """Display images one below the other with specified width."""
    # Sort files by name for consistent display
    image_files = sorted(image_files)
    
    # Display each image in a centered layout, one below the other
    for img_path in image_files:
        # Create a 3-column layout with the middle column wider for centering
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Display the image in the center column
        with col2:
            st.image(img_path, caption=os.path.basename(img_path), width=width)

def categorize_images(image_files):
    """Categorize images based on filename patterns."""
    categories = {"General": []}
    
    # Define patterns to categorize images
    patterns = {
        "correlation": "Correlations",
        "heatmap": "Heatmaps",
        "forest": "Forest Plots",
        "kaplan": "Survival Analysis",
        "trajectory": "Trajectories",
        "volume": "Volume Analysis",
        "classification": "Classification",
        "forecast": "Forecasting",
        "prediction": "Predictions",
        "distribution": "Distributions",
        "time_gap": "Time Analysis"
    }
    
    for img_file in image_files:
        filename = os.path.basename(img_file)
        categorized = False
        
        for pattern, category_name in patterns.items():
            if pattern in filename.lower():
                if category_name not in categories:
                    categories[category_name] = []
                categories[category_name].append(img_file)
                categorized = True
                break
        
        if not categorized:
            categories["General"].append(img_file)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def select_input_folder(step_index):
    """Allow user to select an input folder for specific pipeline steps."""
    import tempfile
    
    st.subheader("Input Selection")
    
    # Get the current folder path from session state
    folder_path = st.session_state.get(f"folder_path_{step_index}", "")
    
    # Container with border for better visual appearance
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            # Text input for manual path entry
            folder_path = st.text_input(
                "Input Folder Path", 
                value=folder_path,
                placeholder="Enter folder path...",
                key=f"folder_input_{step_index}"
            )
        
        st.session_state[f"folder_path_{step_index}"] = folder_path
    
    # Add a button to use default path
    if st.button("Use Default Path", key=f"default_path_{step_index}"):
        if step_index == 0:  # Image Preprocessing
            default_path = "/home/juanqui55/git/mri-longitudinal-analysis/data/input/raw_scans"
        elif step_index == 1:  # Tumor Segmentation
            default_path = "/home/juanqui55/git/mri-longitudinal-analysis/data/output/00_preprocessed_images"
        elif step_index == 2:  # Volume Estimation
            default_path = "/home/juanqui55/git/mri-longitudinal-analysis/data/output/00_segmentation_masks"
        else:
            default_path = ""
        
        st.session_state[f"folder_path_{step_index}"] = default_path
        return default_path
    
    # Add a note about folder selection
    if not folder_path:
        st.info("Please enter a folder path containing your MRI data files.")
    
    return folder_path

def select_cohort_inputs(step_index):
    """Special input selection for cohort creation step."""
    import tempfile
    
    st.subheader("Input Selection")
    
    # Get session state values
    volume_folder = st.session_state.get(f"volume_folder_{step_index}", "")
    clinical_file1 = st.session_state.get(f"bch_filtering_68_{step_index}", "")
    clinical_file2 = st.session_state.get(f"cbtn_filtered_pruned_treatment_513{step_index}", "")
    
    # Volume folder selection
    st.markdown("#### Volume Data")
    with st.container(border=True):
        # Text input for manual path entry
        volume_folder = st.text_input(
            "Volume Data Folder Path", 
            value=volume_folder,
            placeholder="Enter volume data folder path...",
            key=f"volume_input_{step_index}"
        )
        
        st.session_state[f"volume_folder_{step_index}"] = volume_folder
    
    # Clinical data file 1
    st.markdown("#### Clinical Data - Cohort 1")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            # Text input for manual path entry
            clinical_file1 = st.text_input(
                "Clinical Data File 1 Path", 
                value=clinical_file1,
                placeholder="Enter clinical data file 1 path...",
                key=f"clinical1_input_{step_index}"
            )
        with col2:
            # Native Streamlit file uploader
            uploaded_file1 = st.file_uploader("Upload CSV", type=["csv"], key=f"uploader1_{step_index}")
            if uploaded_file1 is not None:
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file1.getvalue())
                    clinical_file1 = tmp_file.name
        
        st.session_state[f"clinical_file1_{step_index}"] = clinical_file1
    
    # Clinical data file 2
    st.markdown("#### Clinical Data - Cohort 2")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            # Text input for manual path entry
            clinical_file2 = st.text_input(
                "Clinical Data File 2 Path", 
                value=clinical_file2,
                placeholder="Enter clinical data file 2 path...",
                key=f"clinical2_input_{step_index}"
            )
        with col2:
            # Native Streamlit file uploader
            uploaded_file2 = st.file_uploader("Upload CSV", type=["csv"], key=f"uploader2_{step_index}")
            if uploaded_file2 is not None:
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file2.getvalue())
                    clinical_file2 = tmp_file.name
        
        st.session_state[f"clinical_file2_{step_index}"] = clinical_file2
    
    # Add a note about input requirements
    if not (volume_folder and clinical_file1 and clinical_file2):
        st.info("Please enter paths for all required inputs for cohort creation.")
    
    # Add a button to use default paths
    if st.button("Use Default Paths", key=f"default_paths_{step_index}"):
        st.session_state[f"volume_folder_{step_index}"] = "/home/juanqui55/git/mri-longitudinal-analysis/data/output/00_volume_trajectories"
        st.session_state[f"clinical_file1_{step_index}"] = "/home/juanqui55/git/mri-longitudinal-analysis/data/input/clinical_data_cohort1.csv"
        st.session_state[f"clinical_file2_{step_index}"] = "/home/juanqui55/git/mri-longitudinal-analysis/data/input/clinical_data_cohort2.csv"
        return True
    
    return None
