"""
Utility functions for the MRI Longitudinal Analysis Pipeline GUI.

This module provides helper functions for displaying outputs and managing the GUI.
"""

import os
import glob
import streamlit as st
from pipeline import PIPELINE_STEPS

def display_step_output(step_index):
    """Display the output visualizations for a specific pipeline step."""
    step = PIPELINE_STEPS[step_index]
    
    # Get the correct paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(app_dir))
    output_dir = os.path.join(project_root, "data", "output", step["output_dir"])
    
    # Display example outputs if available (for steps that haven't been run yet)
    if "example_outputs" in step and st.session_state.pipeline.get_step_status(step_index) == "pending":
        st.markdown("### Output")
        
        # Get the width from the step parameters
        image_width = step.get("example_width", 400)
        
        example_images = [img for img in step["example_outputs"] if os.path.exists(img)]
        
        if example_images:
            for i in range(0, len(example_images), 2):
                cols = st.columns([1, 3, 3, 1])  # Use 4 columns for better centering
                
                # First image in the second column
                with cols[1]:
                    if i < len(example_images):
                        img_path = example_images[i]
                        st.image(img_path, caption=os.path.basename(img_path), width=image_width)
                
                # Second image in the third column
                with cols[2]:
                    if i + 1 < len(example_images):
                        img_path = example_images[i + 1]
                        st.image(img_path, caption=os.path.basename(img_path), width=image_width)
        else:
            st.info("Example outputs are defined but image files not found. Add them to the app/images directory.")
    
    # Check if output directory exists for actual outputs
    if not os.path.exists(output_dir):
        if st.session_state.pipeline.get_step_status(step_index) in ["completed", "running"]:
            st.warning(f"Output directory not found: {output_dir}")
        return
    
    # Find image files in the output directory
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(glob.glob(os.path.join(output_dir, f"*{ext}")))
    
    if not image_files:
        if st.session_state.pipeline.get_step_status(step_index) in ["completed", "running"]:
            st.info("No visualization outputs found for this step yet.")
        return
    
    # Display actual output images
    st.subheader("Visualization Outputs")
    
    # Get the width from the step parameters
    output_img_width = step.get("output_width", 400)
    
    # Create tabs for different categories of images
    image_categories = categorize_images(image_files)
    
    if len(image_categories) > 1:
        tabs = st.tabs(list(image_categories.keys()))
        
        for i, (category, files) in enumerate(image_categories.items()):
            with tabs[i]:
                display_image_grid(files, width=output_img_width)
    else:
        # If only one category, display without tabs
        display_image_grid(image_files, width=output_img_width)

def display_image_grid(image_files, width=400):
    """Display images in a responsive grid with specified width."""
    # Sort files by name for consistent display
    image_files = sorted(image_files)
    
    # Display images in a 2-column grid
    for i in range(0, len(image_files), 2):
        cols = st.columns([1, 3, 3, 1])  # Use 4 columns for better centering
        
        # First image in the second column
        with cols[1]:
            if i < len(image_files):
                img_path = image_files[i]
                st.image(img_path, caption=os.path.basename(img_path), width=width)
        
        # Second image in the third column
        with cols[2]:
            if i + 1 < len(image_files):
                img_path = image_files[i + 1]
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
    """Display a folder selection widget for input data."""
    if step_index == 0:  # Image Preprocessing
        st.subheader("Select Input Folder")
        st.info("Please select the folder containing your MRI images.")
        folder_path = st.text_input("Input folder path", 
                                    placeholder="/path/to/your/mri/images",
                                    key=f"folder_input_{step_index}")
        return folder_path
    
    elif step_index == 1:  # Tumor Segmentation
        st.subheader("Select Preprocessed Images Folder")
        st.info("Please select the folder containing your preprocessed MRI images.")
        folder_path = st.text_input("Preprocessed images folder path", 
                                    placeholder="/path/to/your/preprocessed/images",
                                    key=f"folder_input_{step_index}")
        return folder_path
    
    elif step_index == 2:  # Volume Estimation
        st.subheader("Select Segmentation Files Folder")
        st.info("Please select the folder containing your segmentation files.")
        folder_path = st.text_input("Segmentation files folder path", 
                                    placeholder="/path/to/your/segmentation/files",
                                    key=f"folder_input_{step_index}")
        return folder_path
    
    return None