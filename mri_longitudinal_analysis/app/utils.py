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
    """Allow user to select an input folder for specific pipeline steps."""
    st.subheader("Input Selection")
    
    # Create a more modern UI for folder selection
    folder_path = st.session_state.get(f"folder_path_{step_index}", "")
    
    # Container with border for better visual appearance
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Display current path with placeholder
            folder_display = st.text_input(
                "Input Folder Path", 
                value=folder_path,
                placeholder="Select a folder...",
                key=f"folder_display_{step_index}",
                disabled=True
            )
        
        with col2:
            # Vertically center the browse button
            st.write("")  # Add some space
            if st.button("📂 Browse", key=f"browse_{step_index}", use_container_width=True):
                # Use tkinter to open a folder selection dialog
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes('-topmost', True)  # Bring the dialog to the front
                
                # Open the folder selection dialog
                selected_folder = filedialog.askdirectory()
                root.destroy()
                
                # Update the folder path if a folder was selected
                if selected_folder:
                    st.session_state[f"folder_path_{step_index}"] = selected_folder
                    return selected_folder
    
    # Add a note about folder selection
    if not folder_path:
        st.info("Please select an input folder containing your MRI data files.")
    
    return folder_path
