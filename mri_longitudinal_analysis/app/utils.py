"""
Utility functions for the MRI Longitudinal Analysis Pipeline GUI.

This module provides helper functions for displaying outputs and managing the GUI.
"""

import os
import glob
import time
import streamlit as st
from pipeline import PIPELINE_STEPS

def display_step_output(step_index):
    """Display the output visualizations for a specific pipeline step."""
    step = PIPELINE_STEPS[step_index]
    
    # Get the correct paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(app_dir))
    output_dir = os.path.join(project_root, "data", "output", step["output_dir"])
    
    # Display custom content if available
    if step["custom_text"]:
        st.markdown("### Custom Notes")
        st.markdown(step["custom_text"])
    
    if step["custom_image"]:
        st.markdown("### Custom Image")
        st.image(step["custom_image"], use_column_width=True)
    
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
    
    # Display images
    st.subheader("Visualization Outputs")
    
    # Create tabs for different categories of images
    image_categories = categorize_images(image_files)
    
    if len(image_categories) > 1:
        tabs = st.tabs(list(image_categories.keys()))
        
        for i, (category, files) in enumerate(image_categories.items()):
            with tabs[i]:
                display_image_grid(files)
    else:
        # If only one category, display without tabs
        display_image_grid(image_files)

def categorize_images(image_files):
    """Categorize image files based on filename patterns."""
    categories = {}
    
    # Default category for uncategorized images
    categories["General"] = []
    
    # Define category patterns and their names
    patterns = {
        "forest_plot": "Forest Plots",
        "correlation": "Correlation Plots",
        "trajectory": "Trajectory Plots",
        "cluster": "Clustering Results",
        "survival": "Survival Analysis",
        "confusion": "Model Evaluation",
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

def display_image_grid(image_files):
    """Display images in a responsive grid layout."""
    # Sort files by name for consistent display
    image_files = sorted(image_files)
    
    # Display images in a grid (2 columns)
    for i in range(0, len(image_files), 2):
        cols = st.columns(2)
        
        # First column
        with cols[0]:
            if i < len(image_files):
                img_path = image_files[i]
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
        
        # Second column
        with cols[1]:
            if i + 1 < len(image_files):
                img_path = image_files[i + 1]
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

def process_terminal_queue():
    """Process any pending terminal output from the queue."""
    if "output_queue" in st.session_state and "terminal" in st.session_state:
        # Process all available messages in the queue
        while not st.session_state.output_queue.empty():
            try:
                line = st.session_state.output_queue.get_nowait()
                st.session_state.terminal.add_output(line)
            except:
                break

def auto_refresh(interval_seconds=1):
    """Add JavaScript to auto-refresh the app at specified intervals."""
    # Only add this in "running" state to avoid unnecessary refreshes
    for status in st.session_state.pipeline.steps_status:
        if status == "running":
            st.markdown(
                f"""
                <script>
                    setTimeout(function(){{
                        window.location.reload();
                    }}, {interval_seconds * 1000});
                </script>
                """,
                unsafe_allow_html=True
            )
            break