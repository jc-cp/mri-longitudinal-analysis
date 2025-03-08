"""
Streamlit GUI for MRI Longitudinal Analysis Pipeline.

This application provides a visual interface to run and monitor the MRI longitudinal analysis
pipeline, displaying progress, terminal output, and visualization results.
"""

import os
import streamlit as st
import time
from pipeline import Pipeline, PIPELINE_STEPS
from terminal import TerminalOutput
import utils

# Page configuration
st.set_page_config(
    page_title="MRI Longitudinal Analysis Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for pipeline tracking
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline()

if "terminal" not in st.session_state:
    st.session_state.terminal = TerminalOutput()

# Remove edit mode from session state
if "edit_mode" in st.session_state:
    del st.session_state.edit_mode

# Process any pending terminal output
st.session_state.terminal.process_queue()

# Title
st.title("MRI Longitudinal Analysis Pipeline 🧠")

# Main layout with two columns at the top level - aligned at the same height
cols = st.columns([2, 1])

# Main content column
with cols[0]:
    # Introduction text
    st.markdown("""
    This application guides you through the MRI longitudinal analysis pipeline for pediatric low-grade gliomas (pLGGs).  
    Follow the steps sequentially to process and analyze your data.
    """)
    
    # Display landing page image when no step is selected
    current_step = st.session_state.pipeline.current_step
    if current_step is None:
        # Path to the landing page image
        app_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(app_dir, "images")
        
        # Create images directory if it doesn't exist
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            st.warning("Created images directory. Please add your landing page image to app/images/pipeline_overview.png")
        
        st.info("Select a pipeline step from the sidebar to begin.")

        # Try to display the pipeline overview image
        pipeline_image = os.path.join(image_dir, "pipeline_overview.png")
        if os.path.exists(pipeline_image):
            st.image(pipeline_image, use_container_width=True, caption="Pipeline Overview")
        else:
            st.info("No pipeline overview image found. Add an image named 'pipeline_overview.png' to the app/images/ folder.")
    else:
        # Display current step information
        step_info = PIPELINE_STEPS[current_step]
        
        st.header(step_info["name"])
        
        # Display step description
        st.markdown(step_info["description"])
        
        # Display step illustration image if available
        if "illustration" in step_info and os.path.exists(step_info["illustration"]):
            illustration_width = step_info.get("illustration_width", 500)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:  # Use the middle column to center the image
                st.image(step_info["illustration"], width=illustration_width)
        
        # Display output visualizations if available
        utils.display_step_output(current_step)
        
        # Add execution button at the bottom
        st.markdown("---")
        if st.session_state.pipeline.get_step_status(current_step) == "pending":
            cols = st.columns(2)
            with cols[0]:
                if st.button("▶️ Execute Step", key=f"execute_{current_step}", use_container_width=True):
                    st.session_state.pipeline.run_step(current_step)
            
            # Add "Omit Step" button for preprocessing and segmentation steps
            with cols[1]:
                if current_step in [0, 1]:  # Image Preprocessing and Tumor Segmentation steps
                    if st.button("⏭️ Omit Step", key=f"omit_{current_step}", use_container_width=True):
                        st.session_state.pipeline.omit_step(current_step)
        elif st.session_state.pipeline.get_step_status(current_step) == "running":
            st.info("⏳ Step is currently running...")
        elif st.session_state.pipeline.get_step_status(current_step) == "completed":
            st.success("✅ Step completed successfully!")

# Terminal output column
with cols[1]:
    # Terminal output display
    st.header("Terminal Output")
    terminal_container = st.container(height=500, border=True)
    
    with terminal_container:
        # Update terminal output
        terminal_output = st.session_state.terminal.get_output()
        if terminal_output:
            st.code(terminal_output, language="bash")
        else:
            st.write("No output yet. Execute a step to see terminal output.")
    
    # Add clear button for terminal
    if st.button("Clear Terminal", use_container_width=True):
        st.session_state.terminal.clear()
        st.rerun()

# Sidebar for pipeline navigation and control
with st.sidebar:
    # Return to Home button at the top
    if st.button("🏠 Return to Home", use_container_width=True):
        st.session_state.pipeline.current_step = None
        st.rerun()
    
    st.markdown("---")
    st.header("Pipeline Steps")
    
    # Display pipeline progress
    progress = st.progress(st.session_state.pipeline.progress_percentage / 100)
    
    # Create buttons for each pipeline step
    for i, step in enumerate(PIPELINE_STEPS):
        step_status = st.session_state.pipeline.get_step_status(i)
        button_label = f"{step['name']} {'✅' if step_status == 'completed' else '⏳' if step_status == 'running' else ''}"
        
        # Disable buttons for steps that can't be run yet
        button_disabled = not st.session_state.pipeline.can_run_step(i)
        
        if st.button(button_label, disabled=button_disabled, key=f"btn_{i}"):
            st.session_state.pipeline.current_step = i
            st.rerun()

# Auto-refresh for terminal updates
if any(status == "running" for status in st.session_state.pipeline.steps_status):
    time.sleep(1)  # Small delay to avoid too frequent refreshes
    st.rerun()