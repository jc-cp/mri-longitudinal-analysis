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

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

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
        
        # Add edit mode instructions when edit mode is enabled
        if st.session_state.edit_mode:
            st.markdown("---")
            st.markdown("""
            ### 📝 Edit Mode Instructions
            
            Edit Mode allows you to customize the content for each pipeline step:
            
            1. **Select a step** from the sidebar
            2. Add **custom notes** in the text area
            3. Upload **custom images** if needed
            4. Click **Save Custom Content** to store your changes
            
            Your customizations will be displayed alongside the step's default content and outputs.
            """)
        
    else:
        # Display current step information
        step_info = PIPELINE_STEPS[current_step]
        
        st.header(step_info["name"])
        
        # Display step description
        st.markdown(step_info["description"])
        
        # Custom content editor (only shown in edit mode)
        if st.session_state.edit_mode:
            st.subheader("Customize Step Content")
            st.info("""
            Add your custom notes and/or upload an image for this step. 
            Click 'Save Custom Content' when you're done to save your changes.
            """)
            
            custom_text = st.text_area(
                "Custom Notes", 
                value=step_info["custom_text"],
                height=150,
                key=f"custom_text_{current_step}"
            )
            
            custom_image = st.file_uploader(
                "Upload Custom Image", 
                type=["png", "jpg", "jpeg"],
                key=f"custom_image_{current_step}"
            )
            
            if st.button("Save Custom Content", key=f"save_custom_{current_step}"):
                st.session_state.pipeline.update_custom_content(
                    current_step, 
                    custom_text=custom_text,
                    custom_image=custom_image
                )
                st.success("Custom content saved!")
                st.rerun()
        
        # Display output visualizations if available
        if st.session_state.pipeline.get_step_status(current_step) in ["completed", "running"]:
            utils.display_step_output(current_step)
        
        # Add execution button at the bottom
        st.markdown("---")
        if st.session_state.pipeline.get_step_status(current_step) == "pending":
            if st.button("▶️ Execute Step", key=f"execute_{current_step}", use_container_width=True):
                st.session_state.pipeline.run_step(current_step)
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
    
    # Toggle edit mode with explanation
    edit_mode = st.toggle(
        "Edit Mode", 
        value=st.session_state.edit_mode,
        help="Enable to add custom notes and images to each step"
    )
    if edit_mode != st.session_state.edit_mode:
        st.session_state.edit_mode = edit_mode
        st.rerun()
    
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