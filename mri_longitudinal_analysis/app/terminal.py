"""
Terminal output handling for the MRI Longitudinal Analysis Pipeline GUI.

This module provides functionality to capture, store, and display terminal output
from the pipeline scripts.
"""

from pipeline import output_queue, queue_lock

class TerminalOutput:
    """Manages terminal output capture and display."""
    
    def __init__(self, max_lines=1000):
        """Initialize the terminal output buffer."""
        self.output_lines = []
        self.max_lines = max_lines
    
    def add_output(self, line):
        """Add a line to the terminal output buffer."""
        self.output_lines.append(line)
        
        # Trim buffer if it exceeds max_lines
        if len(self.output_lines) > self.max_lines:
            self.output_lines = self.output_lines[-self.max_lines:]
    
    def get_output(self):
        """Get the full terminal output as a string."""
        return "".join(self.output_lines)
    
    def clear(self):
        """Clear the terminal output buffer."""
        self.output_lines = []
    
    def process_queue(self):
        """Process any pending output from the queue."""
        with queue_lock:
            while not output_queue.empty():
                try:
                    line = output_queue.get_nowait()
                    self.add_output(line)
                except:
                    break