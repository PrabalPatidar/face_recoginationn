"""
Scan controls component for GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging


class ScanControls:
    """Component for scan control buttons and options."""
    
    def __init__(self, parent, scan_callback=None, clear_callback=None):
        """
        Initialize scan controls component.
        
        Args:
            parent: Parent widget
            scan_callback: Callback function for scan button
            clear_callback: Callback function for clear button
        """
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        self.scan_callback = scan_callback
        self.clear_callback = clear_callback
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create control widgets."""
        # File selection
        ttk.Label(self.frame, text="Select Image:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        file_frame = ttk.Frame(self.frame)
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=1)
        
        # Detection method
        ttk.Label(self.frame, text="Detection Method:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        self.detection_method = tk.StringVar(value="haar_cascade")
        method_frame = ttk.Frame(self.frame)
        method_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(method_frame, text="Haar Cascade", 
                       variable=self.detection_method, value="haar_cascade").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(method_frame, text="HOG", 
                       variable=self.detection_method, value="hog").grid(row=1, column=0, sticky=tk.W)
        
        # Scan type
        ttk.Label(self.frame, text="Scan Type:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.scan_type = tk.StringVar(value="detect")
        scan_frame = ttk.Frame(self.frame)
        scan_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(scan_frame, text="Face Detection", 
                       variable=self.scan_type, value="detect").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(scan_frame, text="Face Recognition", 
                       variable=self.scan_type, value="recognize").grid(row=1, column=0, sticky=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        scan_btn = ttk.Button(button_frame, text="Scan Image", command=self.scan_image)
        scan_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        clear_btn = ttk.Button(button_frame, text="Clear Results", command=self.clear_results)
        clear_btn.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.frame, textvariable=self.status_var)
        status_label.grid(row=7, column=0, sticky=tk.W, pady=(10, 0))
        
    def browse_file(self):
        """Browse for image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.set_status(f"Selected: {file_path}")
    
    def scan_image(self):
        """Trigger scan image action."""
        if not self.file_path_var.get():
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.scan_callback:
            self.scan_callback()
    
    def clear_results(self):
        """Trigger clear results action."""
        if self.clear_callback:
            self.clear_callback()
    
    def set_status(self, message: str):
        """Set status message."""
        self.status_var.set(message)
        self.logger.info(f"Status: {message}")
    
    def get_selected_file(self) -> str:
        """Get selected file path."""
        return self.file_path_var.get()
    
    def get_detection_method(self) -> str:
        """Get selected detection method."""
        return self.detection_method.get()
    
    def get_scan_type(self) -> str:
        """Get selected scan type."""
        return self.scan_type.get()
    
    def set_scan_callback(self, callback):
        """Set scan callback function."""
        self.scan_callback = callback
    
    def set_clear_callback(self, callback):
        """Set clear callback function."""
        self.clear_callback = callback
    
    def get_widget(self):
        """Get the main widget."""
        return self.frame
