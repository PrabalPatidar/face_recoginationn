"""
Main GUI window for the Face Scan application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from face_scan.core.face_detector import FaceDetector
from face_scan.core.face_recognizer import FaceRecognizer
from face_scan.core.face_encoder import FaceEncoder


class MainWindow:
    """Main application window."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Scan Project")
        self.root.geometry("1200x800")
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.face_encoder = FaceEncoder()
        
        # Current image
        self.current_image = None
        self.current_image_path = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Scan Project", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File selection
        ttk.Label(control_frame, text="Select Image:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=1)
        
        # Detection options
        ttk.Label(control_frame, text="Detection Method:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        self.detection_method = tk.StringVar(value="haar_cascade")
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(method_frame, text="Haar Cascade", 
                       variable=self.detection_method, value="haar_cascade").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(method_frame, text="HOG", 
                       variable=self.detection_method, value="hog").grid(row=1, column=0, sticky=tk.W)
        
        # Scan options
        ttk.Label(control_frame, text="Scan Type:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.scan_type = tk.StringVar(value="detect")
        scan_frame = ttk.Frame(control_frame)
        scan_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(scan_frame, text="Face Detection", 
                       variable=self.scan_type, value="detect").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(scan_frame, text="Face Recognition", 
                       variable=self.scan_type, value="recognize").grid(row=1, column=0, sticky=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        detect_btn = ttk.Button(button_frame, text="Detect Faces", command=self.detect_faces)
        detect_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        recognize_btn = ttk.Button(button_frame, text="Recognize Faces", command=self.recognize_faces)
        recognize_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        clear_btn = ttk.Button(button_frame, text="Clear Results", command=self.clear_results)
        clear_btn.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Right panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Image canvas
        self.image_canvas = tk.Canvas(image_frame, bg="white", width=600, height=400)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient="vertical", command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(image_frame, orient="horizontal", command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Results panel
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        
        self.results_text = tk.Text(results_frame, height=6, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
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
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load image from file."""
        try:
            self.current_image = cv2.imread(file_path)
            self.current_image_path = file_path
            
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            self.display_image(self.current_image)
            self.log(f"Loaded image: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
            self.logger.error(f"Error loading image: {e}")
    
    def display_image(self, image):
        """Display image on canvas."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized, use default
                canvas_width, canvas_height = 600, 400
            
            # Calculate scaling factor
            scale = min(canvas_width / image.shape[1], canvas_height / image.shape[0])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            
            # Resize image
            resized_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized_image)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image: {e}")
            self.logger.error(f"Error displaying image: {e}")
    
    def detect_faces(self):
        """Detect faces in the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            # Update detector method
            self.face_detector = FaceDetector(self.detection_method.get())
            
            # Detect faces
            faces = self.face_detector.detect_faces(self.current_image)
            
            # Draw face boxes
            result_image = self.face_detector.draw_faces(self.current_image, faces)
            
            # Display result
            self.display_image(result_image)
            
            # Log results
            self.log(f"Face Detection Results:")
            self.log(f"  Method: {self.detection_method.get()}")
            self.log(f"  Faces detected: {len(faces)}")
            
            for i, (x, y, w, h) in enumerate(faces):
                self.log(f"  Face {i+1}: ({x}, {y}, {w}, {h})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting faces: {e}")
            self.logger.error(f"Error detecting faces: {e}")
    
    def recognize_faces(self):
        """Recognize faces in the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            # Detect faces first
            faces = self.face_detector.detect_faces(self.current_image)
            
            if not faces:
                self.log("No faces detected for recognition")
                return
            
            # Get face encodings
            encodings = self.face_encoder.encode_faces(self.current_image, faces)
            
            # Recognize faces
            recognition_results = self.face_recognizer.recognize_faces(encodings)
            
            # Draw results
            result_image = self.current_image.copy()
            for i, ((x, y, w, h), (name, confidence)) in enumerate(zip(faces, recognition_results)):
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                label = f"{name} ({confidence:.2f})"
                cv2.putText(result_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display result
            self.display_image(result_image)
            
            # Log results
            self.log(f"Face Recognition Results:")
            self.log(f"  Faces recognized: {len(recognition_results)}")
            
            for i, (name, confidence) in enumerate(recognition_results):
                self.log(f"  Face {i+1}: {name} (confidence: {confidence:.3f})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error recognizing faces: {e}")
            self.logger.error(f"Error recognizing faces: {e}")
    
    def clear_results(self):
        """Clear results and reset image."""
        if self.current_image is not None:
            self.display_image(self.current_image)
        
        self.results_text.delete(1.0, tk.END)
        self.log("Results cleared")
    
    def log(self, message):
        """Add message to results log."""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
