"""
Face display component for GUI.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging


class FaceDisplay:
    """Component for displaying face detection results."""
    
    def __init__(self, parent, width=300, height=300):
        """
        Initialize face display component.
        
        Args:
            parent: Parent widget
            width: Display width
            height: Display height
        """
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg="white")
        self.canvas.pack()
        
        # Current image
        self.current_image = None
        self.photo = None
        
    def display_image(self, image: np.ndarray):
        """
        Display image on canvas.
        
        Args:
            image: Image to display
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Resize image to fit canvas
            scale = min(self.width / image.shape[1], self.height / image.shape[0])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            
            # Resize image
            resized_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized_image)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            self.current_image = image
            
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
    
    def draw_face_boxes(self, faces: list, color=(0, 255, 0), thickness=2):
        """
        Draw face bounding boxes on the current image.
        
        Args:
            faces: List of face bounding boxes
            color: Box color (BGR)
            thickness: Box thickness
        """
        try:
            if self.current_image is None:
                return
            
            # Create a copy of the current image
            image_with_boxes = self.current_image.copy()
            
            # Draw face boxes
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(image_with_boxes, f"Face {i+1}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the image with boxes
            self.display_image(image_with_boxes)
            
        except Exception as e:
            self.logger.error(f"Error drawing face boxes: {e}")
    
    def draw_face_labels(self, faces: list, labels: list, color=(0, 255, 0)):
        """
        Draw face labels on the current image.
        
        Args:
            faces: List of face bounding boxes
            labels: List of face labels
            color: Label color (BGR)
        """
        try:
            if self.current_image is None:
                return
            
            # Create a copy of the current image
            image_with_labels = self.current_image.copy()
            
            # Draw face boxes and labels
            for i, ((x, y, w, h), label) in enumerate(zip(faces, labels)):
                cv2.rectangle(image_with_labels, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image_with_labels, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the image with labels
            self.display_image(image_with_labels)
            
        except Exception as e:
            self.logger.error(f"Error drawing face labels: {e}")
    
    def clear(self):
        """Clear the display."""
        self.canvas.delete("all")
        self.current_image = None
        self.photo = None
    
    def get_widget(self):
        """Get the main widget."""
        return self.frame
