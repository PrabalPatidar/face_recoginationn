"""
Base model class for face detection and recognition models.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
import numpy as np


class BaseModel(ABC):
    """Base class for all face detection and recognition models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load the model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Any:
        """
        Make prediction on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'version': '1.0.0'
        }
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if input_data is None:
            self.logger.error("Input data is None")
            return False
        
        if not isinstance(input_data, np.ndarray):
            self.logger.error("Input data must be a numpy array")
            return False
        
        if input_data.size == 0:
            self.logger.error("Input data is empty")
            return False
        
        return True
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed input data
        """
        # Default preprocessing - normalize to [0, 1]
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        if input_data.max() > 1.0:
            input_data = input_data / 255.0
        
        return input_data
    
    def postprocess_output(self, output_data: Any) -> Any:
        """
        Postprocess output data.
        
        Args:
            output_data: Raw output data
            
        Returns:
            Postprocessed output data
        """
        return output_data
