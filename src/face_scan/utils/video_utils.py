"""
Video utility functions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
import logging
from pathlib import Path


def extract_frames_from_video(video_path: str, frame_interval: int = 1) -> Generator[np.ndarray, None, None]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract every nth frame (1 = all frames)
        
    Yields:
        Video frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                yield frame
            
            frame_count += 1
            
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video file information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {'error': f"Could not open video file: {video_path}"}
    
    try:
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'format': cap.get(cv2.CAP_PROP_FORMAT)
        }
        
        return info
        
    finally:
        cap.release()


def create_video_from_frames(frames: List[np.ndarray], output_path: str, 
                           fps: float = 30.0, codec: str = 'mp4v') -> bool:
    """
    Create a video from a list of frames.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec
        
    Returns:
        True if video created successfully, False otherwise
    """
    if not frames:
        return False
    
    try:
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        # Release video writer
        out.release()
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return False


def process_video_for_face_detection(video_path: str, output_path: str, 
                                   face_detector, frame_interval: int = 30) -> bool:
    """
    Process video for face detection and create output video with face boxes.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        face_detector: Face detector instance
        frame_interval: Process every nth frame
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame if it's the right interval
            if frame_count % frame_interval == 0:
                # Detect faces
                faces = face_detector.detect_faces(frame)
                
                # Draw face boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return False


def extract_faces_from_video(video_path: str, face_detector, 
                           output_dir: str, frame_interval: int = 30) -> List[str]:
    """
    Extract face images from a video file.
    
    Args:
        video_path: Input video path
        face_detector: Face detector instance
        output_dir: Output directory for face images
        frame_interval: Extract faces from every nth frame
        
    Returns:
        List of extracted face image paths
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted_faces = []
        frame_count = 0
        face_count = 0
        
        for frame in extract_frames_from_video(video_path, frame_interval):
            # Detect faces in frame
            faces = face_detector.detect_faces(frame)
            
            # Extract each face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Save face image
                face_filename = f"face_{frame_count}_{i}.jpg"
                face_path = output_path / face_filename
                
                cv2.imwrite(str(face_path), face_region)
                extracted_faces.append(str(face_path))
                face_count += 1
            
            frame_count += 1
        
        logging.info(f"Extracted {face_count} faces from {frame_count} frames")
        return extracted_faces
        
    except Exception as e:
        logging.error(f"Error extracting faces from video: {e}")
        return []


def create_face_timeline(video_path: str, face_detector, 
                        output_path: str, frame_interval: int = 30) -> dict:
    """
    Create a timeline of face detections in a video.
    
    Args:
        video_path: Input video path
        face_detector: Face detector instance
        output_path: Output JSON file path
        frame_interval: Analyze every nth frame
        
    Returns:
        Dictionary containing face timeline data
    """
    try:
        timeline = {
            'video_path': video_path,
            'frame_interval': frame_interval,
            'detections': []
        }
        
        frame_count = 0
        time_per_frame = 1.0 / 30.0  # Assume 30 FPS
        
        for frame in extract_frames_from_video(video_path, frame_interval):
            # Detect faces in frame
            faces = face_detector.detect_faces(frame)
            
            if faces:
                timestamp = frame_count * frame_interval * time_per_frame
                
                detection = {
                    'timestamp': timestamp,
                    'frame_number': frame_count * frame_interval,
                    'faces_count': len(faces),
                    'faces': []
                }
                
                for (x, y, w, h) in faces:
                    detection['faces'].append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    })
                
                timeline['detections'].append(detection)
            
            frame_count += 1
        
        # Save timeline to file
        import json
        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        return timeline
        
    except Exception as e:
        logging.error(f"Error creating face timeline: {e}")
        return {}


def get_video_thumbnail(video_path: str, output_path: str, 
                       frame_number: int = 0) -> bool:
    """
    Extract a thumbnail from a video.
    
    Args:
        video_path: Input video path
        output_path: Output thumbnail path
        frame_number: Frame number to extract (0 = first frame)
        
    Returns:
        True if thumbnail extracted successfully, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Save thumbnail
            cv2.imwrite(output_path, frame)
            cap.release()
            return True
        else:
            cap.release()
            return False
            
    except Exception as e:
        logging.error(f"Error extracting thumbnail: {e}")
        return False


def convert_video_format(input_path: str, output_path: str, 
                        codec: str = 'mp4v', fps: float = None) -> bool:
    """
    Convert video to different format.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        codec: Output codec
        fps: Output FPS (None = keep original)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = fps if fps is not None else original_fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Copy frames
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True
        
    except Exception as e:
        logging.error(f"Error converting video: {e}")
        return False
