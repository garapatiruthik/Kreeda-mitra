"""
Video Processor Module

Provides utilities for processing video files for pose analysis.
Handles video loading, frame extraction, and batch processing.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import tempfile
import os
import streamlit as st
from PIL import Image


def process_video_stream(
    video_path: str,
    analyzer,
    max_frames: int = 300,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[np.ndarray], List[Dict[str, float]], int]:
    """
    Process video file and return frames with pose overlay.
    
    Args:
        video_path: Path to video file
        analyzer: PoseAnalyzer instance
        max_frames: Maximum number of frames to process
        progress_callback: Optional callback for progress updates
            
    Returns:
        Tuple of:
            - List of annotated frames (RGB)
            - List of angle dictionaries per frame
            - Video FPS
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    all_angles = []
    frame_count = 0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Determine actual frames to process
    frames_to_process = min(total_frames, max_frames)
    
    while cap.is_open() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        
        # Process frame with pose analyzer
        annotated_frame, landmarks, angles = analyzer.process_frame(frame)
        
        # Draw angle overlays if landmarks detected
        if landmarks:
            annotated_frame = analyzer.draw_angle_annotations(
                annotated_frame, landmarks, angles
            )
            all_angles.append(angles)
        
        # Convert to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(annotated_frame)
        
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress = min(frame_count / frames_to_process, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{frames_to_process}")
            
            if progress_callback:
                progress_callback(frame_count, frames_to_process)
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return frames, all_angles, fps


def extract_frames(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    target_size: Tuple[int, int] = (640, 480)
) -> List[np.ndarray]:
    """
    Extract specific frames from a video.
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to extract (None = all frames)
        target_size: Target (width, height) for resizing
            
    Returns:
        List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_idx = 0
    
    while cap.is_open():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be extracted
        if frame_indices is None or frame_idx in frame_indices:
            # Resize and convert to RGB
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if frame_indices and len(frames) >= len(frame_indices):
                break
        
        frame_idx += 1
    
    cap.release()
    return frames


def save_annotated_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
) -> bool:
    """
    Save annotated frames as a video file.
    
    Args:
        frames: List of RGB frames
        output_path: Output video file path
        fps: Frames per second for output video
            
    Returns:
        True if successful
    """
    if not frames:
        return False
    
    # Get frame size
    h, w = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True


def get_video_info(video_path: str) -> Dict[str, any]:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
            
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    Validate that a video file can be processed.
    
    Args:
        video_path: Path to video file
            
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file format. Supported: {', '.join(valid_extensions)}"
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, "Could not open video file"
    
    # Check frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if frame_count <= 0:
        return False, "Video appears to be empty"
    
    return True, ""


def create_video_thumbnail(
    video_path: str,
    output_path: Optional[str] = None,
    frame_position: float = 0.1
) -> Optional[str]:
    """
    Create a thumbnail image from a video.
    
    Args:
        video_path: Path to video file
        output_path: Output thumbnail path (default: same dir as video)
        frame_position: Position of frame to capture (0.0-1.0)
            
    Returns:
        Path to thumbnail image, or None on failure
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Get frame to capture
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(total_frames * frame_position)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Resize
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Determine output path
    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_name}_thumb.jpg")
    
    # Save thumbnail
    img = Image.fromarray(frame)
    img.save(output_path)
    
    return output_path


def process_video_from_upload(
    uploaded_file,
    analyzer,
    max_frames: int = 300
) -> Tuple[List[np.ndarray], List[Dict[str, float]], int]:
    """
    Process an uploaded video file from Streamlit.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        analyzer: PoseAnalyzer instance
        max_frames: Maximum frames to process
            
    Returns:
        Tuple of (frames, angles, fps)
    """
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    
    try:
        # Process video
        frames, angles, fps = process_video_stream(
            temp_path, analyzer, max_frames
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return frames, angles, fps
