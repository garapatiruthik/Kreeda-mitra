"""
Pose Analyzer Module

MediaPipe-based pose detection and analysis for fencing technique evaluation.
Provides real-time pose estimation with skeleton overlay visualization.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional
import streamlit as st


class PoseAnalyzer:
    """
    MediaPipe pose detection and analysis for fencing.
    
    This class handles all MediaPipe pose estimation functionality,
    including frame processing, landmark extraction, and skeleton visualization.
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the pose analyzer with MediaPipe settings.
        
        Args:
            static_image_mode: If True, treats input as static images
            model_complexity: 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks: If True, applies temporal smoothing
            enable_segmentation: If True, enables person segmentation
            min_detection_confidence: Minimum detection confidence (0-1)
            min_tracking_confidence: Minimum tracking confidence (0-1)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Cache for landmark connections
        self._connections = self.mp_pose.POSE_CONNECTIONS
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List, Dict[str, float]]:
        """
        Process a single video frame and extract pose data.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Tuple of:
                - Annotated frame with skeleton overlay
                - List of landmarks
                - Dictionary of calculated angles
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        landmarks = []
        angles = {}
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Draw skeleton on frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self._connections,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate angles using imported AngleCalculator
            from src.angle_calculator import AngleCalculator
            angles = AngleCalculator.calculate_fencing_angles(landmarks)
        
        return frame, landmarks, angles
    
    def get_skeleton_overlay(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Get frame with skeleton overlay in specified color.
        
        Args:
            frame: Input BGR frame
            color: RGB color tuple for skeleton (default: green)
            
        Returns:
            Frame with skeleton overlay
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks
            for landmark in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Draw connections
            for connection in self._connections:
                start = results.pose_landmarks.landmark[connection[0]]
                end = results.pose_landmarks.landmark[connection[1]]
                start_coords = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_coords = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_coords, end_coords, color, 2)
        
        return frame
    
    def get_landmark_coordinates(
        self,
        landmarks: List,
        landmark_index: int,
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates for a specific landmark.
        
        Args:
            landmarks: List of MediaPipe landmarks
            landmark_index: Index of the landmark
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Tuple of (x, y) pixel coordinates, or None if not found
        """
        if not landmarks or landmark_index >= len(landmarks):
            return None
        
        h, w, _ = frame_shape
        landmark = landmarks[landmark_index]
        
        return (int(landmark.x * w), int(landmark.y * h))
    
    def draw_angle_annotations(
        self,
        frame: np.ndarray,
        landmarks: List,
        angles: Dict[str, float]
    ) -> np.ndarray:
        """
        Draw angle values on the frame at landmark positions.
        
        Args:
            frame: Input BGR frame
            landmarks: List of pose landmarks
            angles: Dictionary of calculated angles
            
        Returns:
            Frame with angle annotations
        """
        h, w, _ = frame.shape
        
        # Key joints to display with labels
        key_joints = {
            'right_elbow': (14, "R Elbow"),
            'left_elbow': (13, "L Elbow"),
            'right_knee': (26, "R Knee"),
            'left_knee': (25, "L Knee"),
            'right_hip': (24, "R Hip"),
            'left_hip': (23, "L Hip"),
            'right_shoulder': (12, "R Shoulder"),
            'left_shoulder': (11, "L Shoulder"),
        }
        
        for joint_name, (landmark_idx, label) in key_joints.items():
            if joint_name in angles and landmark_idx < len(landmarks):
                lm = landmarks[landmark_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                
                # Draw angle value
                cv2.putText(
                    frame,
                    f"{label}: {angles[joint_name]:.1f}°",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                # Thinner text for better visibility
                cv2.putText(
                    frame,
                    f"{label}: {angles[joint_name]:.1f}°",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return frame
    
    def detect_pose_quality(self, landmarks: List) -> Dict[str, float]:
        """
        Assess the quality of detected pose.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary with
        """
        quality = {
            'visibility_score': 0.0,
            'stability_score': 0.0,
            'completeness': 0.0
        }
        
        if not landmarks:
            return quality
        
        # Calculate visibility (average visibility of all landmarks)
        visibilities = [lm.visibility for lm in landmarks if hasattr(lm, 'visibility')]
        if visibilities:
            quality['visibility_score'] = float(np.mean(visibilities))
        
        # Calculate completeness (percentage of landmarks detected)
        quality['completeness'] = float(len(landmarks)) / 33.0  # MediaPipe has 33 landmarks
        
        return quality
    
    def close(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()


def create_pose_analyzer(
    model_complexity: int = 1,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5
) -> PoseAnalyzer:
    """
    Factory function to create a configured PoseAnalyzer.
    
    Args:
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        detection_confidence: Minimum detection confidence
        tracking_confidence: Minimum tracking confidence
        
    Returns:
        Configured PoseAnalyzer instance
    """
    return PoseAnalyzer(
        model_complexity=model_complexity,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    )
