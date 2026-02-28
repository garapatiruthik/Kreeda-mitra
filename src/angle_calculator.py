"""
Angle Calculator Module

Provides utilities for calculating angles between body landmarks using vector math.
Used for analyzing fencing techniques by computing joint angles from pose data.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import streamlit as st


class AngleCalculator:
    """
    Calculate angles between three body landmarks using vector math.
    
    This class provides static methods for computing angles between body joints
    using the law of cosines formula. It is designed specifically for analyzing
    fencing movements and techniques.
    """
    
    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Calculate angle at point b given points a, b, c.
        
        Uses the law of cosines formula:
            cos(angle) = (ba · bc) / (|ba| * |bc|)
        
        Args:
            a: First point coordinates (x, y)
            b: Vertex point coordinates (x, y) - the angle to calculate
            c: Third point coordinates (x, y)
            
        Returns:
            Angle in degrees (0-180)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Create vectors from vertex
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle using dot product
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        # Avoid division by zero
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # Clamp to avoid numerical errors with arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return float(angle)
    
    @staticmethod
    def calculate_fencing_angles(landmarks: List) -> Dict[str, float]:
        """
        Calculate all key fencing angles from MediaPipe landmarks.
        
        MediaPipe pose landmark indices:
        - 11: Left shoulder
        - 12: Right shoulder
        - 13: Left elbow
        - 14: Right elbow
        - 15: Left wrist
        - 16: Right wrist
        - 23: Left hip
        - 24: Right hip
        - 25: Left knee
        - 26: Right knee
        - 27: Left ankle
        - 28: Right ankle
        
        Args:
            landmarks: List of MediaPipe pose landmarks
            
        Returns:
            Dictionary of angle names and values in degrees
        """
        angles: Dict[str, float] = {}
        
        # Check if landmarks are valid
        if not landmarks or len(landmarks) < 29:
            return angles
        
        try:
            # Get landmark coordinates for left side
            left_shoulder = (landmarks[11].x, landmarks[11].y)
            left_elbow = (landmarks[13].x, landmarks[13].y)
            left_wrist = (landmarks[15].x, landmarks[15].y)
            left_hip = (landmarks[23].x, landmarks[23].y)
            left_knee = (landmarks[25].x, landmarks[25].y)
            left_ankle = (landmarks[27].x, landmarks[27].y)
            
            # Get landmark coordinates for right side
            right_shoulder = (landmarks[12].x, landmarks[12].y)
            right_elbow = (landmarks[14].x, landmarks[14].y)
            right_wrist = (landmarks[16].x, landmarks[16].y)
            right_hip = (landmarks[24].x, landmarks[24].y)
            right_knee = (landmarks[26].x, landmarks[26].y)
            right_ankle = (landmarks[28].x, landmarks[28].y)
            
            # Calculate arm angles (weapon arm - typically right)
            angles['right_elbow'] = AngleCalculator.calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            angles['left_elbow'] = AngleCalculator.calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            
            # Calculate leg angles (lunge analysis)
            angles['right_knee'] = AngleCalculator.calculate_angle(
                right_hip, right_knee, right_ankle
            )
            angles['left_knee'] = AngleCalculator.calculate_angle(
                left_hip, left_knee, left_ankle
            )
            
            # Calculate hip angles
            angles['right_hip'] = AngleCalculator.calculate_angle(
                right_shoulder, right_hip, right_knee
            )
            angles['left_hip'] = AngleCalculator.calculate_angle(
                left_shoulder, left_hip, left_knee
            )
            
            # Calculate shoulder angles
            angles['right_shoulder'] = AngleCalculator.calculate_angle(
                right_elbow, right_shoulder, right_hip
            )
            angles['left_shoulder'] = AngleCalculator.calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            
            # Calculate torso angle (for lunge depth)
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            angles['torso'] = AngleCalculator.calculate_angle(
                mid_shoulder, mid_hip,
                (mid_hip[0], mid_hip[1] + 0.1)  # Point downward
            )
            
        except Exception as e:
            st.error(f"Error calculating angles: {e}")
        
        return angles
    
    @staticmethod
    def get_angle_statistics(angle_series: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for a series of angle measurements.
        
        Args:
            angle_series: List of angle dictionaries from multiple frames
            
        Returns:
            Dictionary with statistics for each angle type
        """
        if not angle_series:
            return {}
        
        # Get all angle names from first frame
        angle_names = angle_series[0].keys()
        stats: Dict[str, Dict[str, float]] = {}
        
        for angle_name in angle_names:
            values = [frame_angles.get(angle_name, 0) for frame_angles in angle_series if frame_angles]
            if values:
                stats[angle_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
        
        return stats
    
    @staticmethod
    def validate_fencing_form(angles: Dict[str, float]) -> Dict[str, Tuple[bool, str]]:
        """
        Validate fencing form based on calculated angles.
        
        Args:
            angles: Dictionary of calculated angles
            
        Returns:
            Dictionary with validation results for each joint
        """
        validation_results: Dict[str, Tuple[bool, str]] = {}
        
        # Key fencing angle thresholds
        thresholds = {
            'right_knee': (70, 110, 'Knee angle should be 70-110° for safe lunge'),
            'left_knee': (70, 110, 'Knee angle should be 70-110° for safe lunge'),
            'right_elbow': (160, 180, 'Weapon arm should be nearly straight (160-180°)'),
            'left_elbow': (160, 180, 'Non-weapon arm should be 160-180°'),
            'right_shoulder': (150, 180, 'Shoulder should be extended 150-180°'),
            'left_shoulder': (80, 110, 'Non-weapon shoulder should be 80-110° for balance'),
        }
        
        for joint, (min_angle, max_angle, message) in thresholds.items():
            if joint in angles:
                angle = angles[joint]
                is_valid = min_angle <= angle <= max_angle
                status = "✓ Good" if is_valid else "✗ Needs work"
                validation_results[joint] = (is_valid, f"{status}: {angle:.1f}° - {message}")
        
        return validation_results


def calculate_euclidean_distance(series1: List[float], series2: List[float]) -> float:
    """
    Calculate average Euclidean distance between two angle series.
    
    Args:
        series1: First angle series
        series2: Second angle series
        
    Returns:
        Average Euclidean distance
    """
    if not series1 or not series2:
        return float('inf')
    
    # Pad shorter series if needed
    max_len = max(len(series1), len(series2))
    series1 = series1 + [0] * (max_len - len(series1))
    series2 = series2 + [0] * (max_len - len(series2))
    
    return float(np.mean(np.sqrt(np.array(series1)**2 + np.array(series2)**2)))


def normalize_angle_series(series: List[float], target_length: int) -> List[float]:
    """
    Normalize an angle series to a target length using interpolation.
    
    Args:
        series: Original angle series
        target_length: Desired length
        
    Returns:
        Interpolated series of target length
    """
    if not series:
        return [0.0] * target_length
    
    if len(series) == target_length:
        return series
    
    # Use numpy interpolation
    original_indices = np.linspace(0, len(series) - 1, len(series))
    target_indices = np.linspace(0, len(series) - 1, target_length)
    
    return np.interp(target_indices, original_indices, series).tolist()
