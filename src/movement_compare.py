"""
Movement Comparison Module

Provides DTW (Dynamic Time Warping) based movement comparison
for analyzing fencing techniques between coach and student.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


def simple_dtw(series1: List[float], series2: List[float]) -> float:
    """
    Simplified Dynamic Time Warping for movement comparison.
    
    DTW finds the optimal alignment between two time series by minimizing
    the distance between them, allowing for temporal variations in movement speed.
    
    Args:
        series1: First angle/time series
        series2: Second angle/time series
        
    Returns:
        Normalized DTW distance (lower = more similar)
    """
    n, m = len(series1), len(series2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Create cost matrix with infinity
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, :] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost is the absolute difference
            cost = abs(series1[i-1] - series2[j-1])
            
            # Minimum cost path
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    # Return normalized distance
    return float(dtw_matrix[n, m] / max(n, m))


def fast_dtw(series1: List[float], series2: List[float], radius: int = 1) -> float:
    """
    Fast DTW implementation using constrained window.
    
    This is a faster approximation of DTW that restricts the search window,
    significantly reducing computation time for long sequences.
    
    Args:
        series1: First angle/time series
        series2: Second angle/time series
        radius: Size of the Sakoe-Chiba band (default: 1)
        
    Returns:
        Normalized DTW distance
    """
    n, m = len(series1), len(series2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Initialize matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill with constrained window
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            cost = abs(series1[i-1] - series2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    return float(dtw_matrix[n, m] / max(n, m))


def compare_movements(
    coach_angles: Dict[str, float],
    student_angles: Dict[str, float],
    threshold: float = 15.0
) -> Dict[str, Dict[str, any]]:
    """
    Compare student angles to coach angles.
    
    Args:
        coach_angles: Dictionary of coach's joint angles
        student_angles: Dictionary of student's joint angles
        threshold: Maximum difference in degrees for a "match"
        
    Returns:
        Dictionary with comparison results for each joint
    """
    comparison = {}
    
    # Key joints to compare
    key_joints = [
        'right_elbow', 'left_elbow',
        'right_knee', 'left_knee',
        'right_hip', 'left_hip',
        'right_shoulder', 'left_shoulder'
    ]
    
    for joint in key_joints:
        if joint in coach_angles and joint in student_angles:
            coach_angle = coach_angles[joint]
            student_angle = student_angles[joint]
            diff = abs(coach_angle - student_angle)
            
            comparison[joint] = {
                'coach_angle': coach_angle,
                'student_angle': student_angle,
                'difference': diff,
                'status': 'match' if diff <= threshold else 'mismatch',
                'color': (0, 255, 0) if diff <= threshold else (0, 0, 255)  # Green or Red
            }
    
    return comparison


def compare_angle_series(
    coach_series: List[Dict[str, float]],
    student_series: List[Dict[str, float]],
    joint: str = 'right_elbow'
) -> Dict[str, float]:
    """
    Compare angle time series for a specific joint.
    
    Args:
        coach_series: List of angle dictionaries from coach video frames
        student_series: List of angle dictionaries from student video frames
        joint: Joint name to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    # Extract angle values for the specific joint
    coach_vals = [frame_angles.get(joint, 0) for frame_angles in coach_series if frame_angles]
    student_vals = [frame_angles.get(joint, 0) for frame_angles in student_series if frame_angles]
    
    if not coach_vals or not student_vals:
        return {'error': 'No valid angle data'}
    
    # Calculate DTW distance
    dtw_distance = simple_dtw(coach_vals, student_vals)
    
    # Calculate statistics
    coach_mean = np.mean(coach_vals)
    student_mean = np.mean(student_vals)
    mean_diff = abs(coach_mean - student_mean)
    
    # Calculate correlation
    min_len = min(len(coach_vals), len(student_vals))
    if min_len > 1:
        correlation = float(np.corrcoef(coach_vals[:min_len], student_vals[:min_len])[0, 1])
    else:
        correlation = 0.0
    
    return {
        'dtw_distance': dtw_distance,
        'coach_mean': coach_mean,
        'student_mean': student_mean,
        'mean_difference': mean_diff,
        'correlation': correlation,
        'coach_std': float(np.std(coach_vals)),
        'student_std': float(np.std(student_vals))
    }


def calculate_movement_similarity(
    coach_angles_list: List[Dict[str, float]],
    student_angles_list: List[Dict[str, float]]
) -> pd.DataFrame:
    """
    Calculate comprehensive movement similarity metrics.
    
    Args:
        coach_angles_list: List of angle dictionaries from coach video
        student_angles_list: List of angle dictionaries from student video
        
    Returns:
        DataFrame with similarity metrics for each joint
    """
    joints = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
    results = []
    
    for joint in joints:
        comparison = compare_angle_series(
            coach_angles_list,
            student_angles_list,
            joint
        )
        
        results.append({
            'Joint': joint.replace('_', ' ').title(),
            'DTW Distance': comparison.get('dtw_distance', 0),
            'Coach Mean': comparison.get('coach_mean', 0),
            'Student Mean': comparison.get('student_mean', 0),
            'Mean Diff': comparison.get('mean_difference', 0),
            'Correlation': comparison.get('correlation', 0)
        })
    
    return pd.DataFrame(results)


def calculate_overall_score(
    coach_angles_list: List[Dict[str, float]],
    student_angles_list: List[Dict[str, float]]
) -> float:
    """
    Calculate an overall similarity score (0-100).
    
    Args:
        coach_angles_list: Coach's angle time series
        student_angles_list: Student's angle time series
        
    Returns:
        Overall similarity score (0-100)
    """
    joints = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
    scores = []
    
    for joint in joints:
        comparison = compare_angle_series(
            coach_angles_list,
            student_angles_list,
            joint
        )
        
        # Score based on mean difference (lower is better)
        mean_diff = comparison.get('mean_difference', 90)
        score = max(0, 100 - mean_diff)
        scores.append(score)
    
    return float(np.mean(scores)) if scores else 0.0


def create_comparison_report(
    coach_angles_list: List[Dict[str, float]],
    student_angles_list: List[Dict[str, float]],
    session_id: str = "001"
) -> pd.DataFrame:
    """
    Create a comprehensive session comparison report.
    
    Args:
        coach_angles_list: Coach's angle measurements
        student_angles_list: Student's angle measurements
        session_id: Session identifier
        
    Returns:
        DataFrame with accuracy metrics
    """
    # Calculate average angles
    key_angles = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
    
    accuracy_data = []
    
    for angle_name in key_angles:
        coach_values = [
            frame_angles.get(angle_name, 0)
            for frame_angles in coach_angles_list
            if frame_angles
        ]
        student_values = [
            frame_angles.get(angle_name, 0)
            for frame_angles in student_angles_list
            if frame_angles
        ]
        
        if coach_values and student_values:
            coach_avg = np.mean(coach_values)
            student_avg = np.mean(student_values)
            diff = abs(coach_avg - student_avg)
            accuracy = max(0, 100 - diff)
            
            accuracy_data.append({
                'Joint': angle_name.replace('_', ' ').title(),
                'Coach Angle': f"{coach_avg:.1f}°",
                'Student Angle': f"{student_avg:.1f}°",
                'Difference': f"{diff:.1f}°",
                'Accuracy': f"{accuracy:.1f}%"
            })
    
    df = pd.DataFrame(accuracy_data)
    
    # Add summary row
    if accuracy_data:
        avg_accuracy = np.mean([
            float(row['Accuracy'].replace('%', ''))
            for row in accuracy_data
        ])
        summary_row = {
            'Joint': 'OVERALL',
            'Coach Angle': '-',
            'Student Angle': '-',
            'Difference': '-',
            'Accuracy': f"{avg_accuracy:.1f}%"
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    return df
