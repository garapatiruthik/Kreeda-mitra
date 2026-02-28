"""
Unit tests for AngleCalculator module.
"""

import pytest
import numpy as np
from src.angle_calculator import (
    AngleCalculator,
    calculate_euclidean_distance,
    normalize_angle_series
)


class TestAngleCalculator:
    """Test cases for AngleCalculator class."""
    
    def test_calculate_angle_straight_line(self):
        """Test angle calculation for a straight line (180 degrees)."""
        # Points: (0,0), (1,0), (2,0) - straight line
        angle = AngleCalculator.calculate_angle(
            (0, 0), (1, 0), (2, 0)
        )
        assert 179.0 <= angle <= 180.0
    
    def test_calculate_angle_right_angle(self):
        """Test angle calculation for a right angle (90 degrees)."""
        # Points: (0,0), (1,0), (1,1) - right angle
        angle = AngleCalculator.calculate_angle(
            (0, 0), (1, 0), (1, 1)
        )
        assert 89.0 <= angle <= 91.0
    
    def test_calculate_angle_acute(self):
        """Test angle calculation for an acute angle (< 90 degrees)."""
        # Points forming an acute angle
        angle = AngleCalculator.calculate_angle(
            (0, 0), (1, 0), (1, 0.5)
        )
        assert angle < 90.0
    
    def test_calculate_angle_obtuse(self):
        """Test angle calculation for an obtuse angle (> 90 degrees)."""
        # Points forming an obtuse angle
        angle = AngleCalculator.calculate_angle(
            (0, 0), (1, 0), (0.5, 0.5)
        )
        assert 90.0 < angle < 180.0
    
    def test_calculate_angle_collinear_points(self):
        """Test with nearly collinear points."""
        angle = AngleCalculator.calculate_angle(
            (0, 0), (1, 0.0001), (2, 0)
        )
        # Should be very close to 180 degrees
        assert angle > 179.0
    
    def test_calculate_fencing_angles_with_mock_landmarks(self):
        """Test angle calculation with mock MediaPipe landmarks."""
        # Create mock landmarks with typical fencing position
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        # Simulate a lunge position
        landmarks = [
            MockLandmark(0.5, 0.3),   # 0: nose
            MockLandmark(0.5, 0.35),  # 1: left eye inner
            MockLandmark(0.48, 0.35), # 2: left eye
            MockLandmark(0.46, 0.35), # 3: left eye outer
            MockLandmark(0.5, 0.37),  # 4: right eye inner
            MockLandmark(0.52, 0.37), # 5: right eye
            MockLandmark(0.54, 0.37), # 6: right eye outer
            MockLandmark(0.5, 0.4),  # 7: left ear
            MockLandmark(0.5, 0.4),  # 8: right ear
            MockLandmark(0.48, 0.45),# 9: mouth left
            MockLandmark(0.52, 0.45), # 10: mouth right
            MockLandmark(0.45, 0.5), # 11: left shoulder
            MockLandmark(0.55, 0.5), # 12: right shoulder
            MockLandmark(0.42, 0.65),# 13: left elbow
            MockLandmark(0.58, 0.65),# 14: right elbow
            MockLandmark(0.38, 0.75),# 15: left wrist
            MockLandmark(0.62, 0.75),# 16: right wrist
            MockLandmark(0.48, 0.55),# 17: left pinky
            MockLandmark(0.52, 0.55),# 18: right pinky
            MockLandmark(0.45, 0.6), # 19: left index
            MockLandmark(0.55, 0.6), # 20: right index
            MockLandmark(0.5, 0.55), # 21: left thumb
            MockLandmark(0.5, 0.55), # 22: right thumb
            MockLandmark(0.48, 0.75),# 23: left hip
            MockLandmark(0.52, 0.75),# 24: right hip
            MockLandmark(0.45, 0.9), # 25: left knee
            MockLandmark(0.55, 0.9), # 26: right knee
            MockLandmark(0.43, 1.0), # 27: left ankle
            MockLandmark(0.57, 1.0), # 28: right ankle
        ] + [MockLandmark(0, 0)] * 5  # Pad remaining landmarks
        
        angles = AngleCalculator.calculate_fencing_angles(landmarks)
        
        # Check that angles are calculated
        assert 'right_elbow' in angles
        assert 'left_elbow' in angles
        assert 'right_knee' in angles
        assert 'left_knee' in angles
        assert 'right_hip' in angles
        assert 'left_hip' in angles
    
    def test_calculate_fencing_angles_empty_landmarks(self):
        """Test with empty landmarks list."""
        angles = AngleCalculator.calculate_fencing_angles([])
        assert angles == {}
    
    def test_calculate_fencing_angles_none_landmarks(self):
        """Test with None landmarks."""
        angles = AngleCalculator.calculate_fencing_angles(None)
        assert angles == {}
    
    def test_get_angle_statistics(self):
        """Test statistics calculation for angle series."""
        angle_series = [
            {'right_elbow': 170, 'right_knee': 90},
            {'right_elbow': 175, 'right_knee': 95},
            {'right_elbow': 165, 'right_knee': 85},
        ]
        
        stats = AngleCalculator.get_angle_statistics(angle_series)
        
        assert 'right_elbow' in stats
        assert 'right_knee' in stats
        assert stats['right_elbow']['mean'] == 170.0
        assert stats['right_knee']['mean'] == 90.0
    
    def test_get_angle_statistics_empty(self):
        """Test statistics with empty series."""
        stats = AngleCalculator.get_angle_statistics([])
        assert stats == {}


class TestEuclideanDistance:
    """Test cases for euclidean distance calculation."""
    
    def test_identical_series(self):
        """Test with identical series."""
        series = [1.0, 2.0, 3.0, 4.0]
        distance = calculate_euclidean_distance(series, series)
        assert distance == 0.0
    
    def test_different_length_series(self):
        """Test with different length series."""
        series1 = [1.0, 2.0, 3.0]
        series2 = [1.0, 2.0]
        distance = calculate_euclidean_distance(series1, series2)
        assert distance < float('inf')
    
    def test_empty_series(self):
        """Test with empty series."""
        distance = calculate_euclidean_distance([], [1.0, 2.0])
        assert distance == float('inf')


class TestNormalizeAngleSeries:
    """Test cases for angle series normalization."""
    
    def test_same_length(self):
        """Test with same length - should return unchanged."""
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_angle_series(series, 5)
        assert result == series
    
    def test_upsample(self):
        """Test upsampling a series."""
        series = [0.0, 90.0, 180.0]
        result = normalize_angle_series(series, 6)
        assert len(result) == 6
    
    def test_downsample(self):
        """Test downsampling a series."""
        series = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
        result = normalize_angle_series(series, 3)
        assert len(result) == 3
    
    def test_empty_series(self):
        """Test with empty series."""
        result = normalize_angle_series([], 5)
        assert result == [0.0] * 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
