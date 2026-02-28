"""
Unit tests for movement_compare module.
"""

import pytest
import numpy as np
from src.movement_compare import (
    simple_dtw,
    fast_dtw,
    compare_movements,
    compare_angle_series,
    calculate_movement_similarity,
    calculate_overall_score,
    create_comparison_report
)


class TestSimpleDTW:
    """Test cases for DTW algorithm."""
    
    def test_identical_series(self):
        """Test with identical series - should return 0."""
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        distance = simple_dtw(series, series)
        assert distance == 0.0
    
    def test_similar_series(self):
        """Test with similar series."""
        series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        series2 = [1.1, 2.1, 2.9, 4.1, 5.1]
        distance = simple_dtw(series1, series2)
        assert distance < 1.0
    
    def test_different_series(self):
        """Test with very different series."""
        series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        series2 = [50.0, 60.0, 70.0, 80.0, 90.0]
        distance = simple_dtw(series1, series2)
        assert distance > 40.0
    
    def test_empty_series(self):
        """Test with empty series."""
        distance = simple_dtw([], [1.0, 2.0])
        assert distance == float('inf')
    
    def test_single_element_series(self):
        """Test with single element series."""
        distance = simple_dtw([5.0], [5.0])
        assert distance == 0.0
    
    def test_different_lengths(self):
        """Test with different length series."""
        series1 = [1.0, 2.0, 3.0]
        series2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        distance = simple_dtw(series1, series2)
        assert distance < float('inf')


class TestFastDTW:
    """Test cases for Fast DTW algorithm."""
    
    def test_fast_dtw_basic(self):
        """Test basic Fast DTW functionality."""
        series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        series2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        distance = fast_dtw(series1, series2)
        assert distance == 0.0
    
    def test_fast_dtw_with_radius(self):
        """Test Fast DTW with custom radius."""
        series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        series2 = [1.5, 2.5, 3.5, 4.5, 5.5]
        distance = fast_dtw(series1, series2, radius=2)
        assert distance < 1.0
    
    def test_fast_dtw_empty(self):
        """Test Fast DTW with empty series."""
        distance = fast_dtw([], [1.0, 2.0])
        assert distance == float('inf')


class TestCompareMovements:
    """Test cases for movement comparison."""
    
    def test_compare_movements_match(self):
        """Test comparison with matching angles."""
        coach_angles = {
            'right_elbow': 170.0,
            'right_knee': 90.0,
            'right_hip': 170.0
        }
        student_angles = {
            'right_elbow': 168.0,
            'right_knee': 92.0,
            'right_hip': 168.0
        }
        
        result = compare_movements(coach_angles, student_angles, threshold=15.0)
        
        assert result['right_elbow']['status'] == 'match'
        assert result['right_knee']['status'] == 'match'
    
    def test_compare_movements_mismatch(self):
        """Test comparison with mismatched angles."""
        coach_angles = {'right_elbow': 170.0}
        student_angles = {'right_elbow': 140.0}
        
        result = compare_movements(coach_angles, student_angles, threshold=15.0)
        
        assert result['right_elbow']['status'] == 'mismatch'
    
    def test_compare_movements_missing_joint(self):
        """Test with missing joint in student angles."""
        coach_angles = {'right_elbow': 170.0}
        student_angles = {}
        
        result = compare_movements(coach_angles, student_angles)
        
        assert 'right_elbow' not in result


class TestCompareAngleSeries:
    """Test cases for angle series comparison."""
    
    def test_compare_angle_series_basic(self):
        """Test basic angle series comparison."""
        coach_series = [
            {'right_elbow': 170, 'right_knee': 90},
            {'right_elbow': 175, 'right_knee': 95},
            {'right_elbow': 165, 'right_knee': 85},
        ]
        student_series = [
            {'right_elbow': 168, 'right_knee': 92},
            {'right_elbow': 173, 'right_knee': 97},
            {'right_elbow': 163, 'right_knee': 87},
        ]
        
        result = compare_angle_series(coach_series, student_series, 'right_elbow')
        
        assert 'dtw_distance' in result
        assert 'coach_mean' in result
        assert 'student_mean' in result
    
    def test_compare_angle_series_empty(self):
        """Test with empty series."""
        result = compare_angle_series([], [], 'right_elbow')
        
        assert 'error' in result


class TestCalculateMovementSimilarity:
    """Test cases for movement similarity calculation."""
    
    def test_calculate_similarity_basic(self):
        """Test basic similarity calculation."""
        coach_series = [
            {'right_elbow': 170, 'left_elbow': 170, 'right_knee': 90, 'left_knee': 90},
            {'right_elbow': 175, 'left_elbow': 175, 'right_knee': 95, 'left_knee': 95},
        ]
        student_series = [
            {'right_elbow': 168, 'left_elbow': 168, 'right_knee': 92, 'left_knee': 92},
            {'right_elbow': 173, 'left_elbow': 173, 'right_knee': 97, 'left_knee': 97},
        ]
        
        result = calculate_movement_similarity(coach_series, student_series)
        
        assert len(result) > 0
        assert 'Joint' in result.columns


class TestCalculateOverallScore:
    """Test cases for overall score calculation."""
    
    def test_calculate_score_perfect_match(self):
        """Test with perfect matching angles."""
        angles = {'right_elbow': 170, 'right_knee': 90}
        coach_series = [angles, angles, angles]
        student_series = [angles, angles, angles]
        
        score = calculate_overall_score(coach_series, student_series)
        
        assert score == 100.0
    
    def test_calculate_score_no_match(self):
        """Test with very different angles."""
        coach_series = [{'right_elbow': 170, 'right_knee': 90}]
        student_series = [{'right_elbow': 70, 'right_knee': 170}]
        
        score = calculate_overall_score(coach_series, student_series)
        
        assert score < 50.0
    
    def test_calculate_score_empty(self):
        """Test with empty series."""
        score = calculate_overall_score([], [])
        
        assert score == 0.0


class TestCreateComparisonReport:
    """Test cases for comparison report generation."""
    
    def test_create_report_basic(self):
        """Test basic report creation."""
        coach_series = [
            {'right_elbow': 170, 'left_elbow': 170, 'right_knee': 90, 'left_knee': 90},
        ]
        student_series = [
            {'right_elbow': 168, 'left_elbow': 168, 'right_knee': 92, 'left_knee': 92},
        ]
        
        report = create_comparison_report(coach_series, student_series)
        
        assert len(report) > 0
        assert 'Joint' in report.columns
        assert 'Coach Angle' in report.columns
        assert 'Student Angle' in report.columns
        assert 'Accuracy' in report.columns
    
    def test_create_report_summary_row(self):
        """Test that report includes summary row."""
        coach_series = [
            {'right_elbow': 170, 'left_elbow': 170, 'right_knee': 90, 'left_knee': 90},
        ]
        student_series = [
            {'right_elbow': 168, 'left_elbow': 168, 'right_knee': 92, 'left_knee': 92},
        ]
        
        report = create_comparison_report(coach_series, student_series)
        
        # Check for OVERALL row
        assert 'OVERALL' in report['Joint'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
