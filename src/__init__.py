"""
EnGarde AI - Source Package

A comprehensive AI-powered fencing coaching application using MediaPipe and Streamlit.

Modules:
- angle_calculator: Calculate angles between body landmarks
- pose_analyzer: MediaPipe pose detection and analysis
- movement_compare: DTW-based movement comparison
- video_processor: Video processing utilities
- config: Application configuration
- constants: Application constants
- session_manager: Session data management
"""

__version__ = "1.0.0"
__author__ = "EnGarde AI Team"

from src.angle_calculator import AngleCalculator
from src.pose_analyzer import PoseAnalyzer
from src.movement_compare import compare_movements, simple_dtw
from src.video_processor import process_video_stream
from src.config import Config, config, FencingThresholds
from src.constants import *
from src.session_manager import SessionManager, get_session_manager

__all__ = [
    "AngleCalculator",
    "PoseAnalyzer",
    "compare_movements",
    "simple_dtw",
    "process_video_stream",
    "Config",
    "config",
    "FencingThresholds",
    "SessionManager",
    "get_session_manager",
]
