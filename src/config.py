"""
Configuration Module

Central configuration settings for EnGarde AI application.
Load settings from environment variables with fallback defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Application configuration with environment variable support."""
    
    # Video Processing Settings
    MAX_VIDEO_FRAMES: int = int(os.getenv('MAX_VIDEO_FRAMES', '300'))
    VIDEO_WIDTH: int = int(os.getenv('VIDEO_WIDTH', '640'))
    VIDEO_HEIGHT: int = int(os.getenv('VIDEO_HEIGHT', '480'))
    
    # Analysis Settings
    ANGLE_THRESHOLD: float = float(os.getenv('ANGLE_THRESHOLD', '15.0'))
    DTW_RADIUS: int = int(os.getenv('DTW_RADIUS', '1'))
    
    # MediaPipe Settings
    MEDIAPIPE_MODEL_COMPLEXITY: int = int(os.getenv('MEDIAPIPE_MODEL_COMPLEXITY', '1'))
    MIN_DETECTION_CONFIDENCE: float = float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.5'))
    MIN_TRACKING_CONFIDENCE: float = float(os.getenv('MIN_TRACKING_CONFIDENCE', '0.5'))
    
    # UI Settings
    THEME: str = os.getenv('THEME', 'dark')
    PAGE_ICON: str = os.getenv('PAGE_ICON', '🤺')
    
    # Data Settings
    DATA_DIR: str = os.getenv('DATA_DIR', 'data')
    SESSIONS_DIR: str = os.getenv('SESSIONS_DIR', 'data/sessions')
    MODELS_DIR: str = os.getenv('MODELS_DIR', 'data/models')
    
    # Export Settings
    EXPORT_CSV: bool = os.getenv('EXPORT_CSV', 'true').lower() == 'true'
    EXPORT_VIDEO: bool = os.getenv('EXPORT_VIDEO', 'false').lower() == 'true'
    
    # Performance Settings
    ENABLE_CACHE: bool = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    
    @classmethod
    def get_video_size(cls) -> tuple:
        """Get video processing size as tuple."""
        return (cls.VIDEO_WIDTH, cls.VIDEO_HEIGHT)
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.SESSIONS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if key.isupper() and not key.startswith('_')
        }


class FencingThresholds:
    """Target angle thresholds for fencing form validation."""
    
    # Knee angles (for lunge depth)
    FRONT_KNEE_MIN: float = 70.0
    FRONT_KNEE_MAX: float = 110.0
    
    # Arm angles
    WEAPON_ARM_MIN: float = 160.0
    WEAPON_ARM_MAX: float = 180.0
    NON_WEAPON_ARM_MIN: float = 80.0
    NON_WEAPON_ARM_MAX: float = 110.0
    
    # Shoulder angles
    WEAPON_SHOULDER_MIN: float = 150.0
    WEAPON_SHOULDER_MAX: float = 180.0
    
    # Hip angles
    HIP_EXTENSION_MIN: float = 160.0
    HIP_EXTENSION_MAX: float = 180.0
    
    @classmethod
    def get_thresholds(cls, joint: str) -> tuple:
        """Get threshold range for a specific joint."""
        thresholds = {
            'right_knee': (cls.FRONT_KNEE_MIN, cls.FRONT_KNEE_MAX),
            'left_knee': (cls.FRONT_KNEE_MIN, cls.FRONT_KNEE_MAX),
            'right_elbow': (cls.WEAPON_ARM_MIN, cls.WEAPON_ARM_MAX),
            'left_elbow': (cls.WEAPON_ARM_MIN, cls.WEAPON_ARM_MAX),
            'right_shoulder': (cls.WEAPON_SHOULDER_MIN, cls.WEAPON_SHOULDER_MAX),
            'left_shoulder': (cls.WEAPON_SHOULDER_MIN, cls.WEAPON_SHOULDER_MAX),
            'right_hip': (cls.HIP_EXTENSION_MIN, cls.HIP_EXTENSION_MAX),
            'left_hip': (cls.HIP_EXTENSION_MIN, cls.HIP_EXTENSION_MAX),
        }
        return thresholds.get(joint, (0.0, 180.0))


# Default configuration instance
config = Config()
