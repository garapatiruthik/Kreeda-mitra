"""
Constants Module

Application-wide constants for EnGarde AI.
"""

# Application Info
APP_NAME = "EnGarde AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Real-Time Fencing Coach"

# MediaPipe Landmark Indices
class LandmarkIndices:
    """MediaPipe pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Key Joints for Fencing Analysis
KEY_JOINTS = [
    'right_elbow',
    'left_elbow',
    'right_knee',
    'left_knee',
    'right_hip',
    'left_hip',
    'right_shoulder',
    'left_shoulder',
    'torso'
]

# Joint Display Names
JOINT_DISPLAY_NAMES = {
    'right_elbow': 'Right Elbow',
    'left_elbow': 'Left Elbow',
    'right_knee': 'Right Knee',
    'left_knee': 'Left Knee',
    'right_hip': 'Right Hip',
    'left_hip': 'Left Hip',
    'right_shoulder': 'Right Shoulder',
    'left_shoulder': 'Left Shoulder',
    'torso': 'Torso'
}

# Fencing Technique Names
TECHNIQUES = {
    'lunge': {
        'name': 'Lunge',
        'description': 'Primary attacking technique in fencing',
        'key_angles': ['right_knee', 'right_elbow', 'right_hip']
    },
    'en_garde': {
        'name': 'En Garde',
        'description': 'Fundamental ready stance',
        'key_angles': ['right_knee', 'left_knee', 'right_shoulder']
    },
    'extension': {
        'name': 'Extension',
        'description': 'Attack preparation',
        'key_angles': ['right_elbow', 'right_shoulder']
    },
    'retreat': {
        'name': 'Retreat',
        'description': 'Backward movement',
        'key_angles': ['left_knee', 'right_knee']
    },
    'advance': {
        'name': 'Advance',
        'description': 'Forward movement',
        'key_angles': ['left_knee', 'right_knee']
    }
}

# Color Palette (BGR format for OpenCV)
COLORS = {
    'success': (0, 255, 0),      # Green
    'warning': (0, 255, 255),    # Yellow
    'error': (0, 0, 255),        # Red
    'info': (255, 255, 0),        # Cyan
    'primary': (79, 70, 229),    # Indigo
    'secondary': (16, 185, 129),  # Emerald
}

# Video Settings
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
DEFAULT_FPS = 30
MAX_UPLOAD_SIZE_MB = 100

# Session Report Columns
SESSION_REPORT_COLUMNS = [
    'Joint',
    'Coach Angle',
    'Student Angle',
    'Difference',
    'Accuracy'
]

# Theory Lab Content
FENCING_BASICS = {
    'positions': {
        'en_garde': {
            'title': 'En Garde Position',
            'description': 'The fundamental ready stance in fencing',
            'key_points': [
                'Feet shoulder-width apart',
                'Back foot at 90° to front foot',
                'Knees slightly bent',
                'Weight evenly distributed',
                'Weapon arm at 100-110°'
            ]
        },
        'on_guard': {
            'title': 'On Guard Position',
            'description': 'Alternative ready stance',
            'key_points': [
                'More athletic stance',
                'Knees deeply bent',
                'Weapon arm extended forward',
                'Non-weapon arm raised'
            ]
        }
    },
    'attacks': {
        'lunge': {
            'title': 'The Lunge',
            'description': 'Primary attacking technique',
            'steps': [
                'Start in En Garde',
                'Extend weapon arm straight',
                'Push off from back leg',
                'Front leg extends forward',
                'Recover to En Garde'
            ]
        },
        'fleche': {
            'title': 'Fleche (Arrow)',
            'description': 'Running attack',
            'steps': [
                'Begin from En Garde',
                'Accelerate forward',
                'Transfer weight forward',
                'Extend arm during flight'
            ]
        }
    },
    'parries': {
        'sixte': {
            'title': 'Sixte Parry',
            'description': 'High outside parry',
            'position': 'Weapon arm high, point online'
        },
        'quarte': {
            'title': 'Quarte Parry',
            'description': 'High inside parry',
            'position': 'Weapon arm high, hand supinated'
        },
        'prime': {
            'title': 'Prime Parry',
            'description': 'Low outside parry',
            'position': 'Weapon arm low, hand pronated'
        }
    }
}

# Error Messages
ERROR_MESSAGES = {
    'video_load': 'Failed to load video file. Please ensure the file is a valid video format.',
    'pose_detection': 'No pose detected. Please ensure the subject is visible in the frame.',
    'processing': 'Error processing video. Please try with a different video.',
    'export': 'Failed to export data. Please try again.',
    'invalid_file': 'Invalid file format. Please upload a video file (MP4, AVI, MOV).'
}

# Success Messages
SUCCESS_MESSAGES = {
    'video_processed': 'Video processed successfully!',
    'export_complete': 'Export completed successfully!',
    'session_saved': 'Session saved successfully!'
}

# Navigation
NAV_PAGES = [
    {'id': 'dashboard', 'name': 'Dashboard', 'icon': '🏠'},
    {'id': 'theory_lab', 'name': 'Theory Lab', 'icon': '📚'},
    {'id': 'analytics', 'name': 'Performance Analytics', 'icon': '📈'}
]
