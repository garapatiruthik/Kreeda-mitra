"""
EnGarde AI - Real-Time Fencing Coach Application
A comprehensive AI-powered fencing coaching tool using MediaPipe and Streamlit

Features:
- Dual video interface for coach/student comparison
- Real-time pose estimation and skeleton overlay
- Angle calculation for fencing movements
- Movement comparison with DTW
- Theory section and session reports
- Performance analytics
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import io

# Page configuration
st.set_page_config(
    page_title="EnGarde AI - Fencing Coach",
    page_icon="🤺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #1c1f26;
    }
    .stVideo {
        border-radius: 10px;
    }
    .metric-card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .success-text {
        color: #4ade80;
        font-weight: bold;
    }
    .warning-text {
        color: #facc15;
        font-weight: bold;
    }
    .error-text {
        color: #f87171;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)


class AngleCalculator:
    """Calculate angles between three body landmarks using vector math"""
    
    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Calculate angle at point b given points a, b, c
        Uses the law of cosines formula
        Returns angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Create vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        
        # Clamp to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return angle
    
    @staticmethod
    def calculate_fencing_angles(landmarks: dict) -> dict:
        """
        Calculate all key fencing angles from MediaPipe landmarks
        Returns a dictionary of angles
        """
        angles = {}
        
        # Get landmark coordinates
        try:
            # Left side body
            left_shoulder = (landmarks[11].x, landmarks[11].y)
            left_elbow = (landmarks[13].x, landmarks[13].y)
            left_wrist = (landmarks[15].x, landmarks[15].y)
            left_hip = (landmarks[23].x, landmarks[23].y)
            left_knee = (landmarks[25].x, landmarks[25].y)
            left_ankle = (landmarks[27].x, landmarks[27].y)
            
            # Right side body
            right_shoulder = (landmarks[12].x, landmarks[12].y)
            right_elbow = (landmarks[14].x, landmarks[14].y)
            right_wrist = (landmarks[16].x, landmarks[16].y)
            right_hip = (landmarks[24].x, landmarks[24].y)
            right_knee = (landmarks[26].x, landmarks[26].y)
            right_ankle = (landmarks[28].x, landmarks[28].y)
            
            # Calculate angles
            # Arm angles (weapon arm - typically right)
            angles['right_elbow'] = AngleCalculator.calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            angles['left_elbow'] = AngleCalculator.calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            
            # Leg angles (lunge analysis)
            angles['right_knee'] = AngleCalculator.calculate_angle(
                right_hip, right_knee, right_ankle
            )
            angles['left_knee'] = AngleCalculator.calculate_angle(
                left_hip, left_knee, left_ankle
            )
            
            # Hip angles
            angles['right_hip'] = AngleCalculator.calculate_angle(
                right_shoulder, right_hip, right_knee
            )
            angles['left_hip'] = AngleCalculator.calculate_angle(
                left_shoulder, left_hip, left_knee
            )
            
            # Shoulder angles
            angles['right_shoulder'] = AngleCalculator.calculate_angle(
                right_elbow, right_shoulder, right_hip
            )
            angles['left_shoulder'] = AngleCalculator.calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            
            # Torso angle (for lunge depth)
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0])/2, 
                          (left_shoulder[1] + right_shoulder[1])/2)
            mid_hip = ((left_hip[0] + right_hip[0])/2,
                      (left_hip[1] + right_hip[1])/2)
            angles['torso'] = AngleCalculator.calculate_angle(
                mid_shoulder, mid_hip, 
                (mid_hip[0], mid_hip[1] + 0.1)
            )
            
        except Exception as e:
            st.error(f"Error calculating angles: {e}")
            
        return angles


class PoseAnalyzer:
    """MediaPipe pose detection and analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list, dict]:
        """
        Process a single frame and return annotated frame, landmarks, and angles
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        landmarks = []
        angles = {}
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate angles
            angles = AngleCalculator.calculate_fencing_angles(landmarks)
        
        return frame, landmarks, angles
    
    def get_skeleton_overlay(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Get frame with skeleton overlay only"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw in specified color
            for landmark in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start = results.pose_landmarks.landmark[connection[0]]
                end = results.pose_landmarks.landmark[connection[1]]
                start_coords = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_coords = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_coords, end_coords, color, 2)
        
        return frame


def calculate_euclidean_distance(angle_series1: list, angle_series2: list) -> float:
    """Calculate average Euclidean distance between two angle series"""
    if len(angle_series1) != len(angle_series2):
        # Pad shorter series
        max_len = max(len(angle_series1), len(angle_series2))
        angle_series1 = angle_series1 + [0] * (max_len - len(angle_series1))
        angle_series2 = angle_series2 + [0] * (max_len - len(angle_series2))
    
    return np.mean(np.sqrt(np.array(angle_series1)**2 + np.array(angle_series2)**2))


def simple_dtw(series1: list, series2: list) -> float:
    """
    Simplified Dynamic Time Warping for movement comparison
    """
    n, m = len(series1), len(series2)
    
    # Create cost matrix
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, :] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series1[i-1] - series2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    return dtw_matrix[n, m] / max(n, m)


def compare_movements(coach_angles: dict, student_angles: dict, threshold: float = 15.0) -> dict:
    """
    Compare student angles to coach angles
    Returns dict with comparison results
    """
    comparison = {}
    
    key_joints = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee', 
                  'right_hip', 'left_hip', 'right_shoulder', 'left_shoulder']
    
    for joint in key_joints:
        if joint in coach_angles and joint in student_angles:
            diff = abs(coach_angles[joint] - student_angles[joint])
            comparison[joint] = {
                'diff': diff,
                'status': 'match' if diff <= threshold else 'mismatch',
                'color': (0, 255, 0) if diff <= threshold else (0, 0, 255)
            }
    
    return comparison


def draw_angle_overlays(frame: np.ndarray, landmarks: list, angles: dict) -> np.ndarray:
    """Draw angle values on the frame"""
    h, w, _ = frame.shape
    
    # Key joints to display
    key_joints = {
        'right_elbow': (13, "R Elbow"),
        'left_elbow': (13, "L Elbow"),
        'right_knee': (25, "R Knee"),
        'left_knee': (25, "L Knee"),
        'right_hip': (23, "R Hip"),
        'left_hip': (23, "L Hip"),
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
    
    return frame


def process_video_stream(video_file, analyzer: PoseAnalyzer, max_frames: int = 300):
    """Process video file and return frames with pose overlay"""
    cap = cv2.VideoCapture(video_file)
    
    frames = []
    all_angles = []
    frame_count = 0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while cap.is_open() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        
        # Process frame
        annotated_frame, landmarks, angles = analyzer.process_frame(frame)
        
        # Draw angle overlays
        if landmarks:
            annotated_frame = draw_angle_overlays(annotated_frame, landmarks, angles)
            all_angles.append(angles)
        
        # Convert to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(annotated_frame)
        
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / max_frames, 1.0))
            status_text.text(f"Processing frame {frame_count}/{max_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return frames, all_angles, fps


def create_session_report(coach_angles_list: list, student_angles_list: list, session_id: str = "001") -> pd.DataFrame:
    """Create a session report with accuracy metrics"""
    
    # Calculate average angles
    coach_avg = {}
    student_avg = {}
    
    # Aggregate angles from all frames
    key_angles = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
    
    for angle_name in key_angles:
        coach_values = [frame_angles.get(angle_name, 0) for frame_angles in coach_angles_list if frame_angles]
        student_values = [frame_angles.get(angle_name, 0) for frame_angles in student_angles_list if frame_angles]
        
        if coach_values:
            coach_avg[angle_name] = np.mean(coach_values)
        if student_values:
            student_avg[angle_name] = np.mean(student_values)
    
    # Calculate accuracy
    accuracy_data = []
    for angle_name in key_angles:
        if angle_name in coach_avg and angle_name in student_avg:
            diff = abs(coach_avg[angle_name] - student_avg[angle_name])
            accuracy = max(0, 100 - diff)
            accuracy_data.append({
                'Joint': angle_name.replace('_', ' ').title(),
                'Coach Angle': f"{coach_avg[angle_name]:.1f}°",
                'Student Angle': f"{student_avg[angle_name]:.1f}°",
                'Difference': f"{diff:.1f}°",
                'Accuracy': f"{accuracy:.1f}%"
            })
    
    df = pd.DataFrame(accuracy_data)
    
    # Add summary row
    if accuracy_data:
        avg_accuracy = np.mean([float(row['Accuracy'].replace('%', '')) for row in accuracy_data])
        summary_row = {
            'Joint': 'OVERALL',
            'Coach Angle': '-',
            'Student Angle': '-',
            'Difference': '-',
            'Accuracy': f"{avg_accuracy:.1f}%"
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    return df


def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("🤺 EnGarde AI")
    st.sidebar.markdown("---")
    
    # Navigation menu
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Theory Lab", "Performance Analytics"]
    )
    
    # Initialize pose analyzer
    analyzer = PoseAnalyzer()
    
    if menu == "Dashboard":
        render_dashboard(analyzer)
    elif menu == "Theory Lab":
        render_theory_lab()
    elif menu == "Performance Analytics":
        render_performance_analytics()


def render_dashboard(analyzer: PoseAnalyzer):
    """Render the main dashboard with dual video interface"""
    
    st.title("🤺 EnGarde AI - Real-Time Fencing Coach")
    st.markdown("### Upload videos to compare your technique with a coach")
    
    # Create columns for video uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👨‍🏫 Coach / Reference Video")
        coach_video = st.file_uploader(
            "Upload coach video",
            type=['mp4', 'avi', 'mov'],
            key="coach_video"
        )
        coach_video_enabled = st.checkbox("Enable Coach Video Analysis", value=True)
    
    with col2:
        st.markdown("### 👨‍🎓 Student Practice Video")
        student_video = st.file_uploader(
            "Upload student video",
            type=['mp4', 'avi', 'mov'],
            key="student_video"
        )
        student_video_enabled = st.checkbox("Enable Student Video Analysis", value=True)
    
    # Real-time analysis toggle
    st.markdown("---")
    realtime_mode = st.toggle("Real-time Analysis Mode", value=False)
    
    # Process videos button
    if coach_video or student_video:
        if st.button("🚀 Process Videos", type="primary"):
            with st.spinner("Processing videos..."):
                coach_frames = []
                student_frames = []
                coach_angles_list = []
                student_angles_list = []
                
                # Process coach video
                if coach_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_coach:
                        tmp_coach.write(coach_video.read())
                        coach_path = tmp_coach.name
                    
                    st.info("Processing coach video...")
                    coach_frames, coach_angles_list, coach_fps = process_video_stream(
                        coach_path, analyzer
                    )
                    os.unlink(coach_path)
                    st.success(f"Coach video processed: {len(coach_frames)} frames")
                
                # Process student video
                if student_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_student:
                        tmp_student.write(student_video.read())
                        student_path = tmp_student.name
                    
                    st.info("Processing student video...")
                    student_frames, student_angles_list, student_fps = process_video_stream(
                        student_path, analyzer
                    )
                    os.unlink(student_path)
                    st.success(f"Student video processed: {len(student_frames)} frames")
                
                # Display processed videos
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Video display columns
                video_col1, video_col2 = st.columns(2)
                
                with video_col1:
                    st.markdown("### Coach Reference")
                    if coach_frames:
                        # Display first frame with pose
                        st.image(coach_frames[0], caption="First analyzed frame", use_container_width=True)
                        
                        # Display key angles
                        if coach_angles_list:
                            latest_angles = coach_angles_list[-1] if coach_angles_list else {}
                            st.markdown("#### Key Angles:")
                            for angle_name, angle_value in latest_angles.items():
                                st.write(f"**{angle_name.replace('_', ' ').title()}:** {angle_value:.1f}°")
                
                with video_col2:
                    st.markdown("### Student Attempt")
                    if student_frames:
                        st.image(student_frames[0], caption="First analyzed frame", use_container_width=True)
                        
                        # Display key angles
                        if student_angles_list:
                            latest_angles = student_angles_list[-1] if student_angles_list else {}
                            st.markdown("#### Key Angles:")
                            for angle_name, angle_value in latest_angles.items():
                                st.write(f"**{angle_name.replace('_', ' ').title()}:** {angle_value:.1f}°")
                
                # Movement comparison
                if coach_angles_list and student_angles_list:
                    st.markdown("---")
                    st.markdown("## 🔍 Movement Comparison")
                    
                    # Calculate comparison
                    comparison_df = create_session_report(coach_angles_list, student_angles_list)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visual comparison chart
                    if len(coach_angles_list) > 0 and len(student_angles_list) > 0:
                        # Extract angle time series for key joints
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        fig.suptitle("Angle Comparison Over Time", fontsize=14)
                        
                        joints = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
                        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                        
                        for joint, (row, col) in zip(joints, positions):
                            coach_vals = [f.get(joint, 0) for f in coach_angles_list]
                            student_vals = [f.get(joint, 0) for f in student_angles_list]
                            
                            # Normalize length
                            min_len = min(len(coach_vals), len(student_vals))
                            coach_vals = coach_vals[:min_len]
                            student_vals = student_vals[:min_len]
                            
                            axes[row, col].plot(coach_vals, label='Coach', color='green', alpha=0.7)
                            axes[row, col].plot(student_vals, label='Student', color='blue', alpha=0.7)
                            axes[row, col].set_title(joint.replace('_', ' ').title())
                            axes[row, col].set_xlabel('Frame')
                            axes[row, col].set_ylabel('Angle (degrees)')
                            axes[row, col].legend()
                            axes[row, col].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Export report
                    st.markdown("---")
                    st.markdown("## 💾 Export Session Report")
                    
                    if st.button("Save Session Report"):
                        report_df = create_session_report(coach_angles_list, student_angles_list)
                        csv = report_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV Report",
                            data=csv,
                            file_name="fencing_session_report.csv",
                            mime="text/csv"
                        )
                        
                        st.success("Session report generated!")


def render_theory_lab():
    """Render the theory lab with fencing basics"""
    
    st.title("📚 Theory Lab - Fencing Fundamentals")
    st.markdown("Learn the correct form for essential fencing techniques")
    
    # Fencing basics in sidebar style
    st.sidebar.title("📖 Fencing Basics")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["The Lunge", "En Garde", "Extension", "Common Errors"])
    
    with tab1:
        st.markdown("## 🎯 The Lunge")
        st.markdown("""
        The lunge is the primary attacking technique in fencing. Here's how to execute it correctly:
        
        ### Correct Form:
        1. **Starting Position**: Begin in En Garde stance
        2. **Front Leg**: Extend front leg straight forward, knee should not exceed 90°
        3. **Back Leg**: Push off from back leg, keeping it straight
        4. **Torso**: Lean forward slightly to maintain balance
        5. **Weapon Arm**: Extend weapon arm fully (180°) at shoulder height
        6. **Recovery**: Quick return to En Garde position
        """)
        
        st.info("💡 Key Metric: Front knee angle should not exceed 90° to prevent injury")
        
        # Visual guide
        st.markdown("### Key Angles to Monitor:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Front Knee", "90° max", "Critical for safety")
        with col2:
            st.metric("Back Leg", "170-180°", "Keep straight")
        with col3:
            st.metric("Weapon Arm", "180°", "Full extension")
    
    with tab2:
        st.markdown("## 🦶 En Garde Position")
        st.markdown("""
        The En Garde position is the fundamental ready stance in fencing:
        
        ### Correct Form:
        1. **Feet**: Shoulder-width apart, back foot at 90° to front foot
        2. **Knees**: Slightly bent, weight evenly distributed
        3. **Torso**: Upright but relaxed, facing opponent
        4. **Shoulders**: Level and relaxed
        5. **Weapon Arm**: Held at approximately 100-110° at elbow
        6. **Non-Weapon Arm**: Held up for balance, opposite to weapon arm
        """)
        
        st.info("💡 Key Metric: Watch for stability - minimal wobble indicates good balance")
    
    with tab3:
        st.markdown("## 💪 Extension (Attack Preparation)")
        st.markdown("""
        The extension is crucial for initiating attacks and maintaining Right of Way:
        
        ### Correct Form:
        1. **Arm**: Straight weapon arm, 180° at elbow
        2. **Height**: Arm at shoulder height or slightly above
        3. **Weapon**: Weapon tip pointed at opponent's target area
        4. **Body**: Slight forward inclination to accompany the extension
        5. **Timing**: Arm extension should precede the lunge
        """)
        
        st.warning("⚠️ Common Error: Extending arm AFTER lunging gives opponent advantage")
    
    with tab4:
        st.markdown("## ❌ Common Errors & Corrections")
        
        errors = [
            {
                "error": "Knee超过90°",
                "cause": "Lunging too deep or weak back leg",
                "correction": "Strengthen back leg; limit lunge depth"
            },
            {
                "error": "Arm未完全伸直",
                "cause": "Incomplete extension before lunge",
                "correction": "Practice arm extension separately; ensure 180° before lunging"
            },
            {
                "error": "身体后仰",
                "cause": "Poor balance or fear of falling",
                "correction": "Keep torso forward; strengthen core muscles"
            },
            {
                "error": "Back脚弯曲",
                "cause": "Weak push-off or incorrect starting position",
                "correction": "Keep back leg straight when pushing; practice from correct En Garde"
            },
            {
                "error": "Weapon arm过低",
                "cause": "Incorrect arm position",
                "correction": "Keep weapon arm at shoulder height or higher"
            }
        ]
        
        for i, error in enumerate(errors, 1):
            with st.expander(f"Error {i}: {error['error']}"):
                st.markdown(f"**Cause:** {error['cause']}")
                st.markdown(f"**Correction:** {error['correction']}")


def render_performance_analytics():
    """Render performance analytics and tracking"""
    
    st.title("📈 Performance Analytics")
    st.markdown("Track your progress and identify areas for improvement")
    
    # Placeholder for session history
    st.markdown("### Session History")
    
    # Check if there's session data in session state
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    
    # Add sample data for demonstration
    if not st.session_state.session_history:
        st.session_state.session_history = [
            {'session': 'Session 1', 'date': '2024-01-15', 'accuracy': 72.5, 'consistency': 68.0},
            {'session': 'Session 2', 'date': '2024-01-18', 'accuracy': 78.3, 'consistency': 74.0},
            {'session': 'Session 3', 'date': '2024-01-22', 'accuracy': 81.0, 'consistency': 79.5},
            {'session': 'Session 4', 'date': '2024-01-25', 'accuracy': 85.2, 'consistency': 82.0},
            {'session': 'Session 5', 'date': '2024-01-28', 'accuracy': 88.7, 'consistency': 85.5},
        ]
    
    # Display as dataframe
    history_df = pd.DataFrame(st.session_state.session_history)
    st.dataframe(history_df, use_container_width=True)
    
    # Performance charts
    if len(history_df) > 0:
        st.markdown("### 📊 Progress Over Time")
        
        # Accuracy trend
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy chart
        axes[0].plot(history_df['session'], history_df['accuracy'], 
                     marker='o', linewidth=2, color='#4f46e5', label='Accuracy')
        axes[0].fill_between(history_df['session'], history_df['accuracy'], alpha=0.3, color='#4f46e5')
        axes[0].set_title('Accuracy Trend', fontsize=12)
        axes[0].set_xlabel('Session')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3)
        
        # Consistency chart
        axes[1].plot(history_df['session'], history_df['consistency'], 
                     marker='s', linewidth=2, color='#10b981', label='Consistency')
        axes[1].fill_between(history_df['session'], history_df['consistency'], alpha=0.3, color='#10b981')
        axes[1].set_title('Consistency Trend', fontsize=12)
        axes[1].set_xlabel('Session')
        axes[1].set_ylabel('Consistency (%)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics summary
        st.markdown("### 📋 Statistics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_accuracy = np.mean(history_df['accuracy'])
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        with col2:
            avg_consistency = np.mean(history_df['consistency'])
            st.metric("Average Consistency", f"{avg_consistency:.1f}%")
        
        with col3:
            best_accuracy = np.max(history_df['accuracy'])
            st.metric("Best Accuracy", f"{best_accuracy:.1f}%")
        
        with col4:
            improvement = history_df['accuracy'].iloc[-1] - history_df['accuracy'].iloc[0]
            st.metric("Total Improvement", f"+{improvement:.1f}%")
        
        # Tips based on performance
        st.markdown("### 💡 Personalized Tips")
        
        if avg_accuracy < 70:
            st.warning("Focus on mastering the basic En Garde position before progressing to lunges")
        elif avg_accuracy < 85:
            st.info("Work on your arm extension - aim for consistent 180° before lunging")
        else:
            st.success("Great progress! Focus on speed and recovery for advanced training")
        
        if avg_consistency < 75:
            st.info("Practice holding each position for longer to improve consistency")
    
    # Clear history button
    st.markdown("---")
    if st.button("Clear Session History", type="secondary"):
        st.session_state.session_history = []
        st.success("Session history cleared!")


# Entry point
if __name__ == "__main__":
    main()
