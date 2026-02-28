"""
Session Manager Module

Handles session data persistence and management for EnGarde AI.
Provides utilities for saving, loading, and managing training sessions.
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import streamlit as st


class SessionManager:
    """
    Manages fencing training sessions and analytics.
    """
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        """
        Initialize the session manager.
        
        Args:
            sessions_dir: Directory to store session data
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state storage
        self._session_data: Dict[str, Any] = {}
        
    def save_session(
        self,
        session_id: str,
        coach_angles: List[Dict],
        student_angles: List[Dict],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save a training session.
        
        Args:
            session_id: Unique session identifier
            coach_angles: List of coach angle measurements
            student_angles: List of student angle measurements
            metadata: Optional session metadata
            
        Returns:
            True if saved successfully
        """
        try:
            session_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'coach_angles': coach_angles,
                'student_angles': student_angles,
                'metadata': metadata or {},
                'frame_count': len(coach_angles)
            }
            
            # Save as JSON
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load a training session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            return None
    
    def list_sessions(self) -> List[Dict]:
        """
        List all saved sessions.
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id'),
                        'timestamp': data.get('timestamp'),
                        'frame_count': data.get('frame_count', 0),
                        'metadata': data.get('metadata', {})
                    })
            except Exception:
                continue
        
        # Sort by timestamp, newest first
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception as e:
                st.error(f"Failed to delete session: {e}")
                return False
        
        return False
    
    def export_session_csv(
        self,
        session_id: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export session data as CSV.
        
        Args:
            session_id: Session identifier
            output_path: Output file path (optional)
            
        Returns:
            Path to exported CSV or None on failure
        """
        session = self.load_session(session_id)
        
        if not session:
            return None
        
        # Determine output path
        if output_path is None:
            output_path = str(self.sessions_dir / f"{session_id}.csv")
        
        try:
            # Combine coach and student data
            data = []
            
            coach_angles = session.get('coach_angles', [])
            student_angles = session.get('student_angles', [])
            
            frame_count = min(len(coach_angles), len(student_angles))
            
            for i in range(frame_count):
                row = {'frame': i}
                
                # Add coach angles
                for joint, angle in coach_angles[i].items():
                    row[f'coach_{joint}'] = angle
                
                # Add student angles
                for joint, angle in student_angles[i].items():
                    row[f'student_{joint}'] = angle
                
                data.append(row)
            
            # Save as CSV
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            st.error(f"Failed to export CSV: {e}")
            return None
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get a summary of a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dictionary or None
        """
        session = self.load_session(session_id)
        
        if not session:
            return None
        
        coach_angles = session.get('coach_angles', [])
        student_angles = session.get('student_angles', [])
        
        # Calculate average accuracy
        key_joints = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee']
        accuracies = []
        
        for joint in key_joints:
            coach_vals = [f.get(joint, 0) for f in coach_angles if f]
            student_vals = [f.get(joint, 0) for f in student_angles if f]
            
            if coach_vals and student_vals:
                coach_mean = sum(coach_vals) / len(coach_vals)
                student_mean = sum(student_vals) / len(student_vals)
                diff = abs(coach_mean - student_mean)
                accuracy = max(0, 100 - diff)
                accuracies.append(accuracy)
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        return {
            'session_id': session_id,
            'timestamp': session.get('timestamp'),
            'frame_count': len(coach_angles),
            'average_accuracy': avg_accuracy,
            'metadata': session.get('metadata', {})
        }
    
    def save_to_session_state(self, key: str, value: Any):
        """Save value to Streamlit session state."""
        st.session_state[key] = value
    
    def get_from_session_state(self, key: str, default: Any = None) -> Any:
        """Get value from Streamlit session state."""
        return st.session_state.get(key, default)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(sessions_dir: str = "data/sessions") -> SessionManager:
    """
    Get or create the global session manager instance.
    
    Args:
        sessions_dir: Directory for session storage
        
    Returns:
        SessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        _session_manager = SessionManager(sessions_dir)
    
    return _session_manager
