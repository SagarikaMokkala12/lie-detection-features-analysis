"""
Feature Engineering Module
===========================

Computes advanced features from extracted MediaPipe landmarks and AUs.
Useful for additional feature engineering on top of raw extracted features.

Features computed:
- Eye aspect ratio
- Mouth metrics (width, height, aspect ratio)
- Facial symmetry scores
- Head movement dynamics
- Landmark spread analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class LandmarkFeatures:
    """Extract features from MediaPipe facial landmarks"""
    
    # MediaPipe landmark indices
    LEFT_EYE = [33, 133, 159, 158, 144, 145, 153, 154]
    RIGHT_EYE = [362, 263, 387, 386, 373, 374, 380, 381]
    MOUTH = [78, 81, 82, 12, 311, 308, 317, 310, 415, 407, 406, 335]
    NOSE = [1, 2, 98, 327]
    FACE_OUTLINE = list(range(0, 17))
    
    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Euclidean distance between two points"""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    @staticmethod
    def eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """
        Compute eye aspect ratio (EAR)
        Higher EAR = eye open, Lower EAR = eye closed
        """
        # Eye landmarks: [p1, p2, p3, p4, p5, p6]
        # Distance vertical: (p2-p5) + (p3-p4)
        # Distance horizontal: p1-p6
        
        if len(eye_points) < 6:
            return 0.0
        
        vertical1 = LandmarkFeatures.euclidean_distance(eye_points[1], eye_points[4])
        vertical2 = LandmarkFeatures.euclidean_distance(eye_points[2], eye_points[3])
        horizontal = LandmarkFeatures.euclidean_distance(eye_points[0], eye_points[5])
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-8)
        return ear
    
    @staticmethod
    def mouth_width(mouth_points: np.ndarray) -> float:
        """Compute mouth width"""
        if len(mouth_points) < 2:
            return 0.0
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        return LandmarkFeatures.euclidean_distance(left_corner, right_corner)
    
    @staticmethod
    def mouth_height(mouth_points: np.ndarray) -> float:
        """Compute mouth height"""
        if len(mouth_points) < 4:
            return 0.0
        top = mouth_points[2]
        bottom = mouth_points[10]
        return LandmarkFeatures.euclidean_distance(top, bottom)
    
    @staticmethod
    def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
        """Compute mouth aspect ratio (height/width)"""
        width = LandmarkFeatures.mouth_width(mouth_points)
        height = LandmarkFeatures.mouth_height(mouth_points)
        return height / (width + 1e-8)
    
    @staticmethod
    def facial_symmetry(left_points: np.ndarray, right_points: np.ndarray) -> float:
        """
        Compute facial symmetry score
        0 = perfectly symmetric, higher = more asymmetric
        """
        if len(left_points) != len(right_points):
            return 0.0
        
        distances = []
        for l, r in zip(left_points, right_points):
            dist = LandmarkFeatures.euclidean_distance(l, r)
            distances.append(dist)
        
        asymmetry = np.std(distances)
        return asymmetry
    
    @staticmethod
    def landmark_spread(landmarks: np.ndarray) -> float:
        """
        Compute spread of landmarks (how dispersed is the face)
        Uses principal component analysis concept
        """
        if len(landmarks) < 2:
            return 0.0
        
        # Standard deviation across all landmarks
        spread_x = np.std(landmarks[:, 0])
        spread_y = np.std(landmarks[:, 1])
        spread_z = np.std(landmarks[:, 2]) if landmarks.shape[1] > 2 else 0
        
        return np.sqrt(spread_x**2 + spread_y**2 + spread_z**2)


class AUAnalyzer:
    """Analyze Action Units for deception indicators"""
    
    # Known deception-related AUs
    DECEPTION_AUs = {
        1: "Inner Brow Raiser",
        4: "Brow Lowerer",
        5: "Upper Lid Raiser",
        7: "Lid Tightener",
        10: "Upper Lip Raiser",
        12: "Lip Corner Puller (Smile)",
        14: "Dimpler",
        15: "Lip Corner Depressor",
        17: "Chin Raiser",
        20: "Lip Stretcher",
        23: "Lip Tightener",
        24: "Lip Pressor",
        25: "Lips Part",
        26: "Jaw Drop",
        27: "Mouth Stretch"
    }
    
    @staticmethod
    def get_deception_au_subset(au_list: List[int]) -> List[int]:
        """Filter AUs that are known deception indicators"""
        return [au for au in au_list if au in AUAnalyzer.DECEPTION_AUs]
    
    @staticmethod
    def au_intensity_stats(au_intensities: np.ndarray) -> dict:
        """Compute statistics on AU intensities"""
        return {
            'mean_intensity': np.mean(au_intensities),
            'max_intensity': np.max(au_intensities),
            'min_intensity': np.min(au_intensities),
            'std_intensity': np.std(au_intensities),
            'sum_intensity': np.sum(au_intensities)
        }


class TemporalFeatures:
    """Extract temporal/dynamic features"""
    
    @staticmethod
    def compute_velocity(values: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Compute velocity (frame-to-frame change)"""
        return np.diff(values, axis=0) / dt
    
    @staticmethod
    def compute_acceleration(values: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Compute acceleration (change in velocity)"""
        velocity = TemporalFeatures.compute_velocity(values, dt)
        return np.diff(velocity, axis=0) / dt
    
    @staticmethod
    def compute_jitter(values: np.ndarray) -> float:
        """Compute jitter (variability in motion)"""
        if len(values) < 2:
            return 0.0
        velocity = np.diff(values, axis=0)
        return np.mean(np.std(velocity, axis=0))
    
    @staticmethod
    def compute_smoothness(values: np.ndarray) -> float:
        """
        Compute smoothness metric
        Higher = smoother, Lower = jittery
        """
        if len(values) < 3:
            return 1.0
        
        # Compute second derivative
        vel = np.diff(values, axis=0)
        acc = np.diff(vel, axis=0)
        
        # Smoothness inversely proportional to acceleration
        mean_acc = np.mean(np.linalg.norm(acc, axis=1))
        smoothness = 1.0 / (1.0 + mean_acc)
        
        return smoothness


def extract_engineered_features(df: pd.DataFrame, 
                                landmark_cols: List[str]) -> pd.DataFrame:
    """
    Extract engineered features from raw features
    
    Args:
        df: DataFrame with raw features
        landmark_cols: Column names for landmarks (lm_mp_*_x/y/z)
    
    Returns:
        DataFrame with additional engineered features
    """
    
    engineered = pd.DataFrame()
    
    # Extract landmark coordinates
    for col in landmark_cols:
        if col.endswith('_x'):
            landmark_id = col.split('_')[2]
            x_col = f'lm_mp_{landmark_id}_x'
            y_col = f'lm_mp_{landmark_id}_y'
            z_col = f'lm_mp_{landmark_id}_z'
            
            if all(c in df.columns for c in [x_col, y_col, z_col]):
                points = df[[x_col, y_col, z_col]].values
                
                # You can compute per-landmark features here
                # For example, distance from face center, etc.
    
    return engineered


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 60)
    print("\nAvailable classes:")
    print("  - LandmarkFeatures: Eye AR, mouth metrics, symmetry")
    print("  - AUAnalyzer: AU-based deception detection")
    print("  - TemporalFeatures: Velocity, acceleration, jitter")
    print("\nImport and use in your analysis scripts.")
