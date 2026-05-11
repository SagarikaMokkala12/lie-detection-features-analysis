"""
Advanced Feature Engineering for Lie Detection
Derives high-level features from raw landmarks and action units
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class FacialFeatureEngineer:
    """
    Derives advanced facial features from raw landmarks and action units
    """
    
    def __init__(self, df):
        """
        Initialize engineer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with raw features
        """
        self.df = df.copy()
        self.landmark_cols = [col for col in df.columns if 'lm_' in col.lower()]
        self.au_cols = [col for col in df.columns if 'au_' in col.lower() and 'intensity' not in col.lower()]
        self.au_intensity_cols = [col for col in df.columns if 'intensity' in col.lower()]
        
        # Define landmark indices for facial regions
        self.left_eye_indices = [33, 130, 8, 42, 39, 36]      # MediaPipe left eye
        self.right_eye_indices = [263, 359, 133, 155, 246, 161]  # MediaPipe right eye
        self.mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]  # Mouth region
        self.nose_indices = [1, 2, 3, 4, 5, 6]  # Nose region
        self.left_eyebrow_indices = [70, 63, 105, 66, 107]  # Left eyebrow
        self.right_eyebrow_indices = [336, 296, 334, 293, 300]  # Right eyebrow
        
    def extract_landmark_coords(self, row, indices):
        """Extract x, y, z coordinates for specific landmarks"""
        coords = []
        for idx in indices:
            x = row.get(f'lm_mp_{idx}_x', np.nan)
            y = row.get(f'lm_mp_{idx}_y', np.nan)
            z = row.get(f'lm_mp_{idx}_z', np.nan)
            if not pd.isna([x, y, z]).any():
                coords.append([x, y, z])
        return np.array(coords)
    
    def compute_eye_openness(self):
        """Compute eye openness metric (vertical distance between eyelids)"""
        print("  Computing eye openness...")
        
        eye_openness_left = []
        eye_openness_right = []
        
        for idx, row in self.df.iterrows():
            # Left eye: distance between upper and lower eyelid
            left_upper_y = row.get('lm_mp_159_y', np.nan)
            left_lower_y = row.get('lm_mp_145_y', np.nan)
            left_openness = abs(left_upper_y - left_lower_y) if not pd.isna([left_upper_y, left_lower_y]).any() else np.nan
            eye_openness_left.append(left_openness)
            
            # Right eye: distance between upper and lower eyelid
            right_upper_y = row.get('lm_mp_386_y', np.nan)
            right_lower_y = row.get('lm_mp_374_y', np.nan)
            right_openness = abs(right_upper_y - right_lower_y) if not pd.isna([right_upper_y, right_lower_y]).any() else np.nan
            eye_openness_right.append(right_openness)
        
        self.df['eye_openness_left'] = eye_openness_left
        self.df['eye_openness_right'] = eye_openness_right
        self.df['eye_openness_avg'] = np.mean([eye_openness_left, eye_openness_right], axis=0)
        
        return self
    
    def compute_mouth_metrics(self):
        """Compute mouth width, height, and aspect ratio"""
        print("  Computing mouth metrics...")
        
        mouth_width = []
        mouth_height = []
        mouth_aspect_ratio = []
        
        for idx, row in self.df.iterrows():
            # Width: distance between left and right mouth corners
            left_mouth_x = row.get('lm_mp_61_x', np.nan)
            right_mouth_x = row.get('lm_mp_291_x', np.nan)
            width = abs(right_mouth_x - left_mouth_x) if not pd.isna([left_mouth_x, right_mouth_x]).any() else np.nan
            
            # Height: distance between top and bottom lips
            top_mouth_y = row.get('lm_mp_13_y', np.nan)
            bottom_mouth_y = row.get('lm_mp_14_y', np.nan)
            height = abs(bottom_mouth_y - top_mouth_y) if not pd.isna([top_mouth_y, bottom_mouth_y]).any() else np.nan
            
            # Aspect ratio
            ar = height / width if width > 0 and not pd.isna([width, height]).any() else np.nan
            
            mouth_width.append(width)
            mouth_height.append(height)
            mouth_aspect_ratio.append(ar)
        
        self.df['mouth_width'] = mouth_width
        self.df['mouth_height'] = mouth_height
        self.df['mouth_aspect_ratio'] = mouth_aspect_ratio
        
        return self
    
    def compute_facial_symmetry(self):
        """Compute left-right facial symmetry"""
        print("  Computing facial symmetry...")
        
        symmetry_scores = []
        
        for idx, row in self.df.iterrows():
            symmetries = []
            
            # Compare left and right eyes
            left_eye_coords = self.extract_landmark_coords(row, self.left_eye_indices)
            right_eye_coords = self.extract_landmark_coords(row, self.right_eye_indices)
            
            if len(left_eye_coords) > 0 and len(right_eye_coords) > 0:
                eye_sym = np.mean([euclidean(left_eye_coords[i], right_eye_coords[i]) 
                                  for i in range(min(len(left_eye_coords), len(right_eye_coords)))])
                symmetries.append(eye_sym)
            
            # Compare left and right eyebrows
            left_brow_coords = self.extract_landmark_coords(row, self.left_eyebrow_indices)
            right_brow_coords = self.extract_landmark_coords(row, self.right_eyebrow_indices)
            
            if len(left_brow_coords) > 0 and len(right_brow_coords) > 0:
                brow_sym = np.mean([euclidean(left_brow_coords[i], right_brow_coords[i]) 
                                   for i in range(min(len(left_brow_coords), len(right_brow_coords)))])
                symmetries.append(brow_sym)
            
            avg_symmetry = np.mean(symmetries) if symmetries else np.nan
            symmetry_scores.append(avg_symmetry)
        
        self.df['facial_symmetry_score'] = symmetry_scores
        
        return self
    
    def compute_head_dynamics(self):
        """Compute head pose velocity and acceleration"""
        print("  Computing head dynamics...")
        
        # Velocity (change in pitch, yaw, roll)
        self.df['pitch_velocity'] = self.df['pitch'].diff().abs().fillna(0)
        self.df['yaw_velocity'] = self.df['yaw'].diff().abs().fillna(0)
        self.df['roll_velocity'] = self.df['roll'].diff().abs().fillna(0)
        self.df['head_velocity_avg'] = (self.df['pitch_velocity'] + self.df['yaw_velocity'] + self.df['roll_velocity']) / 3
        
        # Acceleration (second derivative)
        self.df['pitch_accel'] = self.df['pitch_velocity'].diff().abs().fillna(0)
        self.df['yaw_accel'] = self.df['yaw_velocity'].diff().abs().fillna(0)
        self.df['roll_accel'] = self.df['roll_velocity'].diff().abs().fillna(0)
        self.df['head_accel_avg'] = (self.df['pitch_accel'] + self.df['yaw_accel'] + self.df['roll_accel']) / 3
        
        return self
    
    def compute_landmark_spread(self):
        """Compute spread/compactness of facial landmarks"""
        print("  Computing landmark spread...")
        
        landmark_spread = []
        
        for idx, row in self.df.iterrows():
            # Extract all landmarks
            all_landmarks = []
            for i in range(468):  # MediaPipe has 468 landmarks
                x = row.get(f'lm_mp_{i}_x', np.nan)
                y = row.get(f'lm_mp_{i}_y', np.nan)
                z = row.get(f'lm_mp_{i}_z', np.nan)
                if not pd.isna([x, y, z]).any():
                    all_landmarks.append([x, y, z])
            
            if len(all_landmarks) > 1:
                all_landmarks = np.array(all_landmarks)
                # Compute pairwise distances
                distances = pdist(all_landmarks)
                spread = np.mean(distances)
                landmark_spread.append(spread)
            else:
                landmark_spread.append(np.nan)
        
        self.df['landmark_spread'] = landmark_spread
        
        return self
    
    def compute_au_intensity_stats(self):
        """Compute statistics on AU intensities"""
        print("  Computing AU intensity statistics...")
        
        au_intensity_mean = []
        au_intensity_max = []
        au_intensity_std = []
        
        for idx, row in self.df.iterrows():
            intensities = [row.get(col, 0) for col in self.au_intensity_cols if col in row.index]
            if intensities:
                au_intensity_mean.append(np.mean(intensities))
                au_intensity_max.append(np.max(intensities))
                au_intensity_std.append(np.std(intensities))
            else:
                au_intensity_mean.append(np.nan)
                au_intensity_max.append(np.nan)
                au_intensity_std.append(np.nan)
        
        self.df['au_intensity_mean'] = au_intensity_mean
        self.df['au_intensity_max'] = au_intensity_max
        self.df['au_intensity_std'] = au_intensity_std
        
        return self
    
    def compute_au_count(self):
        """Count number of active action units"""
        print("  Computing active AU count...")
        
        au_count = []
        
        for idx, row in self.df.iterrows():
            count = sum(1 for col in self.au_cols if row.get(col, 0) > 0)
            au_count.append(count)
        
        self.df['active_au_count'] = au_count
        
        return self
    
    def engineer_all_features(self):
        """Run all feature engineering"""
        print("\n🔧 Feature Engineering in Progress...")
        print(f"   Input shape: {self.df.shape}")
        
        (self.compute_eye_openness()
         .compute_mouth_metrics()
         .compute_facial_symmetry()
         .compute_head_dynamics()
         .compute_landmark_spread()
         .compute_au_intensity_stats()
         .compute_au_count())
        
        print(f"   Output shape: {self.df.shape}")
        print(f"   New features: {self.df.shape[1] - len(self.landmark_cols) - len(self.au_cols) - len(self.au_intensity_cols)}")
        print("✓ Feature engineering complete")
        
        return self.df


def main():
    """Example usage"""
    # Load data
    df = pd.read_csv('your_lie_detection_data.csv')
    
    # Engineer features
    engineer = FacialFeatureEngineer(df)
    df_engineered = engineer.engineer_all_features()
    
    # Save engineered features
    df_engineered.to_csv('lie_detection_data_engineered.csv', index=False)
    print(f"\n✓ Engineered data saved to 'lie_detection_data_engineered.csv'")


if __name__ == '__main__':
    main()
