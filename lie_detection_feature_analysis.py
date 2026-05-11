# MAIN ANALYSIS PIPELINE - Copy this file
"""
State-of-the-Art Feature Analysis Pipeline for Lie Detection Dataset
=====================================================================

This module provides comprehensive feature importance analysis for lie detection
using separate CSV files for truth and lie data from video frame extraction.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from scipy import stats
import shap
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the analysis pipeline"""
    INPUT_FOLDER = './data'
    OUTPUT_FOLDER = './results'
    TRUTH_PATTERN = '*Truth.csv'
    LIE_PATTERN = '*Lie.csv'
    EXCLUDE_COLUMNS = ['file_name']
    DPI = 300
    FIGSIZE_LARGE = (16, 10)
    FIGSIZE_MEDIUM = (14, 8)
    RANDOM_STATE = 42
    N_ESTIMATORS = 200
    CV_FOLDS = 5
    SHAP_SAMPLE_SIZE = 100

# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Handles loading and combining truth/lie CSV files"""
    
    def __init__(self, input_folder=Config.INPUT_FOLDER):
        self.input_folder = input_folder
        self.combined_data = None
        self.y_labels = None
        
    def load_data(self):
        """Load all truth and lie CSV files"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        truth_files = glob.glob(os.path.join(self.input_folder, Config.TRUTH_PATTERN))
        lie_files = glob.glob(os.path.join(self.input_folder, Config.LIE_PATTERN))
        
        print(f"\n✓ Found {len(truth_files)} truth file(s)")
        for f in truth_files:
            print(f"  - {os.path.basename(f)}")
        
        print(f"\n✓ Found {len(lie_files)} lie file(s)")
        for f in lie_files:
            print(f"  - {os.path.basename(f)}")
        
        print("\n📂 Loading truth data...")
        truth_dfs = []
        for file in truth_files:
            df = pd.read_csv(file)
            truth_dfs.append(df)
            print(f"  ✓ {os.path.basename(file)}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        truth_data = pd.concat(truth_dfs, ignore_index=True)
        print(f"\n  Total truth samples: {truth_data.shape[0]}")
        
        print("\n📂 Loading lie data...")
        lie_dfs = []
        for file in lie_files:
            df = pd.read_csv(file)
            lie_dfs.append(df)
            print(f"  ✓ {os.path.basename(file)}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        lie_data = pd.concat(lie_dfs, ignore_index=True)
        print(f"\n  Total lie samples: {lie_data.shape[0]}")
        
        truth_data['label'] = 0
        lie_data['label'] = 1
        
        self.combined_data = pd.concat([truth_data, lie_data], ignore_index=True)
        self.y_labels = self.combined_data['label'].values
        
        print("\n" + "=" * 80)
        print(f"✓ DATA LOADING COMPLETE")
        print("=" * 80)
        print(f"\nDataset Summary:")
        print(f"  Total samples: {len(self.combined_data)}")
        print(f"  Truth samples: {(self.y_labels == 0).sum()}")
        print(f"  Lie samples: {(self.y_labels == 1).sum()}")
        print(f"  Total features: {self.combined_data.shape[1] - 1}")
        
        return self.combined_data, self.y_labels
    
    def get_feature_matrix(self):
        """Extract feature matrix"""
        X = self.combined_data.drop(columns=Config.EXCLUDE_COLUMNS + ['label'])
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)
        return X

# ============================================================================
# FEATURE CATEGORIZATION
# ============================================================================

class FeatureCategorizer:
    """Categorizes features by type"""
    
    @staticmethod
    def categorize_features(feature_names):
        categories = {
            'Action Units': [],
            'AU Intensity': [],
            'Head Pose': [],
            'Landmarks': [],
            'Facial Expression': [],
            'Other': []
        }
        
        for feat in feature_names:
            if 'detected_aus' in feat.lower():
                categories['Action Units'].append(feat)
            elif 'au_intensities' in feat.lower():
                categories['AU Intensity'].append(feat)
            elif feat.lower() in ['pitch', 'yaw', 'roll']:
                categories['Head Pose'].append(feat)
            elif 'lm_mp' in feat.lower():
                categories['Landmarks'].append(feat)
            elif 'facial_expression' in feat.lower():
                categories['Facial Expression'].append(feat)
            else:
                categories['Other'].append(feat)
        
        return categories

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalysis:
    """Statistical comparison between lie and truth groups"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = []
    
    def perform_analysis(self):
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS (Lie vs Truth)")
        print("=" * 80)
        
        truth_data = self.X[self.y == 0]
        lie_data = self.X[self.y == 1]
        
        print(f"\nTruth samples: {len(truth_data)}")
        print(f"Lie samples: {len(lie_data)}")
        print("\n📊 Computing statistical tests...")
        
        for col in self.X.columns:
            truth_vals = truth_data[col].values
            lie_vals = lie_data[col].values
            
            truth_vals = truth_vals[~np.isnan(truth_vals)]
            lie_vals = lie_vals[~np.isnan(lie_vals)]
            
            if len(truth_vals) > 1 and len(lie_vals) > 1:
                t_stat, p_value = stats.ttest_ind(lie_vals, truth_vals)
                
                mean_diff = lie_vals.mean() - truth_vals.mean()
                pooled_std = np.sqrt(((len(lie_vals)-1)*lie_vals.std()**2 + 
                                     (len(truth_vals)-1)*truth_vals.std()**2) / 
                                    (len(lie_vals) + len(truth_vals) - 2))
                cohens_d = mean_diff / (pooled_std + 1e-8)
                
                self.results.append({
                    'Feature': col,
                    'Truth_Mean': truth_vals.mean(),
                    'Lie_Mean': lie_vals.mean(),
                    'Mean_Difference': mean_diff,
                    'T_Statistic': t_stat,
                    'P_Value': p_value,
                    'Cohens_D': cohens_d,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
        
        self.results_df = pd.DataFrame(self.results)
        self.results_df = self.results_df.sort_values('Cohens_D', key=abs, ascending=False)
        
        print(f"✓ Statistical analysis complete for {len(self.results)} features")
        return self.results_df
    
    def get_significant_features(self):
        return self.results_df[self.results_df['P_Value'] < 0.05]

# ============================================================================
# MODEL TRAINING
# ============================================================================

class LieDetectionModel:
    """Train and evaluate lie detection model"""
    
    def __init__(self, X, y, random_state=Config.RANDOM_STATE):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.cv_scores = None
    
    def train_and_evaluate(self):
        print("\n" + "=" * 80)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 80)
        
        print("\n🔄 Scaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("🤖 Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        self.model.fit(self.X_scaled, self.y)
        
        print("📊 Performing cross-validation...")
        self.cv_scores = cross_val_score(
            self.model, self.X_scaled, self.y,
            cv=Config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        print("\n✓ MODEL TRAINING COMPLETE")
        print(f"\nCross-Validation Results (F1-Score):")
        print(f"  Mean: {self.cv_scores.mean():.4f}")
        print(f"  Std:  {self.cv_scores.std():.4f}")
        
        train_score = self.model.score(self.X_scaled, self.y)
        print(f"Training Accuracy: {train_score:.4f}")
        
        return self.model, self.cv_scores

# ============================================================================
# SHAP ANALYSIS
# ============================================================================

class SHAPAnalyzer:
    """SHAP-based feature importance analysis"""
    
    def __init__(self, model, X_scaled, feature_names):
        self.model = model
        self.X_scaled = X_scaled
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def compute_shap_values(self):
        print("\n" + "=" * 80)
        print("COMPUTING SHAP VALUES")
        print("=" * 80)
        
        print("\n🔍 Creating TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        sample_size = min(Config.SHAP_SAMPLE_SIZE, self.X_scaled.shape[0])
        indices = np.random.choice(self.X_scaled.shape[0], sample_size, replace=False)
        X_sample = self.X_scaled[indices]
        
        print(f"📊 Computing SHAP values for {sample_size} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        
        print("✓ SHAP computation complete")
        return self.shap_values
    
    def get_feature_importance(self):
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        importance = np.abs(shap_vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': importance,
            'Mean_SHAP_Value': shap_vals.mean(axis=0)
        }).sort_values('SHAP_Importance', ascending=False)
        
        return importance_df, shap_vals

# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================

class PermutationAnalyzer:
    """Permutation-based feature importance"""
    
    def __init__(self, model, X_scaled, y, feature_names):
        self.model = model
        self.X_scaled = X_scaled
        self.y = y
        self.feature_names = feature_names
        self.perm_importance = None
    
    def compute_permutation_importance(self):
        print("\n" + "=" * 80)
        print("COMPUTING PERMUTATION IMPORTANCE")
        print("=" * 80)
        
        print("\n🔄 Computing permutation importance...")
        
        self.perm_importance = permutation_importance(
            self.model, self.X_scaled, self.y,
            n_repeats=10,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        perm_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Mean': self.perm_importance.importances_mean,
            'Importance_Std': self.perm_importance.importances_std
        }).sort_values('Importance_Mean', ascending=False)
        
        print("✓ Permutation importance computed")
        return perm_df

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Create visualizations"""
    
    def __init__(self, output_folder=Config.OUTPUT_FOLDER):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_shap_summary(self, shap_values, X_scaled, feature_names):
        print("\n📊 Creating SHAP visualizations...")
        
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_LARGE)
        shap.summary_plot(sv, X_scaled, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '01_shap_summary_plot.png'),
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
        print("  ✓ SHAP summary plot saved")
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_LARGE)
        shap.summary_plot(sv, X_scaled, feature_names=feature_names,
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '02_shap_bar_plot.png'),
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
        print("  ✓ SHAP bar plot saved")
    
    def plot_feature_comparison(self, stat_results_df, top_n=20):
        print("\n📊 Creating feature comparison plot...")
        
        top_features = stat_results_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_LARGE)
        colors = ['red' if x > 0 else 'blue' for x in top_features['Mean_Difference']]
        ax.barh(range(len(top_features)), top_features['Cohens_D'].abs(), color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
        ax.set_title(f"Top {top_n} Features Distinguishing Lie vs Truth",
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '03_feature_effect_sizes.png'),
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature comparison plot saved")
    
    def plot_permutation_importance(self, perm_df, top_n=20):
        print("\n📊 Creating permutation importance plot...")
        
        top_features = perm_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        ax.barh(range(len(top_features)), top_features['Importance_Mean'], 
               xerr=top_features['Importance_Std'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel("Permutation Importance", fontsize=12, fontweight='bold')
        ax.set_title(f"Top {top_n} Features by Permutation Importance",
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '05_permutation_importance.png'),
                   dpi=Config.DPI, bbox_inches='tight')
        plt.close()
        print("  ✓ Permutation importance plot saved")

# ============================================================================
# REPORTING
# ============================================================================

class ReportGenerator:
    """Generate reports"""
    
    def __init__(self, output_folder=Config.OUTPUT_FOLDER):
        self.output_folder = output_folder
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_summary_report(self, summary_dict):
        print("\n" + "=" * 80)
        print("GENERATING REPORTS")
        print("=" * 80)
        
        report_path = os.path.join(self.output_folder, '00_ANALYSIS_SUMMARY.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LIE DETECTION FEATURE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            
            for section, content in summary_dict.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"{section}\n")
                f.write(f"{'='*80}\n")
                f.write(str(content) + "\n")
        
        print(f"\n✓ Summary report saved")
        return report_path
    
    def save_dataframe_reports(self, dataframes_dict):
        print("\n📊 Saving CSV reports...")
        
        for name, df in dataframes_dict.items():
            file_path = os.path.join(self.output_folder, f"{name}.csv")
            df.to_csv(file_path, index=False)
            print(f"  ✓ {name}.csv ({len(df)} rows)")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class LieDetectionPipeline:
    """Main orchestration pipeline"""
    
    def __init__(self, input_folder=Config.INPUT_FOLDER):
        self.input_folder = input_folder
    
    def run(self):
        """Execute complete analysis pipeline"""
        
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + "STATE-OF-THE-ART LIE DETECTION FEATURE ANALYSIS".center(78) + "║")
        print("╚" + "=" * 78 + "╝")
        
        print("\n[STEP 1/7] Loading data...")
        loader = DataLoader(self.input_folder)
        X_raw, y = loader.load_data()
        X = loader.get_feature_matrix()
        feature_names = X.columns.tolist()
        
        print("\n[STEP 2/7] Categorizing features...")
        categorizer = FeatureCategorizer()
        categories = categorizer.categorize_features(feature_names)
        
        print("\n[STEP 3/7] Performing statistical analysis...")
        stat_analyzer = StatisticalAnalysis(X, y)
        stat_results = stat_analyzer.perform_analysis()
        sig_features = stat_analyzer.get_significant_features()
        print(f"  Significant features (p < 0.05): {len(sig_features)}")
        
        print("\n[STEP 4/7] Training machine learning model...")
        ml_model = LieDetectionModel(X, y)
        model, cv_scores = ml_model.train_and_evaluate()
        
        print("\n[STEP 5/7] Computing SHAP values...")
        shap_analyzer = SHAPAnalyzer(model, ml_model.X_scaled, feature_names)
        shap_values = shap_analyzer.compute_shap_values()
        shap_importance, shap_vals = shap_analyzer.get_feature_importance()
        
        print("\n[STEP 6/7] Computing permutation importance...")
        perm_analyzer = PermutationAnalyzer(model, ml_model.X_scaled, y, feature_names)
        perm_importance = perm_analyzer.compute_permutation_importance()
        
        print("\n[STEP 7/7] Creating visualizations and reports...")
        visualizer = Visualizer()
        visualizer.plot_shap_summary(shap_values, ml_model.X_scaled, feature_names)
        visualizer.plot_feature_comparison(stat_results, top_n=20)
        visualizer.plot_permutation_importance(perm_importance, top_n=20)
        
        summary = {
            'DATASET SUMMARY': f"""Total Samples: {len(X)}
  - Truth: {(y == 0).sum()}
  - Lie: {(y == 1).sum()}
Total Features: {len(feature_names)}""",
            'MODEL PERFORMANCE': f"""F1-Score Mean: {cv_scores.mean():.4f}
F1-Score Std: {cv_scores.std():.4f}
Training Accuracy: {model.score(ml_model.X_scaled, y):.4f}""",
            'SIGNIFICANT FEATURES': f"""Count: {len(sig_features)}
Top 5: {sig_features.head(5)[['Feature', 'Cohens_D', 'P_Value']].to_string()}"""
        }
        
        reporter = ReportGenerator()
        reporter.generate_summary_report(summary)
        
        dataframes = {
            '01_statistical_analysis': stat_results,
            '02_shap_importance': shap_importance,
            '03_permutation_importance': perm_importance,
            '04_significant_features': sig_features
        }
        reporter.save_dataframe_reports(dataframes)
        
        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {Config.OUTPUT_FOLDER}/")

if __name__ == "__main__":
    pipeline = LieDetectionPipeline(input_folder=Config.INPUT_FOLDER)
    pipeline.run()
