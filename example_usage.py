"""
Simple example to run the analysis
"""

from lie_detection_feature_analysis import LieDetectionPipeline, Config

if __name__ == "__main__":
    print(f"\nInput folder: {Config.INPUT_FOLDER}")
    print(f"Output folder: {Config.OUTPUT_FOLDER}")
    
    pipeline = LieDetectionPipeline()
    pipeline.run()
