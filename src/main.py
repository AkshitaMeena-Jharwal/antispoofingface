# src/main.py
import click
import cv2
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.anti_spoofing import EnhancedFaceAntiSpoofing
from utils.config import load_config

@click.command()
@click.option('--mode', type=click.Choice(['live', 'image', 'video']), default='live', help='Detection mode')
@click.option('--input', type=str, help='Input file path for image/video mode')
@click.option('--output', type=str, help='Output file path')
@click.option('--config', type=str, help='Config file path')
@click.option('--model', type=str, help='Model file path')
def main(mode, input, output, config, model):
    """Face Anti-Spoofing System CLI"""
    
    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        cfg = load_config('configs/default_config.yaml')
    
    # Initialize detector
    detector = EnhancedFaceAntiSpoofing()
    
    # Load model if provided
    if model:
        detector.load_model(model)
    
    if mode == 'live':
        click.echo("Starting live detection...")
        detector.run_live_detection()
    
    elif mode == 'image':
        if not input:
            click.echo("Error: Input file required for image mode")
            return
        
        # Process single image
        image = cv2.imread(input)
        if image is None:
            click.echo(f"Error: Could not load image {input}")
            return
        
        is_real, confidence, results = detector.comprehensive_spoof_detection(image)
        
        click.echo(f"Image: {input}")
        click.echo(f"Result: {'REAL' if is_real else 'FAKE'}")
        click.echo(f"Confidence: {confidence:.2f}")
        click.echo(f"Blink Count: {results['blink_stats']['total_blinks']}")
        click.echo(f"Texture Score: {results['texture_score']:.2f}")
    
    elif mode == 'video':
        if not input:
            click.echo("Error: Input file required for video mode")
            return
        
        # Process video file
        detector.process_video(input, output)

if __name__ == '__main__':
    main()

# src/utils/config.py
import yaml
import os
from pathlib import Path

def load_config(config_path='configs/default_config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        # Create default config if not exists
        create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_default_config(config_path):
    """Create default configuration file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    default_config = {
        'detection': {
            'eye_ar_thresh': 0.25,
            'eye_ar_consec_frames': 3,
            'texture_threshold': 100.0,
            'blink_timeout': 10.0,
            'min_blink_rate': 10,
            'max_blink_rate': 25
        },
        'model': {
            'input_size': [64, 64],
            'model_path': 'models/face_antispoofing_model.h5',
            'predictor_path': 'shape_predictor_68_face_landmarks.dat'
        },
        'camera': {
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30
        },
        'output': {
            'save_frames': False,
            'output_dir': 'output',
            'log_level': 'INFO'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

# src/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class RealTimeVisualizer:
    def __init__(self):
        self.detection_history = []
        self.blink_history = []
        self.confidence_history = []
        self.max_history = 100
    
    def update_history(self, detection_results: Dict):
        """Update visualization history"""
        self.detection_history.append(detection_results['is_real'])
        self.blink_history.append(detection_results['blink_stats']['total_blinks'])
        self.confidence_history.append(detection_results['confidence_score'])
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
            self.blink_history.pop(0)
            self.confidence_history.pop(0)
    
    def draw_overlay(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
