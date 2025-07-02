[#!/usr/bin/env python3
"""
RE-ID Feature Extractor for UCMC Tracking Results
Extracts and stores RE-ID features from tracked object crops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torchreid
from torchvision import transforms
from pathlib import Path
import cv2
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
from datetime import datetime


class ReIDFeatureExtractor:
    """
    Extracts RE-ID features from tracking results and stores them efficiently.
    """
    
    def __init__(self, 
                 model_name: str = 'osnet_x1_0',
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32):
        """
        Initialize RE-ID feature extractor.
        
        Args:
            model_name: Name of the torchreid model to use
            model_path: Path to custom model weights (optional)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for feature extraction
        """
        self.device = device
        self.batch_size = batch_size
        
        # Load RE-ID model
        print(f"Loading RE-ID model: {model_name}")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,  # We're only using for feature extraction
            pretrained=True if not model_path else False
        )
        
        if model_path and os.path.exists(model_path):
            print(f"Loading custom weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get feature dimension
        self.feature_dim = self.model.feature_dim
        print(f"Feature dimension: {self.feature_dim}")
        
        # Setup image preprocessing
        # For inference, we want deterministic transforms
        from torchvision import transforms
        self.preprocess_inference = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # height, width for RE-ID
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of images.
        
        Args:
            images: List of images as numpy arrays (BGR format from cv2)
        
        Returns:
            Features as numpy array of shape (N, feature_dim)
        """
        if not images:
            return np.array([])
        
        features = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess images
            batch_tensors = []
            for img in batch_images:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Apply transforms
                img_tensor = self.preprocess_inference(img_rgb)
                batch_tensors.append(img_tensor)
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch)
                # Normalize features
                batch_features = F.normalize(batch_features, p=2, dim=1)
            
            features.append(batch_features.cpu().numpy())
        
        return np.vstack(features) if features else np.array([])
    
    def process_video_directory(self, video_dir: Union[str, Path], 
                              max_crops_per_track: Optional[int] = None) -> Dict:
        """
        Process a single video directory and extract features for all tracks.
        
        Args:
            video_dir: Path to video directory (containing track_data/)
            max_crops_per_track: Maximum number of crops to use per track (None = use all)
        
        Returns:
            Dictionary containing features and metadata for all tracks
        """
        video_dir = Path(video_dir)
        track_data_dir = video_dir / "track_data"
        crops_dir = track_data_dir / "crops"
        csv_path = track_data_dir / "tracking_results.csv"
        metadata_path = track_data_dir / "metadata.json"
        
        # Validate directory structure
        if not track_data_dir.exists():
            raise ValueError(f"track_data directory not found in {video_dir}")
        if not crops_dir.exists():
            raise ValueError(f"crops directory not found in {track_data_dir}")
        if not csv_path.exists():
            raise ValueError(f"tracking_results.csv not found in {track_data_dir}")
        
        # Load tracking results
        df = pd.read_csv(csv_path)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Group by track_id
        track_groups = df.groupby('track_id')
        
        # Process each track
        track_features = {}
        
        for track_id, track_df in tqdm(track_groups, desc="Processing tracks"):
            # Get all crops for this track
            track_crops = []
            crop_info = []
            
            # Sort by frame to maintain temporal order
            track_df_sorted = track_df.sort_values('frame')
            
            for _, row in track_df_sorted.iterrows():
                # Try different possible formats
                crop_filename = f"f{row['frame']}_t{track_id}.jpg"
                crop_path = crops_dir / crop_filename
                
                # If exact match doesn't exist, try to find the file
                if not crop_path.exists():
                    # Try with integer frame (no leading zeros)
                    crop_filename = f"f{int(row['frame'])}_t{int(track_id)}.jpg"
                    crop_path = crops_dir / crop_filename
                
                if crop_path.exists():
                    img = cv2.imread(str(crop_path))
                    if img is not None:
                        track_crops.append(img)
                        crop_info.append({
                            'frame': row['frame'],
                            'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                            'conf': row['conf'],
                            'class': row['class']
                        })
            
            if not track_crops:
                # Debug: Print expected vs actual
                print(f"Warning: No crops found for track {track_id}")
                if track_id <= 5:  # Debug info for first few tracks
                    print(f"  Expected frame numbers: {track_df_sorted['frame'].tolist()[:5]}...")
                    pattern = f"*_t{int(track_id)}.jpg"
                    matching_files = list(crops_dir.glob(pattern))
                    if matching_files:
                        print(f"  Found {len(matching_files)} files with pattern {pattern}")
                        print(f"  Examples: {[f.name for f in matching_files[:3]]}")
                continue
            
            # Limit crops if specified
            if max_crops_per_track and len(track_crops) > max_crops_per_track:
                # Sample evenly across the track
                indices = np.linspace(0, len(track_crops)-1, max_crops_per_track, dtype=int)
                track_crops = [track_crops[i] for i in indices]
                crop_info = [crop_info[i] for i in indices]
            
            # Extract features
            features = self.extract_features(track_crops)
            
            # Store track information
            track_features[track_id] = {
                'features': features,  # Shape: (n_crops, feature_dim)
                'crop_info': crop_info,
                'num_crops': len(track_crops),
                'first_frame': track_df['frame'].min(),
                'last_frame': track_df['frame'].max(),
                'duration_frames': track_df['frame'].max() - track_df['frame'].min() + 1
            }
        
        # Compile results
        results = {
            'video_name': video_dir.name,
            'video_metadata': metadata,
            'tracks': track_features,
            'num_tracks': len(track_features),
            'feature_dim': self.feature_dim,
            'model_name': self.model.__class__.__name__,
            'extraction_date': datetime.now().isoformat()
        }
        
        return results
    
    def save_features(self, features_dict: Dict, output_path: Union[str, Path]):
        """
        Save extracted features to file.
        
        Args:
            features_dict: Dictionary containing features and metadata
            output_path: Path to save features
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        save_dict = features_dict.copy()
        save_dict['tracks'] = {}
        
        for track_id, track_data in features_dict['tracks'].items():
            save_dict['tracks'][str(track_id)] = {
                'features': track_data['features'].tolist(),
                'crop_info': track_data['crop_info'],
                'num_crops': int(track_data['num_crops']),
                'first_frame': int(track_data['first_frame']),
                'last_frame': int(track_data['last_frame']),
                'duration_frames': int(track_data['duration_frames'])
            }
        
        # Save as JSON
        if str(output_path).endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(save_dict, f, indent=2)
        
        # Save as numpy archive (more efficient)
        elif str(output_path).endswith('.npz'):
            # Create arrays dict for npz
            arrays_dict = {}
            metadata_dict = {
                'video_name': features_dict['video_name'],
                'video_metadata': json.dumps(features_dict['video_metadata']),
                'num_tracks': int(features_dict['num_tracks']),
                'feature_dim': int(features_dict['feature_dim']),
                'model_name': features_dict['model_name'],
                'extraction_date': features_dict['extraction_date']
            }
            
            # Store features as separate arrays
            for track_id, track_data in features_dict['tracks'].items():
                arrays_dict[f'track_{track_id}_features'] = track_data['features']
                arrays_dict[f'track_{track_id}_metadata'] = json.dumps({
                    'crop_info': track_data['crop_info'],
                    'num_crops': int(track_data['num_crops']),
                    'first_frame': int(track_data['first_frame']),
                    'last_frame': int(track_data['last_frame']),
                    'duration_frames': int(track_data['duration_frames'])
                })
            
            # Save everything
            np.savez_compressed(output_path, 
                               metadata=json.dumps(metadata_dict),
                               **arrays_dict)
        else:
            raise ValueError("Output path must end with .json or .npz")
    
    def load_features(self, features_path: Union[str, Path]) -> Dict:
        """
        Load previously extracted features.
        
        Args:
            features_path: Path to saved features file
        
        Returns:
            Dictionary containing features and metadata
        """
        features_path = Path(features_path)
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        if str(features_path).endswith('.json'):
            with open(features_path, 'r') as f:
                loaded = json.load(f)
            
            # Convert lists back to numpy arrays
            for track_id in loaded['tracks']:
                loaded['tracks'][track_id]['features'] = np.array(
                    loaded['tracks'][track_id]['features']
                )
            
            return loaded
        
        elif str(features_path).endswith('.npz'):
            data = np.load(features_path, allow_pickle=True)
            
            # Parse metadata
            metadata = json.loads(str(data['metadata']))
            
            # Reconstruct tracks dictionary
            tracks = {}
            track_ids = set()
            
            # Find all track IDs
            for key in data.files:
                if key.startswith('track_') and key.endswith('_features'):
                    track_id = int(key.split('_')[1])
                    track_ids.add(track_id)
            
            # Load each track
            for track_id in track_ids:
                track_metadata = json.loads(
                    str(data[f'track_{track_id}_metadata'])
                )
                tracks[track_id] = {
                    'features': data[f'track_{track_id}_features'],
                    **track_metadata
                }
            
            return {
                'video_name': metadata['video_name'],
                'video_metadata': json.loads(metadata['video_metadata']),
                'tracks': tracks,
                'num_tracks': metadata['num_tracks'],
                'feature_dim': metadata['feature_dim'],
                'model_name': metadata['model_name'],
                'extraction_date': metadata['extraction_date']
            }
        
        else:
            raise ValueError("Features file must be .json or .npz")
    
    def process_batch(self, output_base_dir: Union[str, Path], 
                     features_subdir: str = 'reid_features',
                     max_crops_per_track: Optional[int] = None,
                     skip_existing: bool = True):
        """
        Process all video directories in a batch output directory.
        
        Args:
            output_base_dir: Base directory containing video subdirectories
            features_subdir: Subdirectory name for storing features
            max_crops_per_track: Maximum crops per track
            skip_existing: Skip if features already exist
        """
        output_base_dir = Path(output_base_dir)
        
        # Find all video directories
        video_dirs = [d for d in output_base_dir.iterdir() 
                      if d.is_dir() and (d / 'track_data').exists()]
        
        if not video_dirs:
            print(f"No video directories with track_data found in {output_base_dir}")
            return
        
        print(f"Found {len(video_dirs)} video directories to process")
        
        # Process each video
        for video_dir in tqdm(video_dirs, desc="Processing videos"):
            features_dir = video_dir / features_subdir
            features_path = features_dir / 'features.npz'
            
            # Skip if exists
            if skip_existing and features_path.exists():
                print(f"Skipping {video_dir.name} - features already exist")
                continue
            
            try:
                print(f"\nProcessing {video_dir.name}")
                
                # Extract features
                features = self.process_video_directory(
                    video_dir, 
                    max_crops_per_track=max_crops_per_track
                )
                
                # Save features
                features_dir.mkdir(exist_ok=True)
                self.save_features(features, features_path)
                
                print(f"Saved features for {features['num_tracks']} tracks to {features_path}")
                
            except Exception as e:
                print(f"Error processing {video_dir.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract RE-ID features from tracking results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing video tracking results')
    parser.add_argument('--model', type=str, default='osnet_x1_0',
                        help='RE-ID model name (default: osnet_x1_0)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to custom model weights')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--max-crops', type=int, default=None,
                        help='Maximum crops per track (default: use all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Process even if features already exist')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ReIDFeatureExtractor(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Process batch
    extractor.process_batch(
        output_base_dir=args.input_dir,
        max_crops_per_track=args.max_crops,
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    # # Example direct usage
    # extractor = ReIDFeatureExtractor(model_name='osnet_x1_0')
    
    # # Process a batch of videos
    # extractor.process_batch(
    #     output_base_dir="/mnt/c/Users/felix/Downloads/mclarens_res_full",
    #     # max_crops_per_track=50,  # Use max 50 crops per track
    #     skip_existing=False
    # )
    
    # Or use command line
    main()