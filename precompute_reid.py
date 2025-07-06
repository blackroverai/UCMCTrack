#!/usr/bin/env python3
"""
Optimized RE-ID Feature Extractor for UCMC Tracking Results
Based on working torchreid code with osnet_ain_x1_0
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from torchreid.reid.utils import FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReIDFeatureExtractor:
    """Optimized RE-ID feature extractor with batching and parallel I/O."""
    
    def __init__(self, model_name='osnet_ain_x1_0', batch_size=64, num_workers=4):
        """Initialize RE-ID feature extractor."""
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Use torchreid's FeatureExtractor directly
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=str(self.device)
        )
        
        # Enable cudnn benchmark for better performance
        torch.backends.cudnn.benchmark = True
        
        print(f"Loaded {model_name} on {self.device}")
    
    def load_image_batch(self, image_paths):
        """Load multiple images in parallel."""
        def load_single_image(path):
            img = cv2.imread(str(path))
            return img if img is not None else None
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(load_single_image, image_paths))
        
        return images
    
    def extract_features(self, crops):
        """Extract features from a list of crop images with optimizations."""
        if not crops:
            return np.array([])
        
        # Process in batches
        all_features = []
        
        with torch.no_grad():  # Disable gradient computation
            for i in range(0, len(crops), self.batch_size):
                batch = crops[i:i + self.batch_size]
                features = self.extractor(batch)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features) if all_features else np.array([])
    
    def process_video_directory(self, video_dir):
        """Process a single video directory with optimized batching."""
        video_dir = Path(video_dir)
        track_data_dir = video_dir / "track_data"
        crops_dir = track_data_dir / "crops"
        csv_path = track_data_dir / "tracking_results.csv"
        
        # Load tracking results
        df = pd.read_csv(csv_path)
        
        # Group by track_id
        track_groups = df.groupby('track_id')
        
        # Collect all image paths first
        all_crop_paths = []
        crop_to_track_mapping = []
        track_info_dict = {}
        
        print(f"Collecting crop paths for {len(track_groups)} tracks...")
        
        for track_id, track_df in track_groups:
            track_df_sorted = track_df.sort_values('frame')
            track_crops_info = []
            
            for _, row in track_df_sorted.iterrows():
                crop_filename = f"f{int(row['frame'])}_t{int(track_id)}.jpg"
                crop_path = crops_dir / crop_filename
                
                if crop_path.exists():
                    all_crop_paths.append(crop_path)
                    crop_to_track_mapping.append((track_id, len(track_crops_info)))
                    track_crops_info.append({
                        'frame': int(row['frame']),
                        'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                        'conf': row['conf'],
                        'class': row['class']
                    })
            
            if track_crops_info:
                track_info_dict[track_id] = {
                    'crop_info': track_crops_info,
                    'first_frame': int(track_df['frame'].min()),
                    'last_frame': int(track_df['frame'].max())
                }
        
        # Load all images in batches with parallel I/O
        print(f"Loading {len(all_crop_paths)} images...")
        all_images = []
        
        for i in tqdm(range(0, len(all_crop_paths), self.batch_size * 2)):
            batch_paths = all_crop_paths[i:i + self.batch_size * 2]
            batch_images = self.load_image_batch(batch_paths)
            all_images.extend(batch_images)
        
        # Filter out failed loads
        valid_indices = [i for i, img in enumerate(all_images) if img is not None]
        valid_images = [all_images[i] for i in valid_indices]
        valid_mappings = [crop_to_track_mapping[i] for i in valid_indices]
        
        # Extract features for all images at once
        print(f"Extracting features for {len(valid_images)} images...")
        all_features = self.extract_features(valid_images)
        
        # Organize features by track
        track_features = {}
        
        for track_id in track_info_dict:
            track_indices = [i for i, (tid, _) in enumerate(valid_mappings) if tid == track_id]
            
            if track_indices:
                track_feat = all_features[track_indices]
                track_features[int(track_id)] = {
                    'feat': track_feat,
                    'track': track_info_dict[track_id]['crop_info'],
                    'num_crops': len(track_feat),
                    'first_frame': track_info_dict[track_id]['first_frame'],
                    'last_frame': track_info_dict[track_id]['last_frame']
                }
        
        return {
            'video_name': video_dir.name,
            'tracks': track_features,
            'num_tracks': len(track_features)
        }
    
    def save_features(self, features_dict, output_path):
        """Save extracted features to npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_dict = {
            'video_name': features_dict['video_name'],
            'num_tracks': features_dict['num_tracks']
        }
        
        # Save each track's features
        for track_id, track_data in features_dict['tracks'].items():
            save_dict[f'track_{track_id}_feat'] = track_data['feat']
            save_dict[f'track_{track_id}_info'] = json.dumps({
                'track': track_data['track'],
                'num_crops': track_data['num_crops'],
                'first_frame': track_data['first_frame'],
                'last_frame': track_data['last_frame']
            })
        
        np.savez_compressed(output_path, **save_dict)
    
    def load_features(self, features_path):
        """Load previously extracted features."""
        data = np.load(features_path, allow_pickle=True)
        
        # Find all track IDs
        track_ids = set()
        for key in data.files:
            if key.startswith('track_') and key.endswith('_feat'):
                track_id = int(key.split('_')[1])
                track_ids.add(track_id)
        
        # Load tracks
        tracks = {}
        for track_id in track_ids:
            track_info = json.loads(str(data[f'track_{track_id}_info']))
            tracks[track_id] = {
                'feat': data[f'track_{track_id}_feat'],
                'track': track_info['track'],
                'num_crops': track_info['num_crops'],
                'first_frame': track_info['first_frame'],
                'last_frame': track_info['last_frame']
            }
        
        return {
            'video_name': str(data['video_name']),
            'tracks': tracks,
            'num_tracks': int(data['num_tracks'])
        }
    
    def process_batch(self, output_base_dir, features_subdir='reid_features', skip_existing=True):
        """Process all video directories in a batch output directory."""
        output_base_dir = Path(output_base_dir)
        
        # Find all video directories
        video_dirs = [d for d in output_base_dir.iterdir() 
                      if d.is_dir() and (d / 'track_data').exists()]
        
        print(f"Found {len(video_dirs)} video directories to process")
        
        # Process each video
        for video_dir in video_dirs:
            features_path = video_dir / features_subdir / 'features.npz'
            
            # Skip if exists
            if skip_existing and features_path.exists():
                print(f"Skipping {video_dir.name} - features already exist")
                continue
            
            try:
                # Extract features
                features = self.process_video_directory(video_dir)
                
                # Save features
                self.save_features(features, features_path)
                
                print(f"Saved features for {features['num_tracks']} tracks in {video_dir.name}")
                
            except Exception as e:
                print(f"Error processing {video_dir.name}: {str(e)}")
                continue


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract RE-ID features from tracking results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing video tracking results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for feature extraction (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for parallel image loading (default: 4)')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Process even if features already exist')
    
    args = parser.parse_args()
    
    # Initialize extractor with optimized settings
    extractor = ReIDFeatureExtractor(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Process batch
    extractor.process_batch(
        output_base_dir=args.input_dir,
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    main()
