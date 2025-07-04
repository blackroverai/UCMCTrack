#!/usr/bin/env python3
"""
Simplified RE-ID Matcher based on working torchreid code
"""

import torch
import numpy as np
from pathlib import Path
import json
from torchreid.reid.metrics.distance import compute_distance_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReIDMatcher:
    """Matches tracks across video clips using pre-computed RE-ID features."""
    
    def __init__(self, base_dir, features_subdir='reid_features'):
        """Initialize RE-ID matcher."""
        self.base_dir = Path(base_dir)
        self.features_subdir = features_subdir
        self.loaded_features = {}  # Cache
    
    def load_clip_features(self, clip_name):
        """Load pre-computed features for a video clip."""
        if clip_name in self.loaded_features:
            return self.loaded_features[clip_name]
        
        # Find the features file
        features_path = self.base_dir / clip_name / self.features_subdir / 'features.npz'
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found for clip: {clip_name}")
        
        # Load features
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
                'feat': torch.from_numpy(data[f'track_{track_id}_feat']).to(device),
                'track': track_info['track'],
                'num_crops': track_info['num_crops']
            }
        
        features_dict = {
            'video_name': str(data['video_name']),
            'tracks': tracks,
            'num_tracks': int(data['num_tracks'])
        }
        
        # Cache
        self.loaded_features[clip_name] = features_dict
        
        return features_dict
    
    def compare_tracks(self, track1_feat, track2_feat, mode='mean'):
        """Compare two tracks using their features."""
        # Use torchreid's compute_distance_matrix
        mat = compute_distance_matrix(track1_feat, track2_feat, 'cosine')
        
        if mode == 'mean':
           return float(torch.mean(mat))
        elif mode == 'median':
            return float(torch.median(mat))
        elif mode == 'min':
            return float(torch.min(mat))
        elif mode == 'median_of_row_mins':
            row_mins = mat.min(dim=1)[0]
            return float(torch.median(row_mins))
        else:
            return float(torch.mean(mat))
    
    def match_track(self, query_clip, query_track_id, gallery_clip, mode='mean', threshold=0.5):
        """Find matching track in gallery clip for a query track."""
        # Load features
        query_features = self.load_clip_features(query_clip)
        gallery_features = self.load_clip_features(gallery_clip)
        
        # Get query track features
        if query_track_id not in query_features['tracks']:
            raise ValueError(f"Track {query_track_id} not found in query clip {query_clip}")
        
        query_track = query_features['tracks'][query_track_id]
        
        # Compare against all tracks in gallery
        all_scores = {}
        best_score = float('inf')  # Lower is better for distance
        best_track = None
        
        for gallery_track_id, gallery_track in gallery_features['tracks'].items():
            # Compare tracks
            score = self.compare_tracks(
                query_track['feat'],
                gallery_track['feat'],
                mode=mode
            )
            all_scores[gallery_track_id] = score
            
            if score < best_score:
                best_score = score
                best_track = gallery_track_id
        
        # Determine if it's a match
        matched_track = best_track if best_score < threshold else None
        
        return {
            'query_clip': query_clip,
            'query_track': query_track_id,
            'gallery_clip': gallery_clip,
            'matched_track': matched_track,
            'distance': best_score,
            'all_scores': all_scores,
            'mode': mode
        }
    
    def find_track_across_clips(self, query_clip, query_track_id, gallery_clips=None, mode='mean', threshold=0.5):
        """Find a track across multiple gallery clips."""
        # Get all available clips if not specified
        if gallery_clips is None:
            gallery_clips = []
            for clip_dir in self.base_dir.iterdir():
                if clip_dir.is_dir() and clip_dir.name != query_clip:
                    features_path = clip_dir / self.features_subdir / 'features.npz'
                    if features_path.exists():
                        gallery_clips.append(clip_dir.name)
        
        results = []
        for gallery_clip in gallery_clips:
            if gallery_clip == query_clip:
                continue
                
            try:
                result = self.match_track(
                    query_clip=query_clip,
                    query_track_id=query_track_id,
                    gallery_clip=gallery_clip,
                    mode=mode,
                    threshold=threshold
                )
                results.append(result)
            except Exception as e:
                print(f"Error matching with {gallery_clip}: {str(e)}")
                continue
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x['distance'])
        
        return results
    
    def compare_two_track_lists(self, clips1, clips2, mode='mean'):
        """Compare all tracks between two lists of clips."""
        # Load all features
        tracks1 = []
        tracks2 = []
        
        for clip in clips1:
            features = self.load_clip_features(clip)
            for track_id, track_data in features['tracks'].items():
                tracks1.append({
                    'clip': clip,
                    'track_id': track_id,
                    'feat': track_data['feat']
                })
        
        for clip in clips2:
            features = self.load_clip_features(clip)
            for track_id, track_data in features['tracks'].items():
                tracks2.append({
                    'clip': clip,
                    'track_id': track_id,
                    'feat': track_data['feat']
                })
        
        # Build distance matrix
        n1 = len(tracks1)
        n2 = len(tracks2)
        dist_matrix = np.zeros((n1, n2), dtype=float)
        
        for i in range(n1):
            for j in range(n2):
                dist = self.compare_tracks(
                    tracks1[i]['feat'],
                    tracks2[j]['feat'],
                    mode=mode
                )
                dist_matrix[i, j] = dist
            if i % 10 == 0:
                print(f"Compared {i}/{n1} tracks")
        
        return dist_matrix, tracks1, tracks2


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Match tracks across video clips using RE-ID')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing video results')
    parser.add_argument('--query-clip', type=str, required=True,
                        help='Query video clip name')
    parser.add_argument('--query-track', type=int, required=True,
                        help='Query track ID')
    parser.add_argument('--gallery-clip', type=str,
                        help='Gallery video clip name (if not specified, search all)')
    parser.add_argument('--mode', type=str, default='mean',
                        choices=['mean', 'median'],
                        help='Comparison mode')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Distance threshold for matching (lower is better)')
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = ReIDMatcher(args.base_dir)
    
    if args.gallery_clip:
        # Match against specific gallery
        result = matcher.match_track(
            query_clip=args.query_clip,
            query_track_id=args.query_track,
            gallery_clip=args.gallery_clip,
            mode=args.mode,
            threshold=args.threshold
        )
        
        print(f"\nQuery: {result['query_clip']} - Track {result['query_track']}")
        print(f"Gallery: {result['gallery_clip']}")
        
        if result['matched_track'] is not None:
            print(f"✓ MATCH FOUND: Track {result['matched_track']}")
            print(f"  Distance: {result['distance']:.3f}")
        else:
            print(f"✗ NO MATCH (best distance: {result['distance']:.3f})")
        
        # Show top 5 scores
        print(f"\nTop 5 tracks (mode={result['mode']}):")
        sorted_tracks = sorted(result['all_scores'].items(), key=lambda x: x[1])
        for track_id, score in sorted_tracks[:5]:
            marker = " *" if track_id == result['matched_track'] else ""
            print(f"  Track {track_id}: {score:.3f}{marker}")
    
    else:
        # Search across all clips
        results = matcher.find_track_across_clips(
            query_clip=args.query_clip,
            query_track_id=args.query_track,
            mode=args.mode,
            threshold=args.threshold
        )
        
        print(f"\nSearching for {args.query_clip} - Track {args.query_track} across {len(results)} clips:")
        print("="*60)
        
        for result in results[:5]:  # Top 5
            print(f"\n{result['gallery_clip']}:")
            if result['matched_track'] is not None:
                print(f"  ✓ Match: Track {result['matched_track']} (distance: {result['distance']:.3f})")
            else:
                print(f"  ✗ No match (best distance: {result['distance']:.3f})")


if __name__ == "__main__":
    main()