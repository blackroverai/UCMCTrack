#!/usr/bin/env python3
"""
RE-ID Matcher for finding tracks across different video clips
Uses pre-computed features from ReIDFeatureExtractor
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from pathlib import Path
import json
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


@dataclass
class MatchResult:
    """Result of a RE-ID match query"""
    query_clip: str
    query_track: int
    gallery_clip: str
    matched_track: Optional[int]
    similarity_score: float
    confidence: float
    all_scores: Dict[int, float]  # track_id -> similarity score
    comparison_mode: str


class ReIDMatcher:
    """
    Matches tracks across video clips using pre-computed RE-ID features.
    """
    
    def __init__(self, base_dir: Union[str, Path], features_subdir: str = 'reid_features'):
        """
        Initialize RE-ID matcher.
        
        Args:
            base_dir: Base directory containing video subdirectories with features
            features_subdir: Subdirectory name where features are stored
        """
        self.base_dir = Path(base_dir)
        self.features_subdir = features_subdir
        self.loaded_features = {}  # Cache for loaded features
        
    def load_clip_features(self, clip_name: str) -> Dict:
        """
        Load pre-computed features for a video clip.
        
        Args:
            clip_name: Name of the video clip
            
        Returns:
            Dictionary containing features and metadata
        """
        if clip_name in self.loaded_features:
            return self.loaded_features[clip_name]
        
        # Find the features file
        clip_dir = self.base_dir / clip_name
        features_path = clip_dir / self.features_subdir / 'features.npz'
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found for clip: {clip_name} at {features_path}")
        
        # Load features
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
            track_metadata = json.loads(str(data[f'track_{track_id}_metadata']))
            tracks[track_id] = {
                'features': data[f'track_{track_id}_features'],
                **track_metadata
            }
        
        features_dict = {
            'video_name': metadata['video_name'],
            'tracks': tracks,
            'num_tracks': metadata['num_tracks'],
            'feature_dim': metadata['feature_dim'],
            'model_name': metadata['model_name']
        }
        
        # Cache the loaded features
        self.loaded_features[clip_name] = features_dict
        
        return features_dict
    
    def compute_similarity_matrix(self, 
                                features1: np.ndarray, 
                                features2: np.ndarray,
                                metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity matrix between two sets of features.
        
        Args:
            features1: Features from track 1, shape (M, D)
            features2: Features from track 2, shape (N, D)
            metric: Distance metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity matrix of shape (M, N)
        """
        if metric == 'cosine':
            # Cosine similarity (higher is better)
            # Features should already be L2 normalized
            similarity = np.dot(features1, features2.T)
            return similarity
        
        elif metric == 'euclidean':
            # Convert euclidean distance to similarity (lower distance = higher similarity)
            distances = cdist(features1, features2, metric='euclidean')
            # Convert to similarity: sim = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distances)
            return similarity
        
        else:
            raise ValueError(f"Unknown metric: {mfetric}")
    
    def aggregate_similarity(self,
                           similarity_matrix: np.ndarray,
                           mode: Literal['global_mean', 'global_max', 'mean_of_row_max', 'mean_of_col_max']) -> float:
        """
        Aggregate similarity matrix into a single score.
        
        Args:
            similarity_matrix: MxN similarity matrix
            mode: Aggregation mode
                - 'global_mean': Average of all similarities
                - 'global_max': Maximum similarity across all pairs
                - 'mean_of_row_max': Average of maximum similarity for each query crop
                - 'mean_of_col_max': Average of maximum similarity for each gallery crop
                
        Returns:
            Aggregated similarity score
        """
        if similarity_matrix.size == 0:
            return 0.0
        
        if mode == 'global_mean':
            return np.mean(similarity_matrix)
        
        elif mode == 'global_max':
            return np.max(similarity_matrix)
        
        elif mode == 'mean_of_row_max':
            # For each query crop, find best match in gallery
            row_maxes = np.max(similarity_matrix, axis=1)
            return np.mean(row_maxes)
        
        elif mode == 'mean_of_col_max':
            # For each gallery crop, find best match in query
            col_maxes = np.max(similarity_matrix, axis=0)
            return np.mean(col_maxes)
        
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
    
    def match_track(self,
                   query_clip: str,
                   query_track_id: int,
                   gallery_clip: str,
                   comparison_mode: Literal['global_mean', 'global_max', 'mean_of_row_max', 'mean_of_col_max'] = 'mean_of_row_max',
                   similarity_threshold: float = 0.5,
                   metric: str = 'cosine') -> MatchResult:
        """
        Find matching track in gallery clip for a query track.
        
        Args:
            query_clip: Name of query video clip
            query_track_id: Track ID in query clip
            gallery_clip: Name of gallery video clip
            comparison_mode: How to aggregate crop-wise similarities
            similarity_threshold: Minimum similarity to consider a match
            metric: Distance metric to use
            
        Returns:
            MatchResult with match information
        """
        # Load features
        query_features = self.load_clip_features(query_clip)
        gallery_features = self.load_clip_features(gallery_clip)
        
        # Get query track features
        if query_track_id not in query_features['tracks']:
            raise ValueError(f"Track {query_track_id} not found in query clip {query_clip}")
        
        query_track_features = query_features['tracks'][query_track_id]['features']
        
        # Compare against all tracks in gallery
        all_scores = {}
        best_score = -1
        best_track = None
        
        for gallery_track_id, gallery_track_data in gallery_features['tracks'].items():
            gallery_track_features = gallery_track_data['features']
            
            # Compute similarity matrix
            sim_matrix = self.compute_similarity_matrix(
                query_track_features,
                gallery_track_features,
                metric=metric
            )
            
            # Aggregate to single score
            score = self.aggregate_similarity(sim_matrix, mode=comparison_mode)
            all_scores[gallery_track_id] = score
            
            if score > best_score:
                best_score = score
                best_track = gallery_track_id
        
        # Determine if it's a match
        matched_track = best_track if best_score >= similarity_threshold else None
        
        # Calculate confidence (relative to second best)
        sorted_scores = sorted(all_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            confidence = 1.0 if matched_track is not None else 0.0
        
        return MatchResult(
            query_clip=query_clip,
            query_track=query_track_id,
            gallery_clip=gallery_clip,
            matched_track=matched_track,
            similarity_score=best_score,
            confidence=confidence,
            all_scores=all_scores,
            comparison_mode=comparison_mode
        )
    
    def match_all_tracks(self,
                        query_clip: str,
                        gallery_clip: str,
                        comparison_mode: Literal['global_mean', 'global_max', 'mean_of_row_max', 'mean_of_col_max'] = 'mean_of_row_max',
                        similarity_threshold: float = 0.5,
                        metric: str = 'cosine') -> List[MatchResult]:
        """
        Match all tracks from query clip to gallery clip.
        
        Args:
            query_clip: Name of query video clip
            gallery_clip: Name of gallery video clip
            comparison_mode: How to aggregate crop-wise similarities
            similarity_threshold: Minimum similarity to consider a match
            metric: Distance metric to use
            
        Returns:
            List of MatchResults for each query track
        """
        query_features = self.load_clip_features(query_clip)
        results = []
        
        for query_track_id in query_features['tracks'].keys():
            result = self.match_track(
                query_clip=query_clip,
                query_track_id=query_track_id,
                gallery_clip=gallery_clip,
                comparison_mode=comparison_mode,
                similarity_threshold=similarity_threshold,
                metric=metric
            )
            results.append(result)
        
        return results
    
    def find_track_across_clips(self,
                               query_clip: str,
                               query_track_id: int,
                               gallery_clips: Optional[List[str]] = None,
                               comparison_mode: Literal['global_mean', 'global_max', 'mean_of_row_max', 'mean_of_col_max'] = 'mean_of_row_max',
                               similarity_threshold: float = 0.5,
                               metric: str = 'cosine') -> List[MatchResult]:
        """
        Find a track across multiple gallery clips.
        
        Args:
            query_clip: Name of query video clip
            query_track_id: Track ID in query clip
            gallery_clips: List of gallery clips to search (None = search all)
            comparison_mode: How to aggregate crop-wise similarities
            similarity_threshold: Minimum similarity to consider a match
            metric: Distance metric to use
            
        Returns:
            List of MatchResults, one per gallery clip
        """
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
                continue  # Skip self-comparison
                
            try:
                result = self.match_track(
                    query_clip=query_clip,
                    query_track_id=query_track_id,
                    gallery_clip=gallery_clip,
                    comparison_mode=comparison_mode,
                    similarity_threshold=similarity_threshold,
                    metric=metric
                )
                results.append(result)
            except Exception as e:
                print(f"Error matching with {gallery_clip}: {str(e)}")
                continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def print_match_result(self, result: MatchResult, detailed: bool = False):
        """Pretty print a match result."""
        print(f"\nQuery: {result.query_clip} - Track {result.query_track}")
        print(f"Gallery: {result.gallery_clip}")
        
        if result.matched_track is not None:
            print(f"✓ MATCH FOUND: Track {result.matched_track}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
        else:
            print(f"✗ NO MATCH (best similarity: {result.similarity_score:.3f})")
        
        if detailed:
            print(f"\nAll scores ({result.comparison_mode}):")
            sorted_tracks = sorted(result.all_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
            for track_id, score in sorted_tracks[:5]:  # Top 5
                marker = " *" if track_id == result.matched_track else ""
                print(f"  Track {track_id}: {score:.3f}{marker}")


def main():
    """Example usage and command line interface"""
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
    parser.add_argument('--mode', type=str, default='mean_of_row_max',
                        choices=['global_mean', 'global_max', 'mean_of_row_max', 'mean_of_col_max'],
                        help='Comparison mode')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for matching')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Distance metric')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed results')
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = ReIDMatcher(args.base_dir)
    
    if args.gallery_clip:
        # Match against specific gallery
        result = matcher.match_track(
            query_clip=args.query_clip,
            query_track_id=args.query_track,
            gallery_clip=args.gallery_clip,
            comparison_mode=args.mode,
            similarity_threshold=args.threshold,
            metric=args.metric
        )
        matcher.print_match_result(result, detailed=args.detailed)
    else:
        # Search across all clips
        results = matcher.find_track_across_clips(
            query_clip=args.query_clip,
            query_track_id=args.query_track,
            comparison_mode=args.mode,
            similarity_threshold=args.threshold,
            metric=args.metric
        )
        
        print(f"\nSearching for {args.query_clip} - Track {args.query_track} across {len(results)} clips:")
        print("="*60)
        
        for result in results:
            matcher.print_match_result(result, detailed=args.detailed)


if __name__ == "__main__":
    # Use command line interface
    main()

    # # Example direct usage
    # base_dir = "/mnt/c/Users/felix/Downloads/mclarens_res_full"
    
    # # Initialize matcher
    # matcher = ReIDMatcher(base_dir)
    
    # # Example 1: Match specific track between two clips
    # result = matcher.match_track(
    #     query_clip="mclarens-suburban-sunset-chase-6",
    #     query_track_id=656,  # Use actual track ID from your data
    #     gallery_clip="mclarens-suburban-sunset-chase-5",
    #     comparison_mode="mean_of_row_max"
    # )
    # matcher.print_match_result(result, detailed=True)
    
    # # Example 2: Find track across all clips
    # results = matcher.find_track_across_clips(
    #     query_clip="mclarens-suburban-sunset-chase-6",
    #     query_track_id=656,
    #     comparison_mode="mean_of_row_max"
    # )
    
    # print(f"\nTop matches across all clips:")
    # for result in results[:3]:  # Top 3
    #     matcher.print_match_result(result)
    
    # # Or use command line
    # # main()