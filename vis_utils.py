import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

def load_track_crops(base_dir, clip_name, track_id, max_crops=4):
    """Load sample crops for a track"""
    crops_dir = Path(base_dir) / clip_name / "track_data" / "crops"
    csv_path = Path(base_dir) / clip_name / "track_data" / "tracking_results.csv"
    
    # Load tracking results
    df = pd.read_csv(csv_path)
    track_df = df[df['track_id'] == track_id].sort_values('frame')
    
    if len(track_df) == 0:
        return []
    
    # Sample frames evenly
    total_frames = len(track_df)
    if total_frames <= max_crops:
        sample_indices = list(range(total_frames))
    else:
        sample_indices = np.linspace(0, total_frames-1, max_crops, dtype=int)
    
    crops = []
    for idx in sample_indices:
        row = track_df.iloc[idx]
        crop_filename = f"f{int(row['frame'])}_t{int(track_id)}.jpg"
        crop_path = crops_dir / crop_filename
        
        if crop_path.exists():
            img = cv2.imread(str(crop_path))
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                crops.append(img)
    
    return crops

def show_results_with_top_k(results, BASE_DIR, NUM_CROPS_TO_SHOW=3, top_k=1, max_clips=50):
    """
    Display results with option to show top-k tracks for each clip.
    
    Parameters:
    - results: List of result dictionaries
    - BASE_DIR: Base directory for loading crops
    - NUM_CROPS_TO_SHOW: Number of crops to display per track
    - top_k: Number of top tracks to show per clip (default=1)
    - max_clips: Maximum number of clips to display (default=50)
    """
    num_matches_to_show = min(max_clips, len(results))
    
    for i in range(num_matches_to_show):
        result = results[i]
        
        print(f"\n{'='*60}")
        print(f"Clip {i+1}: {result['gallery_clip']}")
        print(f"{'='*60}")
        
        # Sort tracks by distance (ascending)
        sorted_tracks = sorted(result['all_scores'].items(), key=lambda x: x[1])
        
        # Determine how many tracks to show
        num_tracks_to_show = min(top_k, len(sorted_tracks))
        
        for track_idx in range(num_tracks_to_show):
            track_name, distance = sorted_tracks[track_idx]
            
            # Check if this is the matched track
            is_match = (track_name == result['matched_track'])
            
            print(f"\n  Top-{track_idx+1}: Track {track_name}")
            print(f"  Distance: {distance:.3f}")
            print(f"  Status: {'MATCHED' if is_match else 'NOT MATCHED'}")
            
            # Load and display crops
            gallery_crops = load_track_crops(BASE_DIR, result['gallery_clip'], track_name, NUM_CROPS_TO_SHOW)
            
            if gallery_crops:
                fig, axes = plt.subplots(1, len(gallery_crops), figsize=(3*len(gallery_crops), 3))
                if len(gallery_crops) == 1:
                    axes = [axes]
                
                for idx, crop in enumerate(gallery_crops):
                    axes[idx].imshow(crop)
                    axes[idx].set_title(f"Frame {idx+1}")
                    axes[idx].axis('off')
                
                # Color code based on match status
                title_color = 'green' if is_match else 'orange' if track_idx == 0 else 'gray'
                plt.suptitle(
                    f"{result['gallery_clip']} - Track {track_name} (Top-{track_idx+1}, Distance: {distance:.3f})", 
                    fontsize=12, 
                    color=title_color
                )
                plt.tight_layout()
                plt.show()
            else:
                print(f"  (No crops available for this track)")


# Alternative version with more compact display
def show_results_grid_view(results, BASE_DIR, NUM_CROPS_TO_SHOW=3, top_k=3, max_clips=20):
    """
    Display results in a grid view showing multiple tracks per clip.
    
    Parameters:
    - results: List of result dictionaries
    - BASE_DIR: Base directory for loading crops
    - NUM_CROPS_TO_SHOW: Number of crops to display per track
    - top_k: Number of top tracks to show per clip (default=3)
    - max_clips: Maximum number of clips to display (default=20)
    """
    num_matches_to_show = min(max_clips, len(results))
    
    for i in range(num_matches_to_show):
        result = results[i]
        
        # Sort tracks by distance (ascending)
        sorted_tracks = sorted(result['all_scores'].items(), key=lambda x: x[1])
        num_tracks_to_show = min(top_k, len(sorted_tracks))
        
        # Create subplot grid for all tracks
        fig, axes = plt.subplots(num_tracks_to_show, NUM_CROPS_TO_SHOW, 
                                figsize=(3*NUM_CROPS_TO_SHOW, 3*num_tracks_to_show))
        
        # Handle single track case
        if num_tracks_to_show == 1:
            axes = axes.reshape(1, -1)
        # Handle single crop case
        if NUM_CROPS_TO_SHOW == 1:
            axes = axes.reshape(-1, 1)
        
        for track_idx in range(num_tracks_to_show):
            track_name, distance = sorted_tracks[track_idx]
            is_match = (track_name == result['matched_track'])
            
            # Load crops for this track
            gallery_crops = load_track_crops(BASE_DIR, result['gallery_clip'], track_name, NUM_CROPS_TO_SHOW)
            
            # Display crops in row
            for crop_idx in range(NUM_CROPS_TO_SHOW):
                ax = axes[track_idx, crop_idx]
                
                if gallery_crops and crop_idx < len(gallery_crops):
                    ax.imshow(gallery_crops[crop_idx])
                    if crop_idx == 0:  # Add track info on first image
                        ax.set_ylabel(f"Track {track_name}\nDist: {distance:.3f}", 
                                     fontsize=10, rotation=0, ha='right', va='center')
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center', va='center')
                
                ax.axis('off')
                
                # Add border color based on match status
                if is_match:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
        
        plt.suptitle(f"Clip {i+1}: {result['gallery_clip']} - Top {num_tracks_to_show} Tracks", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Example usage:
# Show top 3 tracks for each clip
# show_results_with_top_k(results, BASE_DIR, NUM_CROPS_TO_SHOW=3, top_k=3, max_clips=20)

# Or use the grid view for more compact display
# show_results_grid_view(results, BASE_DIR, NUM_CROPS_TO_SHOW=3, top_k=3, max_clips=10)