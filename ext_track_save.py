#!/usr/bin/env python3
"""
Modular frame-by-frame tracker for UCMCTrack using live YOLO12 + RT-DETR detection.
Extended for batch processing of video folders with organized output structure.
"""

from typing import Sequence, Callable, List
from functools import cached_property
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR
from torchvision.ops import nms, box_iou
import pandas as pd
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime


from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper


def merge_frame_results(res_list, iou_thresh=0.5):
    template = res_list[0]
    all_boxes = torch.cat([r.boxes.data for r in res_list], dim=0)
    if all_boxes.numel() == 0:
        return template.new()
    keep = nms(all_boxes[:, :4], all_boxes[:, 4], iou_thresh)
    merged_boxes = all_boxes[keep]
    new = template.new()
    new.update(boxes=merged_boxes)
    return new

def fuse_union(res, iou_thr=0.5, ioa_thr=0.7):
    # Return empty result if no boxes
    if res.boxes is None or res.boxes.data.numel() == 0:
        return res.new()
    
    data = res.boxes.data
    coords = data[:, :4]
    scores = data[:, 4]
    labels = data[:, 5].long()
    N = len(scores)
    
    # Handle single detection case
    if N == 1:
        out = res.new()
        out.update(boxes=data)
        return out
    
    iou_mat = box_iou(coords, coords)
    x1 = torch.max(coords[:, 0].view(-1, 1), coords[:, 0])
    y1 = torch.max(coords[:, 1].view(-1, 1), coords[:, 1])
    x2 = torch.min(coords[:, 2].view(-1, 1), coords[:, 2])
    y2 = torch.min(coords[:, 3].view(-1, 1), coords[:, 3])
    inter_area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0))
    area = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    ioa_mat = inter_area / (torch.min(area.view(-1,1), area.view(1,-1)) + 1e-6)
    conn = (iou_mat > iou_thr) | (ioa_mat > ioa_thr)
    visited = torch.zeros(N, dtype=torch.bool, device=data.device)
    groups = []
    
    for i in range(N):
        if visited[i]:
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            stack.extend(torch.where(conn[u])[0].tolist())
        groups.append(comp)
    
    fused = []
    for comp in groups:
        idx = torch.tensor(comp, device=data.device)
        x1_min, y1_min = coords[idx, :2].min(0)[0]
        x2_max, y2_max = coords[idx, 2:4].max(0)[0]
        scs = scores[idx]
        best = scs.argmax()
        fused.append(torch.tensor([
            x1_min, y1_min, x2_max, y2_max,
            scs[best], labels[idx][best].float()
        ], device=data.device))
    
    # Handle case where no fused boxes remain
    if not fused:
        return res.new()
        
    fused_data = torch.stack(fused, 0)

    out = res.new()
    if isinstance(data, torch.Tensor):      # convert InferenceTensor ➜ normal Tensor
        data = data.detach().clone()
    out.update(boxes=fused_data)
    return out


class Detection:
    def __init__(self, det_id, bb_left=0, bb_top=0, bb_width=0, bb_height=0,
                 conf=0.0, det_class=0):
        self.id = det_id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return (f"d{self.id}, bb:[{self.bb_left},{self.bb_top},"
                f"{self.bb_width},{self.bb_height}], "
                f"conf={self.conf:.2f}, cls{self.det_class}, "
                f"uv:[{self.bb_left+self.bb_width/2:.0f},"
                f"{self.bb_top+self.bb_height/2:.0f}], "
                f"mapped:[{self.y[0,0]:.1f},{self.y[1,0]:.1f}]")

    __repr__ = __str__

class LiveDetector:
    def __init__(
        self,
        cam_para_file: str,
        models: Sequence[Callable],  # each must expose .predict(frame)[0]
        merge_iou=0.5,
        fuse_iou=0.5,
        fuse_ioa=0.7,
    ):
        if not models:
            raise ValueError("Need at least one model")
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.models = models
        self.merge_iou = merge_iou
        self.fuse_iou = fuse_iou
        self.fuse_ioa = fuse_ioa

    @cached_property
    def _single_model(self):
        """True when nothing to merge/fuse."""
        return len(self.models) == 1

    def _model_preds(self, frame):
        # Runs every model and returns a list of Ultralytics Results objects
        return [m.predict(frame, agnostic_nms=True, iou=0.5, verbose=False)[0]
                for m in self.models]

    def detect_frame(self, frame, frame_id, conf_thresh=0.01):
        preds = self._model_preds(frame)

        if self._single_model:
        # nothing to merge, but still fuse overlapping boxes
            fused = fuse_union(
                preds[0], 
                iou_thr=self.fuse_iou,
                ioa_thr=self.fuse_ioa
            )
        else:
            merged = merge_frame_results(preds, iou_thresh=self.merge_iou)
            fused  = fuse_union(
                merged,
                iou_thr=self.fuse_iou,
                ioa_thr=self.fuse_ioa
            )

        dets, det_id = [], 0

        # Check if fused has boxes and if boxes.data exists
        if fused.boxes is not None and fused.boxes.data.numel() > 0:
            for box in fused.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf < conf_thresh:
                    continue
                w, h = x2 - x1, y2 - y1
                det = Detection(det_id, x1, y1, w, h, conf, int(cls))
                det.y, det.R = self.mapper.mapto([x1, y1, w, h])
                dets.append(det)
                det_id += 1

        return dets


class UCMCTracker:
    def __init__(self, cam_para_path, wx=5.0, wy=5.0, vmax=10.0, a=100.0, cdt=60.0, 
                 fps=30.0, high_score=0.5, conf_thresh=0.01, save_crops=False, crops_dir=None):
        # self.detector = LiveDetector(cam_para_path)
        self.detector = LiveDetector(cam_para_path, [YOLO("yolo12x.pt"), RTDETR("rtdetr-x.pt")])
        # self.detector = LiveDetector(cam_para_path, [RTDETR("rtdetr-x.pt")])
        # self.detector = LiveDetector(cam_para_path, [YOLO("/home/felix/models/detr_visdrone5/weights/best.pt")])
        self.tracker = UCMCTrack(a, a, wx, wy, vmax, cdt, fps, "MOT", high_score, False, None)
        self.conf_thresh = conf_thresh
        self.frame_id = 1
        
        # Crop saving additions
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.tracking_results = []  # Store tracking results for CSV
        
        if self.save_crops and self.crops_dir:
            os.makedirs(self.crops_dir, exist_ok=True)

    def process_frame(self, frame):
        dets = self.detector.detect_frame(frame, self.frame_id, self.conf_thresh)
        self.tracker.update(dets, self.frame_id)
        
        # Save crops and tracking results if enabled
        if self.save_crops and self.crops_dir:
            for det in dets:
                if det.track_id > 0:  # Only save tracked objects
                    # Extract crop
                    x1 = int(det.bb_left)
                    y1 = int(det.bb_top)
                    x2 = int(det.bb_left + det.bb_width)
                    y2 = int(det.bb_top + det.bb_height)
                    
                    # Ensure bounds are valid
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        
                        # Save crop with standardized naming
                        crop_filename = f"f{self.frame_id}_t{det.track_id}.jpg"
                        crop_path = os.path.join(self.crops_dir, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        
                        # Store tracking result
                        self.tracking_results.append({
                            'frame': self.frame_id,
                            'track_id': det.track_id,
                            'x1': det.bb_left,
                            'y1': det.bb_top,
                            'x2': det.bb_left + det.bb_width,
                            'y2': det.bb_top + det.bb_height,
                            'conf': det.conf,
                            'class': det.det_class
                        })
        
        self.frame_id += 1
        return dets
    
    def save_tracking_results(self, csv_path):
        """Save tracking results to CSV file"""
        if self.tracking_results:
            df = pd.DataFrame(self.tracking_results)
            df.to_csv(csv_path, index=False)
            print(f"Saved tracking results to {csv_path}")
    

def run_ucmc_on_video(video_path, cam_para_path, output_path=None, save_dir=None, 
                      create_video=True, copy_original=True):
    """
    Run UCMC tracker on video.
    
    Args:
        video_path: Path to input video
        cam_para_path: Path to camera parameters file
        output_path: Path for output video (optional if create_video=False)
        save_dir: Directory to save crops and tracking CSV (optional)
        create_video: Whether to create output video (default True)
        copy_original: Whether to copy original video to output dir (default True)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        print("Warning: Could not determine FPS. Defaulting to 30.")
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if needed
    writer = None
    if create_video:
        if not output_path:
            raise ValueError("output_path required when create_video=True")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    # Setup save directory
    crops_dir = None
    save_crops = save_dir is not None
    if save_crops:
        os.makedirs(save_dir, exist_ok=True)
        
        # Copy original video to output directory if requested
        if copy_original:
            video_name = os.path.basename(video_path)
            dest_video_path = os.path.join(save_dir, video_name)
            print(f"Copying original video to {dest_video_path}")
            shutil.copy2(video_path, dest_video_path)
        
        # Create track_data directory
        track_data_dir = os.path.join(save_dir, "track_data")
        os.makedirs(track_data_dir, exist_ok=True)
        
        # Create crops directory
        crops_dir = os.path.join(track_data_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

    tracker = UCMCTracker(
        cam_para_path=cam_para_path,
        fps=fps,
        wx=5.0, wy=5.0,
        vmax=10.0,
        a=100.0,
        cdt=120.0,
        high_score=0.5,
        conf_thresh=0.01,
        save_crops=save_crops,
        crops_dir=crops_dir
    )

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = tracker.process_frame(frame)

        # Draw tracked boxes and write video if enabled
        if create_video:
            for det in detections:
                if det.track_id <= 0:
                    continue
                x1 = int(det.bb_left)
                y1 = int(det.bb_top)
                x2 = int(det.bb_left + det.bb_width)
                y2 = int(det.bb_top + det.bb_height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(det.track_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            writer.write(frame)

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    if writer:
        writer.release()
    
    # Save tracking results CSV if crops were saved
    if save_crops:
        csv_path = os.path.join(track_data_dir, "tracking_results.csv")
        tracker.save_tracking_results(csv_path)
        
        # Save metadata
        metadata = {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "total_frames": frame_id,
            "fps": fps,
            "width": width,
            "height": height,
            "total_tracks": len(set(t['track_id'] for t in tracker.tracking_results)),
            "total_detections": len(tracker.tracking_results),
            "processing_date": datetime.now().isoformat()
        }
        metadata_path = os.path.join(track_data_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved crops to {crops_dir}")
        print(f"Saved tracking results to {csv_path}")
        print(f"Saved metadata to {metadata_path}")
    
    if create_video:
        print(f"Done. Output video saved to {output_path}")
    else:
        print("Done. Processing complete.")
    
    return True  # Success indicator


def process_video_folder(input_dir, output_dir, cam_para_path, 
                        video_extensions=('.mp4', '.avi', '.mov', '.mkv', '.ts'),
                        create_videos=False, copy_originals=True, skip_existing=True):
    """
    Process all videos in a folder with UCMC tracking.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Base output directory
        cam_para_path: Path to camera parameters file
        video_extensions: Tuple of video file extensions to process
        create_videos: Whether to create output videos with tracking visualization
        copy_originals: Whether to copy original videos to output directories
        skip_existing: Whether to skip videos that have already been processed
    
    Returns:
        Dictionary with processing results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
    
    if not video_files:
        print(f"No video files found in {input_dir} with extensions {video_extensions}")
        return {"processed": 0, "skipped": 0, "failed": 0, "results": []}
    
    print(f"Found {len(video_files)} video files to process")
    
    results = []
    processed = 0
    skipped = 0
    failed = 0
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.stem  # filename without extension
        video_output_dir = output_path / video_name
        
        # Check if already processed
        if skip_existing and video_output_dir.exists():
            metadata_file = video_output_dir / "track_data" / "metadata.json"
            if metadata_file.exists():
                print(f"Skipping {video_name} - already processed")
                skipped += 1
                continue
        
        print(f"\nProcessing: {video_path.name}")
        
        try:
            # Setup output paths
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_video_path = None
            if create_videos:
                output_video_path = str(video_output_dir / f"{video_name}_tracked.mp4")
            
            # Run tracking
            success = run_ucmc_on_video(
                video_path=str(video_path),
                cam_para_path=cam_para_path,
                output_path=output_video_path,
                save_dir=str(video_output_dir),
                create_video=create_videos,
                copy_original=copy_originals
            )
            
            if success:
                processed += 1
                results.append({
                    "video": video_path.name,
                    "status": "success",
                    "output_dir": str(video_output_dir)
                })
            else:
                failed += 1
                results.append({
                    "video": video_path.name,
                    "status": "failed",
                    "error": "Processing returned False"
                })
                
        except Exception as e:
            failed += 1
            print(f"Error processing {video_path.name}: {str(e)}")
            results.append({
                "video": video_path.name,
                "status": "failed",
                "error": str(e)
            })
    
    # Save overall processing summary
    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "total_videos": len(video_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "processing_date": datetime.now().isoformat(),
        "results": results
    }
    
    summary_path = output_path / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total videos: {len(video_files)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Summary saved to: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='UCMC Tracker - Single video or batch processing')
    
    # Mode selection
    parser.add_argument('mode', choices=['single', 'batch'], 
                        help='Processing mode: single video or batch folder')
    
    # Common arguments
    parser.add_argument('--cam-para', type=str, default='demo/cam_para.txt',
                        help='Path to camera parameters file')
    parser.add_argument('--create-videos', action='store_true',
                        help='Create output videos with tracking visualization')
    parser.add_argument('--no-copy-original', action='store_true',
                        help='Do not copy original videos to output directories')
    
    # Single video mode arguments
    parser.add_argument('--video', type=str,
                        help='Path to input video (for single mode)')
    parser.add_argument('--output', type=str,
                        help='Path for output video (for single mode)')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save crops and tracking data')
    
    # Batch mode arguments
    parser.add_argument('--input-dir', type=str,
                        help='Input directory containing videos (for batch mode)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results (for batch mode)')
    parser.add_argument('--extensions', nargs='+', 
                        default=['.mp4', '.avi', '.mov', '.mkv', '.ts'],
                        help='Video file extensions to process')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Process videos even if output already exists')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.video:
            parser.error("--video is required for single mode")
        
        run_ucmc_on_video(
            video_path=args.video,
            cam_para_path=args.cam_para,
            output_path=args.output,
            save_dir=args.save_dir,
            create_video=args.create_videos or args.output is not None,
            copy_original=not args.no_copy_original
        )
    
    elif args.mode == 'batch':
        if not args.input_dir or not args.output_dir:
            parser.error("--input-dir and --output-dir are required for batch mode")
        
        process_video_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cam_para_path=args.cam_para,
            video_extensions=tuple(args.extensions),
            create_videos=args.create_videos,
            copy_originals=not args.no_copy_original,
            skip_existing=not args.no_skip_existing
        )


if __name__ == "__main__":
    # You can run the script in different ways:
    
    # 1. Command line interface
    # main()
    
    # 2. Direct batch processing
    # process_video_folder(
    #     input_dir="/mnt/c/Users/felix/Downloads/mclarens_video",
    #     output_dir="/mnt/c/Users/felix/Downloads/mclarens_res_full",
    #     cam_para_path="demo/cam_para.txt",
    #     create_videos=True,  # Set to True if you want tracked videos
    #     copy_originals=False,  # Copy original videos to output dirs
    #     skip_existing=True    # Skip already processed videos
    # )

    process_video_folder(
        input_dir="/mnt/c/Users/felix/Downloads/trucks",
        output_dir="/mnt/c/Users/felix/Downloads/trucks_res",
        cam_para_path="demo/cam_para.txt",
        create_videos=True,  # Set to True if you want tracked videos
        copy_originals=False,  # Copy original videos to output dirs
        skip_existing=True    # Skip already processed videos
    )

    
    
    # 3. Single video processing (original functionality)
    # run_ucmc_on_video(
    #     video_path="/mnt/c/Users/felix/Downloads/seg0.mp4",
    #     cam_para_path="demo/cam_para.txt",
    #     save_dir="/mnt/c/Users/felix/Downloads/seg0_td",
    #     create_video=False
    # )