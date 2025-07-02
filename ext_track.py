#!/usr/bin/env python3
"""
Modular frame-by-frame tracker for UCMCTrack using live YOLO12 + RT-DETR detection.
"""

from typing import Sequence, Callable, List, Dict, Optional
from functools import cached_property
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR
from torchvision.ops import nms, box_iou


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
                 conf=0.0, det_class=0, class_name=None):
        self.id = det_id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.class_name = class_name  # Add class name attribute
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return (f"d{self.id}, bb:[{self.bb_left},{self.bb_top},"
                f"{self.bb_width},{self.bb_height}], "
                f"conf={self.conf:.2f}, cls{self.det_class}, "
                f"name={self.class_name}, "
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
        
        # Get class names from the first model (assuming all models use same classes)
        if hasattr(models[0], 'names'):
            self.class_names = models[0].names
        else:
            self.class_names = {}

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
                
                # Get class name
                class_name = self.class_names.get(int(cls), f"class_{int(cls)}")
                
                det = Detection(det_id, x1, y1, w, h, conf, int(cls), class_name)
                det.y, det.R = self.mapper.mapto([x1, y1, w, h])
                dets.append(det)
                det_id += 1

        return dets


class UCMCTracker:
    def __init__(self, cam_para_path, wx=5.0, wy=5.0, vmax=10.0, a=100.0, cdt=60.0, 
                 fps=30.0, high_score=0.5, conf_thresh=0.01, show_class_names=True):
        # self.detector = LiveDetector(cam_para_path)
        self.detector = LiveDetector(cam_para_path, [YOLO("yolo12x.pt"), RTDETR("rtdetr-x.pt")])
        # self.detector = LiveDetector(cam_para_path, [RTDETR("rtdetr-x.pt")])
        # self.detector = LiveDetector(cam_para_path, [YOLO("/home/felix/models/detr_visdrone5/weights/best.pt")])
        self.tracker = UCMCTrack(a, a, wx, wy, vmax, cdt, fps, "MOT", high_score, False, None)
        self.conf_thresh = conf_thresh
        self.show_class_names = show_class_names  # Add option to show class names
        self.frame_id = 1

    def process_frame(self, frame):
        dets = self.detector.detect_frame(frame, self.frame_id, self.conf_thresh)
        self.tracker.update(dets, self.frame_id)
        self.frame_id += 1
        return dets
    

def run_ucmc_on_video(video_path, cam_para_path, output_path, show_class_names=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        print("Warning: Could not determine FPS. Defaulting to 30.")
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    # writer.set(cv2.VIDEOWRITER_PROP_BITRATE, 2_000_000)

    tracker = UCMCTracker(
        cam_para_path=cam_para_path,
        fps=fps,
        wx=5.0, wy=5.0,
        vmax=10.0,
        a=100.0,
        cdt=120.0,
        high_score=0.5,
        conf_thresh=0.01,
        show_class_names=show_class_names
    )

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = tracker.process_frame(frame)

        # Draw tracked boxes
        for det in detections:
            if det.track_id <= 0:
                continue
            x1 = int(det.bb_left)
            y1 = int(det.bb_top)
            x2 = int(det.bb_left + det.bb_width)
            y2 = int(det.bb_top + det.bb_height)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            if show_class_names and det.class_name:
                label = f"{det.track_id}: {det.class_name}"
            else:
                label = str(det.track_id)
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 5, y1 - 5),
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 2, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    writer.release()
    print(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    # Example with class names enabled (default)
    run_ucmc_on_video(
        video_path="/mnt/c/Users/felix/Downloads/barge1.mp4",
        cam_para_path="demo/cam_para.txt",
        output_path="/mnt/c/Users/felix/Downloads/barge1_res_full.mp4",
        show_class_names=False
    )

    # # Example with class names disabled
    # run_ucmc_on_video(
    #     video_path="/mnt/c/Users/felix/Downloads/TSS6366_crop.ts",
    #     cam_para_path="demo/cam_para.txt",
    #     output_path="/mnt/c/Users/felix/Downloads/res_fused_TSS6366_crop_120cdt.mp4",
    #     show_class_names=False
    # )