#!/usr/bin/env python3
"""
Modular frame-by-frame tracker for UCMCTrack using live YOLO12 + RT-DETR detection.
"""
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
    if res.boxes is None or res.boxes.data.numel() == 0:
        return res.new()
    data = res.boxes.data
    coords = data[:, :4]
    scores = data[:, 4]
    labels = data[:, 5].long()
    N = len(scores)
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
    fused_data = torch.stack(fused, 0)
    out = res.new()
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
    def __init__(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.yolo12 = YOLO("yolo12x.pt")
        self.detr = RTDETR("rtdetr-x.pt")

    def detect_frame(self, frame, frame_id, conf_thresh=0.01):
        yolo_res = self.yolo12.predict(frame, agnostic_nms=True, iou=0.5, verbose=False)[0]
        detr_res = self.detr.predict(frame, agnostic_nms=True, iou=0.5, verbose=False)[0]
        merged = merge_frame_results([yolo_res, detr_res], iou_thresh=0.5)
        fused = fuse_union(merged, iou_thr=0.5, ioa_thr=0.7)
        dets, det_id = [], 0
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
    def __init__(self, cam_para_path, wx=5.0, wy=5.0, vmax=10.0, a=100.0, cdt=60.0, fps=30.0, high_score=0.5, conf_thresh=0.01):
        self.detector = LiveDetector(cam_para_path)
        self.tracker = UCMCTrack(a, a, wx, wy, vmax, cdt, fps, "MOT", high_score, False, None)
        self.conf_thresh = conf_thresh
        self.frame_id = 1

    def process_frame(self, frame):
        dets = self.detector.detect_frame(frame, self.frame_id, self.conf_thresh)
        self.tracker.update(dets, self.frame_id)
        self.frame_id += 1
        return dets
    

def run_ucmc_on_video(video_path, cam_para_path, output_path):
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
        cdt=60.0,
        high_score=0.5,
        conf_thresh=0.01
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(det.track_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(frame)
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    writer.release()
    print(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    run_ucmc_on_video(
        video_path="/mnt/c/Users/felix/Downloads/seg0.mp4",
        cam_para_path="demo/cam_para.txt",
        output_path="/mnt/c/Users/felix/Downloads/seg0_ucmc.mp4"
    )

