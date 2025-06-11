#!/usr/bin/env python3
"""
Head-less demo for UCMCTrack using pre-recorded JSON detections.
Reads an MP4, loads a per-frame detection JSON, runs UCMCTrack on every frame,
and writes the annotated result to a new MP4.
"""

import os
import cv2
import json
import argparse
import numpy as np

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper


class Detection:
    """Simple container matching the original repo’s Detection struct."""
    def __init__(self, det_id,
                 bb_left=0, bb_top=0,
                 bb_width=0, bb_height=0,
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


class Detector:
    def __init__(self, cam_para_file, det_json_path):
        self.mapper = Mapper(cam_para_file, "MOT17")
        with open(det_json_path, "r") as f:
            self.raw = json.load(f)

    def get_dets(self, frame_id, conf_thresh=0.01):
        idx = frame_id - 1
        if idx < 0 or idx >= len(self.raw):
            return []

        entry = self.raw[idx]
        dets, det_id = [], 0

        # we assume entry is a list of dicts with "box" & "confidence"
        for i, dd in enumerate(entry):
            try:
                # attempt to parse
                conf = float(dd["confidence"])
                if conf < conf_thresh:
                    continue

                b = dd["box"]
                x1, y1 = float(b["x1"]), float(b["y1"])
                x2, y2 = float(b["x2"]), float(b["y2"])
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                det = Detection(det_id, x1, y1, w, h, conf, int(dd["class"]))
                det.y, det.R = self.mapper.mapto(
                    [det.bb_left, det.bb_top, det.bb_width, det.bb_height])
                dets.append(det)
                det_id += 1

            except Exception as e:
                # debug output: frame and index, plus the bad entry
                print(f"[DEBUG] Failed to parse detection in frame {frame_id}, "
                      f"entry {i}: {dd!r}\n         Error: {e}")
                # skip this detection and continue
                continue

        return dets






def run(args):
    # --------------------------------------------------
    # I/O
    # --------------------------------------------------
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.isfile(args.cam_para):
        raise FileNotFoundError(f"Camera-parameter file not found: {args.cam_para}")
    if not os.path.isfile(args.det_json):
        raise FileNotFoundError(f"Detections JSON not found: {args.det_json}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    writer.set(cv2.VIDEOWRITER_PROP_BITRATE, 2_000_000)

    # --------------------------------------------------
    # init detector + tracker
    # --------------------------------------------------
    detector = Detector(args.cam_para, args.det_json)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy,
                        args.vmax, args.cdt, fps,
                        "MOT", args.high_score, False, None)

    # --------------------------------------------------
    # main loop
    # --------------------------------------------------
    frame_id = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.get_dets(frame_id, args.conf_thresh)
        tracker.update(dets, frame_id)

        # draw results
        for det in dets:
            if det.track_id <= 0:
                continue
            x1, y1 = int(det.bb_left), int(det.bb_top)
            x2, y2 = int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(det.track_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(frame)
        frame_id += 1
        if frame_id % 100 == 0:
            print('***********************************************************')
            print(f"[UCMCTrack] processed {frame_id} frames…", flush=True)
            print('***********************************************************')

    cap.release()
    writer.release()
    print(f"Done. Output saved to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Head-less demo for UCMCTrack with JSON detections.")
    p.add_argument("--video",      type=str, required=True,
                   help="input video file")
    p.add_argument("--cam_para",   type=str, required=True,
                   help="camera parameter file")
    p.add_argument("--det_json",   type=str, required=True,
                   help="pre-saved detections JSON file")
    p.add_argument("--out",        type=str, default="output/output.mp4",
                   help="output video file")
    p.add_argument("--wx",         type=float, default=5.0)
    p.add_argument("--wy",         type=float, default=5.0)
    p.add_argument("--vmax",       type=float, default=10.0)
    p.add_argument("--a",          type=float, default=100.0,
                   help="assignment threshold")
    p.add_argument("--cdt",        type=float, default=10.0,
                   help="coasted deletion time")
    p.add_argument("--high_score", type=float, default=0.5)
    p.add_argument("--conf_thresh", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
