"""Run Video -> Detector -> Tracker; save an annotated video and print track statistics.

Exploratory script (not part of the package). The proper visualization module
comes later; this exists so we can SEE whether tracking works.

Example (Colab, GPU):
    python scripts/track_video.py data/traffic.mp4 --weights yolo11m.pt --imgsz 1280
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from traffic_analytics.core.types import Tracks
from traffic_analytics.detection.base import Detector
from traffic_analytics.tracking.base import Tracker
from traffic_analytics.video import VideoSource


def color_for(track_id: int) -> tuple:
    rng = np.random.default_rng(track_id)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def draw(frame: np.ndarray, tracks: Tracks, history: dict) -> np.ndarray:
    out = frame.copy()
    for box, tid in zip(tracks.xyxy, tracks.track_id):
        tid = int(tid)
        x1, y1, x2, y2 = box.astype(int)
        color = color_for(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"#{tid}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        history[tid].append(((x1 + x2) // 2, (y1 + y2) // 2))
        pts = np.array(history[tid][-40:], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, color, 2)
    return out


def run(source: VideoSource, detector: Detector, tracker: Tracker,
        out_path: str, max_frames: int | None = None) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    info = source.info
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    history: dict = defaultdict(list)
    lifetime: dict = defaultdict(int)   # track_id -> number of frames it appeared in
    per_frame: list = []
    n = 0
    start = time.perf_counter()
    for frame in source.frames(max_frames):
        tracks = tracker.update(detector.detect(frame))
        for tid in tracks.track_id:
            lifetime[int(tid)] += 1
        per_frame.append(len(tracks))
        writer.write(draw(frame, tracks, history))
        n += 1
    writer.release()
    elapsed = time.perf_counter() - start
    return {
        "frames": n, "seconds": elapsed, "fps": n / elapsed if elapsed else 0.0,
        "unique_ids": len(lifetime), "lifetime": dict(lifetime), "per_frame": per_frame,
    }


def print_report(stats: dict, short: int = 10) -> None:
    life = stats["lifetime"]
    pf = stats["per_frame"]
    print(f"frames={stats['frames']}  time={stats['seconds']:.1f}s  speed={stats['fps']:.2f} FPS")
    print(f"unique track IDs: {stats['unique_ids']}")
    print(f"tracks per frame: min={min(pf)} median={int(np.median(pf))} max={max(pf)}")
    print(f"track lifetimes (frames): {sorted(life.values(), reverse=True)}")
    n_short = sum(1 for v in life.values() if v < short)
    print(f"short tracks (< {short} frames): {n_short}  <- many of these = fragmentation/flicker")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/traffic.mp4")
    ap.add_argument("--out", default="outputs/tracked.mp4")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--activation", type=float, default=0.15, help="ByteTrack activation threshold")
    ap.add_argument("--buffer", type=int, default=30, help="ByteTrack lost_track_buffer")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()

    from traffic_analytics.detection import YoloDetector
    from traffic_analytics.tracking import ByteTrackTracker

    with VideoSource(a.video) as src:
        detector = YoloDetector(weights=a.weights, conf=a.conf, iou=a.iou,
                                imgsz=a.imgsz, device=a.device)
        tracker = ByteTrackTracker(fps=src.info.fps, activation_threshold=a.activation,
                                   lost_track_buffer=a.buffer)
        print(f"device={detector.device} weights={a.weights} imgsz={a.imgsz} "
              f"conf={a.conf} iou={a.iou} activation={a.activation} buffer={a.buffer}")
        stats = run(src, detector, tracker, a.out, a.max_frames)
    print_report(stats)
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
