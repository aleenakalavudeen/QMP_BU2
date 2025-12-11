"""
Feature Collection Script

This script runs the integrated traffic perception pipeline on one or more
video files and saves:

- features.npy : Fused feature vectors for selected frames
- labels.npy   : Integer labels (0 = safe, 1 = risk)
- log.csv      : Small CSV log with per-frame information

Risk Label Rule (simple, rule-based):
- Label = 1 (risk) if:
    - A red traffic light is detected, OR
    - A road anomaly (pothole/speedbump) is detected that appears "close"
      (its bounding box height is large compared to the image height)
- Otherwise, Label = 0 (safe)

This prepares data for a future quantum module (VQC).

Author: Integrated Traffic Perception System
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from main import IntegratedTrafficPerception


def determine_risk_label(
    detections: List[dict],
    frame_height: int,
    close_threshold: float = 0.2,
) -> Tuple[int, str]:
    """
    Determine a simple risk label for a frame based on detections.

    Args:
        detections (list): List of detection dictionaries produced by the pipeline.
        frame_height (int): Height of the frame in pixels.
        close_threshold (float): Fraction of image height above which an anomaly
                                 is considered "close" to the vehicle.

    Returns:
        (label, reason)
        label (int): 0 for safe, 1 for risk
        reason (str): Text explanation of why the label was assigned
    """
    # 1. Check for red traffic light
    for det in detections:
        if det.get("model_type") == "traffic_signal":
            class_name = str(det.get("class_name", "")).lower()
            if "red" in class_name:
                return 1, "red_light"

    # 2. Check for close road anomaly (pothole or speedbump)
    for det in detections:
        if det.get("model_type") == "road_anomaly":
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
            box_height = max(0, int(y2) - int(y1))
            # If the anomaly box covers a large fraction of the image height,
            # we treat it as "close" and therefore risky.
            if frame_height > 0 and (box_height / frame_height) >= close_threshold:
                return 1, "close_anomaly"

    # 3. Otherwise, consider frame safe
    return 0, "safe"


def collect_features_from_videos(
    video_paths: List[str],
    output_dir: str,
    every_n_frames: int = 5,
) -> None:
    """
    Run the integrated pipeline on video(s) and save features, labels, and a CSV log.

    Args:
        video_paths (list): List of paths to input video files.
        output_dir (str): Directory where features.npy, labels.npy, and log.csv will be saved.
        every_n_frames (int): Process every Nth frame to reduce computation (default: 5).
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize integrated pipeline (loads all models once)
    pipeline = IntegratedTrafficPerception()

    # Lists to store data across all videos
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []

    # Prepare CSV log file
    csv_file = output_path / "log.csv"
    csv_header = [
        "video_name",
        "frame_index",
        "label",          # 0 = safe, 1 = risk
        "reason",         # textual reason
        "num_signs",
        "num_signals",
        "num_anomalies",
    ]

    with csv_file.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        # Process each video
        for video_path in video_paths:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                print(f"Warning: Video file not found, skipping: {video_path}")
                continue

            print("\n" + "=" * 70)
            print(f"Processing video: {video_path_obj}")
            print("=" * 70)

            cap = cv2.VideoCapture(str(video_path_obj))
            if not cap.isOpened():
                print(f"Error: Could not open video: {video_path_obj}")
                continue

            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = video_path_obj.name

            frame_index = -1
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                # Subsample frames to save time and storage
                if every_n_frames > 1 and (frame_index % every_n_frames != 0):
                    continue

                # Run integrated pipeline on this frame
                results = pipeline.process_frame(frame)

                # Get fused feature vector (256 dimensions)
                fused_vector = results["fused_feature_vector"]

                # Determine risk label for this frame
                detections = results["detections"]
                label, reason = determine_risk_label(detections, frame_height)

                # Count detections for logging
                num_signs = sum(1 for d in detections if d["model_type"] == "traffic_sign")
                num_signals = sum(1 for d in detections if d["model_type"] == "traffic_signal")
                num_anomalies = sum(1 for d in detections if d["model_type"] == "road_anomaly")

                # Store data in memory
                all_features.append(fused_vector.astype(np.float32))
                all_labels.append(int(label))

                # Write CSV row
                writer.writerow(
                    [
                        video_name,
                        frame_index,
                        label,
                        reason,
                        num_signs,
                        num_signals,
                        num_anomalies,
                    ]
                )

                processed_frames += 1

                if processed_frames % 50 == 0:
                    print(
                        f"  Processed {processed_frames} frames "
                        f"(video progress: {frame_index + 1}/{total_frames})"
                    )

            cap.release()

            print(
                f"Finished video: {video_name} | "
                f"Total frames processed for features: {processed_frames}"
            )

    # After processing all videos, save numpy arrays
    if len(all_features) == 0:
        print("\nNo frames were processed. No feature files were saved.")
        return

    features_array = np.stack(all_features, axis=0)
    labels_array = np.array(all_labels, dtype=np.int64)

    np.save(output_path / "features.npy", features_array)
    np.save(output_path / "labels.npy", labels_array)

    print("\n" + "=" * 70)
    print("Feature collection complete!")
    print("=" * 70)
    print(f"Saved features to: {output_path / 'features.npy'} (shape: {features_array.shape})")
    print(f"Saved labels to:   {output_path / 'labels.npy'} (shape: {labels_array.shape})")
    print(f"Saved log to:      {csv_file}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the feature collection script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect fused features and risk labels from videos using the "
            "integrated traffic perception pipeline."
        )
    )

    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to input video files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="feature_data",
        help="Directory to save features.npy, labels.npy, and log.csv (default: feature_data).",
    )

    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=5,
        help="Process every Nth frame to reduce computation (default: 5).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Example usage:

    # Process a single video and save data in 'feature_data' folder
    python collect_features.py --videos path/to/video.mp4

    # Process multiple videos and save to a custom folder
    python collect_features.py --videos v1.mp4 v2.mp4 --output-dir my_features

    # Process more frames (every frame instead of every 5th)
    python collect_features.py --videos v1.mp4 --every-n-frames 1
    """
    args = parse_args()
    collect_features_from_videos(
        video_paths=args.videos,
        output_dir=args.output_dir,
        every_n_frames=args.every_n_frames,
    )


