#!/usr/bin/env python3
"""Generate synthetic or animated face test video files for reproducible testing and benchmarking.

Does not contain personal or private biometric data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def generate_geometric_video(
    output_path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    num_frames: int = 150,
) -> Path:
    """Generate a synthetic video with moving geometric elements (no human faces)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Gradient background
            frame[:, :, 0] = np.linspace(20, 60, width, dtype=np.uint8)
            frame[:, :, 1] = np.linspace(30, 70, width, dtype=np.uint8)
            frame[:, :, 2] = np.linspace(40, 80, width, dtype=np.uint8)

            # Moving element
            cx = int(width / 2 + (width / 4) * np.sin(2 * np.pi * i / num_frames))
            cy = int(height / 2 + (height / 6) * np.cos(2 * np.pi * i / num_frames))
            cv2.circle(frame, (cx, cy), 50, (0, 180, 240), -1)

            cv2.putText(
                frame,
                f"Synthetic Geometric Frame {i + 1}/{num_frames}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def generate_face_animation_video(
    face_image_path: Path,
    output_path: Path,
    target_width: int = 640,
    target_height: int = 480,
    fps: float = 30.0,
    num_frames: int = 150,
) -> Path:
    """Generate a multi-frame test video animating a standard reference face fixture."""
    if not face_image_path.exists():
        raise FileNotFoundError(f"Face reference image not found: {face_image_path}")

    base_img = cv2.imread(str(face_image_path))
    if base_img is None:
        raise ValueError(f"Could not decode image at: {face_image_path}")

    # Resize base image to fit nicely within target canvas
    src_h, src_w = base_img.shape[:2]
    scale = min(target_width * 0.7 / src_w, target_height * 0.7 / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    scaled_face = cv2.resize(base_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        base_x = (target_width - new_w) // 2
        base_y = (target_height - new_h) // 2

        for i in range(num_frames):
            # Canvas with subtle gradient
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            canvas[:, :, :] = (40, 40, 40)

            # Deterministic translation
            shift_x = int(15 * np.sin(2 * np.pi * i / num_frames))
            shift_y = int(10 * np.cos(2 * np.pi * i / num_frames))
            pos_x = max(0, min(target_width - new_w, base_x + shift_x))
            pos_y = max(0, min(target_height - new_h, base_y + shift_y))

            # Composite face
            canvas[pos_y : pos_y + new_h, pos_x : pos_x + new_w] = scaled_face

            # Watermark
            cv2.putText(
                canvas,
                f"Face Test Frame {i + 1}/{num_frames}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            writer.write(canvas)
    finally:
        writer.release()

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic or face test video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test_video.mp4"),
        help="Output video file path (default: data/test_video.mp4)",
    )
    parser.add_argument(
        "--face-image",
        type=Path,
        help="Path to reference face image fixture (animates face if provided)",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30.0)")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames (default: 150)")
    args = parser.parse_args()

    try:
        if args.face_image:
            out = generate_face_animation_video(
                face_image_path=args.face_image,
                output_path=args.output,
                target_width=args.width,
                target_height=args.height,
                fps=args.fps,
                num_frames=args.frames,
            )
        else:
            out = generate_geometric_video(
                output_path=args.output,
                width=args.width,
                height=args.height,
                fps=args.fps,
                num_frames=args.frames,
            )
        msg = f"Generated test video ({args.width}x{args.height}, {args.frames} frames)"
        print(f"{msg} -> {out}")
        return 0
    except Exception as e:
        print(f"Error generating video: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
