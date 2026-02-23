"""
Extract frames from videos and save them to disk.
Used to build data/frame-images from data/abnormal subdirectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2

# Common video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_frames_from_video(
    video_path: str | Path,
    num_frames: int = 300,
    output_dir: Optional[str | Path] = None,
    frame_prefix: str = "frame",
    image_format: str = "png",
) -> int:
    """
    Extract up to num_frames evenly sampled from a video and save as images.

    Args:
        video_path: Path to the video file.
        num_frames: Maximum number of frames to extract (default 300).
        output_dir: Directory to save frame images. If None, frames are not saved.
        frame_prefix: Filename prefix for saved frames (e.g. frame_000001.png).
        image_format: Image extension for saved frames (e.g. 'png', 'jpg').

    Returns:
        Number of frames actually saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # Evenly sample up to num_frames (indices 0, 1, ..., total_frames-1)
    n = min(num_frames, total_frames)
    if n > 1:
        indices = [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]
    else:
        indices = [0]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        if output_dir is not None:
            name = f"{frame_prefix}_{saved + 1:06d}.{image_format}"
            out_path = output_dir / name
            cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def split_abnormal_videos_into_frames(
    data_folder: str | Path,
    num_frames: int = 300,
    frame_images_folder_name: str = "frame-images",
    abnormal_folder_name: str = "abnormal",
    video_extensions: Optional[List[str]] = None,
    image_format: str = "png",
) -> dict[str, int]:
    """
    For each video in subdirectories of data/abnormal, extract num_frames (default 300)
    and save them under data/frame-images, mirroring the abnormal directory structure.

    Creates:
      - data/frame-images/
      - data/frame-images/<subdir>/  for each subdir in data/abnormal
      - data/frame-images/<subdir>/<video_stem>/  for each video (folder named after video)
      - Frame images inside each video folder (e.g. frame_000001.png, ...).

    Args:
        data_folder: Path to the data folder (e.g. PROJECT_ROOT / "data").
        num_frames: Number of frames to extract per video (default 300).
        frame_images_folder_name: Name of the output folder under data (default "frame-images").
        abnormal_folder_name: Name of the abnormal folder under data (default "abnormal").
        video_extensions: List of extensions to treat as video (default: .mp4, .avi, .mov, .mkv, .webm).
        image_format: Format for saved frames (default "png").

    Returns:
        Dict mapping subdir name to total number of frames written for that subdir.
    """
    data_folder = Path(data_folder)
    abnormal_dir = data_folder / abnormal_folder_name
    frame_images_dir = data_folder / frame_images_folder_name

    if not abnormal_dir.is_dir():
        raise FileNotFoundError(f"Abnormal directory not found: {abnormal_dir}")

    ext_set = (
        {e if e.startswith(".") else f".{e}" for e in video_extensions}
        if video_extensions
        else VIDEO_EXTENSIONS
    )

    frames_per_subdir: dict[str, int] = {}

    for subdir in sorted(abnormal_dir.iterdir()):
        if not subdir.is_dir():
            continue
        subdir_name = subdir.name
        out_subdir = frame_images_dir / subdir_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        total_saved = 0

        for video_path in sorted(subdir.iterdir()):
            if video_path.suffix.lower() not in ext_set:
                continue
            # Folder named after the video (stem, e.g. "07" for "07.mp4")
            video_stem = video_path.stem
            video_out_dir = out_subdir / video_stem
            n = extract_frames_from_video(
                video_path,
                num_frames=num_frames,
                output_dir=video_out_dir,
                frame_prefix="frame",
                image_format=image_format,
            )
            total_saved += n

        frames_per_subdir[subdir_name] = total_saved

    return frames_per_subdir
