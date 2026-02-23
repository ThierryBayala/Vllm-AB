"""
Describe videos in data/abnormal using Gemini (one representative frame per video)
and extract named entities with spaCy. Writes one JSON per context folder
(e.g. attack.json inside data/abnormal/Attack/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

from vllmd.utils.frame_describe import (
    VISUAL_DESCRIPTION_PROMPT,
    extract_named_entities_spacy,
)
from vllmd.utils.video_frames import VIDEO_EXTENSIONS

# Pattern for frame filenames like frame_000001.png
FRAME_FILENAME_PATTERN = re.compile(r"^frame_(\d+)\.", re.IGNORECASE)
FRAME_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _frame_sort_key(path: Path) -> int:
    """Extract numeric frame index from filename (e.g. frame_000001.png -> 1) for sorting."""
    m = FRAME_FILENAME_PATTERN.match(path.name)
    return int(m.group(1)) if m else 0


def _get_middle_frame_from_video(video_path: Path) -> Optional[bytes]:
    """
    Extract the middle frame from a video as JPEG bytes (RGB).
    Returns None if the video cannot be read.
    """
    try:
        from vllmd.llm.llm_based import _frame_to_bytes
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV and vllmd.llm.llm_based required for video frame extraction") from e

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _frame_to_bytes(frame_rgb)


def _get_representative_frame_path(
    context_dir: Path,
    video_stem: str,
    frame_images_root: Optional[Path],
) -> Optional[Path]:
    """
    If frame_images_root is set and contains frames for this video,
    return the path to the middle frame image. Otherwise return None.
    """
    if frame_images_root is None:
        return None
    video_frames_dir = frame_images_root / context_dir.name / video_stem
    if not video_frames_dir.is_dir():
        return None
    paths = [
        p
        for p in video_frames_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in FRAME_IMAGE_EXTENSIONS
        and FRAME_FILENAME_PATTERN.match(p.name)
    ]
    if not paths:
        return None
    paths.sort(key=_frame_sort_key)
    return paths[len(paths) // 2]


def _video_index_from_stem(stem: str, fallback_index: int) -> int:
    """Convert video stem (e.g. '01', '02') to integer video_index; fallback to 1-based index."""
    s = stem.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    return fallback_index


def describe_videos_in_abnormal(
    data_folder: str | Path,
    *,
    abnormal_folder_name: str = "abnormal",
    frame_images_folder_name: str = "frame-images",
    frame_images_root: Optional[str | Path] = None,
    model_name: str = "gemini-2.0-flash",
    prompt: str = VISUAL_DESCRIPTION_PROMPT,
    api_key: Optional[str] = None,
    video_extensions: Optional[List[str]] = None,
) -> dict[str, Path]:
    """
    Describe each video in data/abnormal using Gemini and save one JSON per context folder.

    For each context (e.g. Attack, Arrest):
      - For each video (e.g. 01.mp4, 02.mp4), uses one representative frame (middle).
      - Frame is taken from data/frame-images/<context>/<video_stem>/ if available,
        otherwise extracted from the video file.
      - Gets a description from Gemini using the given prompt (default: accessibility-style, ≤60 words).
      - Extracts named entities from the description using spaCy.
      - Writes <context_lower>.json inside the context folder (e.g. data/abnormal/Attack/attack.json).

    Each object in the JSON has:
      - video_index: int (e.g. 1 for 01.mp4)
      - context: str (e.g. "attack")
      - description: str (from Gemini)
      - named_entities: list of {"text": str, "label": str} from spaCy

    Dependencies:
      - Gemini: pip install google-genai, set GEMINI_API_KEY in .env
      - spaCy: pip install spacy && python -m spacy download en_core_web_sm
      - OpenCV (cv2) when frame-images are not present

    Args:
        data_folder: Path to the data folder (e.g. PROJECT_ROOT / "data").
        abnormal_folder_name: Name of the abnormal folder under data (default "abnormal").
        frame_images_folder_name: Name of frame-images folder under data (default "frame-images").
        frame_images_root: Override root for frame images; if None, uses data_folder / frame_images_folder_name.
        model_name: Gemini model (e.g. gemini-2.0-flash).
        prompt: Prompt for the vision model (default: VISUAL_DESCRIPTION_PROMPT).
        api_key: Optional Gemini API key; else uses GEMINI_API_KEY from env.
        video_extensions: Extensions to treat as video (default: .mp4, .avi, .mov, .mkv, .webm).

    Returns:
        Dict mapping context name to the path of the written JSON file.
    """
    from vllmd.llm.llm_based import describe_image

    data_folder = Path(data_folder)
    abnormal_dir = data_folder / abnormal_folder_name
    if not abnormal_dir.is_dir():
        raise FileNotFoundError(f"Abnormal directory not found: {abnormal_dir}")

    root_frames = Path(frame_images_root) if frame_images_root is not None else data_folder / frame_images_folder_name
    ext_set = (
        {e if e.startswith(".") else f".{e}" for e in video_extensions}
        if video_extensions
        else VIDEO_EXTENSIONS
    )

    written: dict[str, Path] = {}

    for context_dir in sorted(abnormal_dir.iterdir()):
        if not context_dir.is_dir():
            continue
        context_name = context_dir.name
        context_lower = context_name.lower()

        records: List[dict[str, Any]] = []
        video_list = sorted(
            [p for p in context_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set],
            key=lambda p: p.stem,
        )

        for idx, video_path in enumerate(video_list):
            video_stem = video_path.stem
            video_index = _video_index_from_stem(video_stem, idx + 1)

            frame_path = _get_representative_frame_path(context_dir, video_stem, root_frames)
            if frame_path is not None and frame_path.exists():
                description = describe_image(
                    frame_path,
                    model_name,
                    prompt,
                    api_key=api_key,
                )
            else:
                image_bytes = _get_middle_frame_from_video(video_path)
                if image_bytes is None:
                    description = "[Could not extract frame from video]"
                else:
                    description = describe_image(
                        image_bytes,
                        model_name,
                        prompt,
                        api_key=api_key,
                    )

            named_entities = extract_named_entities_spacy(description)
            records.append({
                "video_index": video_index,
                "context": context_lower,
                "description": description,
                "named_entities": named_entities,
            })

        if not records:
            continue

        out_path = context_dir / f"{context_lower}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        written[context_name] = out_path

    return written
