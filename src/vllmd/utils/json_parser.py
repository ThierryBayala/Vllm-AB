"""
Parse video description JSON files (as produced by describe_videos_in_abnormal).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


def load_video_descriptions_json(path: str | Path) -> List[dict[str, Any]]:
    """
    Parse a video descriptions JSON file (as produced by describe_videos_in_abnormal).

    Each entry has: video_index (int), context (str), description (str),
    named_entities (list of {"text": str, "label": str}).

    Args:
        path: Path to the JSON file (e.g. data/abnormal/Attack/attack.json).

    Returns:
        List of video description records.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def load_all_video_descriptions(
    abnormal_root: str | Path,
) -> dict[str, List[dict[str, Any]]]:
    """
    Discover and parse all context-level video description JSON files under abnormal_root.

    Expects structure: abnormal_root/<Context>/<context_lower>.json
    (e.g. abnormal/Attack/attack.json, abnormal/Arrest/arrest.json).

    Args:
        abnormal_root: Root of the abnormal folder (e.g. data/abnormal).

    Returns:
        Dict mapping context folder name (e.g. "Attack") to list of video description records.
    """
    abnormal_root = Path(abnormal_root)
    if not abnormal_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {abnormal_root}")

    result: dict[str, List[dict[str, Any]]] = {}
    for context_dir in sorted(abnormal_root.iterdir()):
        if not context_dir.is_dir():
            continue
        context_name = context_dir.name
        json_path = context_dir / f"{context_name.lower()}.json"
        if json_path.is_file():
            result[context_name] = load_video_descriptions_json(json_path)
    return result
