"""
Describe frame images in data/frame-images using Gemini and extract named entities with spaCy.
Writes one JSON file per video subfolder (e.g. attack_01.json inside Attack/01/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

# Default prompt for visual description (accessibility-style, factual, ≤60 words)
VISUAL_DESCRIPTION_PROMPT = """You are a professional Visual Description Assistant whose job is to produce clear, useful descriptions of images for humans and accessibility tools. To describe the image, you need to follow below task instruction.

Think step by step.

Task Instruction:
    1 Try to understand the image deeply by your step by step thinking process.
    2 Carefully examine the input image before generating any description.
    3 Identify and list only objective visual elements (objects, people, actions, positions, numbers, background details).
    4 If the image is ambiguous, clearly state uncertainty and lower the confidence score.
    5 If the image contains sensitive content (e.g., violence, nudity, weapons), include a brief safety note in the description.
    6 Keep the description strictly factual (what is visibly present).

Format:
    You need to generate the description within 60 words with easy understandable sentence."""

# Supported image extensions for frame files
FRAME_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
# Pattern for frame filenames like frame_000001.png
FRAME_FILENAME_PATTERN = re.compile(r"^frame_(\d+)\.", re.IGNORECASE)

_nlp = None


def _get_spacy_nlp():
    """Lazy-load spaCy NER model. Requires: pip install spacy && python -m spacy download en_core_web_sm"""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception as e:
        raise ImportError(
            "spaCy NER requires: pip install spacy && python -m spacy download en_core_web_sm"
        ) from e


def extract_named_entities_spacy(text: str) -> List[dict[str, Any]]:
    """
    Extract named entities from text using spaCy.

    Returns:
        List of {"text": str, "label": str} for each entity (e.g. PERSON, ORG, GPE).
    """
    if not text or not text.strip():
        return []
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def _frame_sort_key(path: Path) -> int:
    """Extract numeric frame index from filename (e.g. frame_000001.png -> 1) for sorting."""
    m = FRAME_FILENAME_PATTERN.match(path.name)
    return int(m.group(1)) if m else 0


def _collect_frame_images(video_dir: Path) -> List[Path]:
    """Return sorted list of frame image paths in a video subfolder."""
    paths = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in FRAME_IMAGE_EXTENSIONS
        and FRAME_FILENAME_PATTERN.match(p.name)
    ]
    return sorted(paths, key=_frame_sort_key)


def describe_frame_images_in_folder(
    frame_images_root: str | Path,
    *,
    model_name: str = "gemini-2.0-flash",
    prompt: str = VISUAL_DESCRIPTION_PROMPT,
    api_key: Optional[str] = None,
    image_extensions: Optional[List[str]] = None,
) -> dict[str, Path]:
    """
    Describe each frame image under frame-images using Gemini and save one JSON per video folder.

    Directory layout expected:
      frame_images_root/
        <context>/           (e.g. Attack, Arrest, Fighting)
          <video_stem>/      (e.g. 01, 02)
            frame_000001.png, frame_000002.png, ...

    For each <context>/<video_stem>/:
      - Describes each frame with Gemini using the given prompt.
      - Extracts named entities from each description using spaCy.
      - Writes <context_lower>_<video_stem>.json inside that folder.

    Each entry in the JSON is:
      - frame_index: int (from filename, e.g. 1 for frame_000001.png)
      - context: str (e.g. "Attack")
      - description: str (from Gemini)
      - named_entities: list of {"text": str, "label": str} from spaCy

    Dependencies:
      - Gemini: pip install google-genai, set GEMINI_API_KEY in .env
      - spaCy: pip install spacy && python -m spacy download en_core_web_sm

    Args:
        frame_images_root: Root path of the frame-images folder.
        model_name: Gemini model (e.g. gemini-2.0-flash).
        prompt: Prompt for the vision model (default: VISUAL_DESCRIPTION_PROMPT).
        api_key: Optional Gemini API key; else uses GEMINI_API_KEY from env.
        image_extensions: Extensions to treat as frame images (default: .png, .jpg, .jpeg, .webp).

    Returns:
        Dict mapping "<context>/<video_stem>" to the path of the written JSON file.
    """
    from vllmd.llm.llm_based import describe_image

    root = Path(frame_images_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Frame images root is not a directory: {root}")

    ext_set = (
        {e if e.startswith(".") else f".{e}" for e in image_extensions}
        if image_extensions
        else FRAME_IMAGE_EXTENSIONS
    )

    written: dict[str, Path] = {}

    for context_dir in sorted(root.iterdir()):
        if not context_dir.is_dir():
            continue
        context = context_dir.name

        for video_dir in sorted(context_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_stem = video_dir.name

            frame_paths = [
                p
                for p in video_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in ext_set
                and FRAME_FILENAME_PATTERN.match(p.name)
            ]
            frame_paths = sorted(frame_paths, key=_frame_sort_key)
            if not frame_paths:
                continue

            records: List[dict[str, Any]] = []
            for path in frame_paths:
                m = FRAME_FILENAME_PATTERN.match(path.name)
                frame_index = int(m.group(1)) if m else 0

                description = describe_image(
                    path,
                    model_name,
                    prompt,
                    api_key=api_key,
                )
                named_entities = extract_named_entities_spacy(description)

                records.append({
                    "frame_index": frame_index,
                    "context": context,
                    "description": description,
                    "named_entities": named_entities,
                })

            json_name = f"{context.lower()}_{video_stem}.json"
            out_path = video_dir / json_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

            key = f"{context}/{video_stem}"
            written[key] = out_path

    return written
