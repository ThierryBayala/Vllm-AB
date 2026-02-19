"""
Display utilities for frames and descriptions (e.g. in Jupyter).
"""

import base64
import io
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


def frame_to_base64(frame: Optional[np.ndarray], max_width: int = 320) -> Optional[str]:
    """Convert numpy frame to base64 PNG; optionally scale for display."""
    if frame is None:
        return None
    # Frames from processor are RGB float32 [0,1]; convert to uint8 for PIL
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # if BGR from cv2
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        new_w, new_h = max_width, int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_descriptions_to_html(
    frames: Sequence[Optional[np.ndarray]],
    descriptions: List[str],
    frame_indices: Optional[Sequence[int]] = None,
    max_width: int = 320,
) -> str:
    """Build an HTML table string with frame images and their descriptions."""
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    rows = []
    for i, (frame, desc) in enumerate(zip(frames, descriptions)):
        b64 = frame_to_base64(frame, max_width=max_width)
        if b64 is None:
            img_html = "<em>No image</em>"
        else:
            img_html = f'<img src="data:image/png;base64,{b64}" style="max-width:{max_width}px; height:auto;" />'
        idx = frame_indices[i] if i < len(frame_indices) else i
        rows.append(
            f"<tr><td style='vertical-align:top; padding:8px;'>{img_html}</td>"
            f"<td style='vertical-align:top; padding:8px; max-width:400px;'>"
            f"<strong>Frame {idx}</strong><br/><br/>{desc}</td></tr>"
        )
    return (
        "<table style='border-collapse: collapse;'>"
        "<thead><tr><th style='text-align:left; padding:8px;'>Image</th>"
        "<th style='text-align:left; padding:8px;'>Description</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def display_frames_with_descriptions(
    frames: Sequence[Optional[np.ndarray]],
    descriptions: List[str],
    frame_indices: Optional[Sequence[int]] = None,
    max_width: int = 320,
) -> Optional[str]:
    """
    Display frames and descriptions in a two-column HTML table (e.g. in Jupyter).
    If IPython is available, runs display(HTML(...)); otherwise returns the HTML string.
    """
    html = frames_descriptions_to_html(frames, descriptions, frame_indices, max_width)
    try:
        from IPython.display import HTML, display

        display(HTML(html))
        return None
    except ImportError:
        return html
