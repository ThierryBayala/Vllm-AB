"""
LLM-based image description using Google Gemini vision API.

Dependencies:
  pip install google-genai Pillow python-dotenv

API key: set GEMINI_API_KEY in a .env file in the project root.
  GEMINI_API_KEY=your_key_here
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal
import warnings
warnings.filterwarnings("ignore")

def _project_root() -> Path:
    """Find project root (directory containing pyproject.toml or .env)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return p.parent.parent.parent


# Load .env so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root() / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Supported Gemini model names
# ---------------------------------------------------------------------------
MODEL_NAMES = (
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
)

ModelName = Literal[
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]


def describe_image_gemini(
    image_input: str | Path | bytes,
    model_name: str,
    prompt: str = """
      Role: You are a professional Visual Description Assistant whose job is to produce clear, useful descriptions of images for humans and accessibility tools. To describe the image, you need to follow below task instruction.
  
      Think step by step.
      
      Task Instruction:
          1 Try to understand the image deeply by your step by step thinking process.
          2 Carefully examine the input image before generating any description.
          3 Identify and list only objective visual elements (objects, people, actions, positions, numbers, background details).
          4 If the image is ambiguous, clearly state uncertainty and lower the confidence score.
          5 If the image contains sensitive content (e.g., violence, nudity, weapons), include a brief safety note in the description.
          6 Keep the description strictly factual (what is visibly present).
      
      Format:
          You need to generate the description within 60 words with easy understandable sentence.
    """,
    api_key: str | None = None,
) -> str:
    """Describe image using Google Gemini vision API (google.genai SDK)."""
    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import types  # type: ignore[import-untyped]
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install with: pip install google-genai Pillow") from e

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY in .env (or GOOGLE_API_KEY) or pass api_key=..."
        )

    client = genai.Client(api_key=key)

    if isinstance(image_input, (str, Path)):
        data = Path(image_input).read_bytes()
    else:
        data = image_input
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    ]
    response = client.models.generate_content(model=model_name, contents=contents)

    # Extract text from parts to avoid "non-text parts" warning (e.g. thought_signature)
    if not response or not response.candidates:
        return "[No description returned]"
    parts = response.candidates[0].content.parts
    text_parts = [p.text for p in parts if getattr(p, "text", None)]
    if not text_parts:
        return "[No description returned]"
    return " ".join(text_parts).strip()


def describe_image(
    image_input: str | Path | bytes,
    model_name: ModelName | str,
    prompt: str = "Describe this image in 2–3 concise sentences.",
    *,
    api_key: str | None = None,
) -> str:
    """
    Describe an image using a Gemini vision model.

    Models: gemini-2.0-flash, gemini-1.5-pro, gemini-3-pro-preview, gemini-3-flash-preview

    Args:
        image_input: Path to image file (str/Path) or raw bytes.
        model_name: One of MODEL_NAMES.
        prompt: Text prompt for the vision model.
        api_key: Optional; else uses GEMINI_API_KEY from .env or GOOGLE_API_KEY.

    Returns:
        Description string from the model.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model_name={model_name!r}. Choose one of: {list(MODEL_NAMES)}"
        )
    return describe_image_gemini(image_input, model_name, prompt, api_key=api_key)


# ---------------------------------------------------------------------------
# Class for use in notebooks and scripts
# ---------------------------------------------------------------------------

def _frame_to_bytes(frame) -> bytes:
    """Convert numpy frame (H,W,3) float 0-1 or uint8 0-255 to JPEG bytes."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(frame)
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode="L")
    else:
        pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ExternalLLMImageDescriber:
    """
    Image description via Google Gemini vision API.
    API key is read from .env (GEMINI_API_KEY).
    """

    MODEL_NAMES = MODEL_NAMES

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: ModelName | str = "gemini-2.0-flash",
        default_prompt: str = "Describe this image in 2–3 concise sentences.",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.default_model = default_model
        self.default_prompt = default_prompt

    @property
    def model_names(self) -> tuple[str, ...]:
        """List of supported model names."""
        return MODEL_NAMES

    def describe(
        self,
        image_input: str | Path | bytes,
        model_name: ModelName | str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Describe a single image (file path, Path, or bytes).

        Args:
            image_input: Path to image (str/Path) or raw image bytes.
            model_name: One of MODEL_NAMES; uses default_model if None.
            prompt: Prompt for the vision model; uses default_prompt if None.

        Returns:
            Description string.
        """
        model_name = model_name or self.default_model
        prompt = prompt if prompt is not None else self.default_prompt
        return describe_image(
            image_input,
            model_name,
            prompt,
            api_key=self.api_key,
        )

    def describe_frame(
        self,
        frame,
        model_name: ModelName | str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Describe a single frame (numpy array H,W,3, float 0-1 or uint8 0-255).
        """
        image_bytes = _frame_to_bytes(frame)
        return self.describe(image_bytes, model_name=model_name, prompt=prompt)

    def describe_frames(
        self,
        frames,
        model_name: ModelName | str | None = None,
        prompt: str | None = None,
    ) -> list[str]:
        """
        Describe multiple frames (list or array of numpy frames).
        Returns a list of description strings in the same order.
        """
        import numpy as np
        if hasattr(frames, "shape") and frames.ndim == 4:
            frames = [frames[i] for i in range(frames.shape[0])]
        return [
            self.describe_frame(f, model_name=model_name, prompt=prompt)
            for f in frames
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_based.py <image_path> [model_name]")
        print(f"Models: {', '.join(MODEL_NAMES)}")
        sys.exit(1)

    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAMES[0]

    try:
        describer = ExternalLLMImageDescriber()
        desc = describer.describe(path, model_name=model)
        print(f"[{model}]\n{desc}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
