"""
LLM-based frame description (Ollama and BLIP).
"""

from __future__ import annotations

import textwrap
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch  # pyright: ignore[reportMissingImports]

# Defaults for VLM description (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"


class FrameDescriber:
    """Wraps Ollama and BLIP-based frame description after prediction."""

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
    ) -> None:
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _ollama_describe(
        self,
        prompt: str,
        model_name: str | None = None,
        max_tokens: int = 180,
    ) -> str:
        """Get a short description from local Ollama."""
        model_name = model_name or self.ollama_model
        try:
            r = requests.post(
                self.ollama_url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"

    def describe_frames_after_predict_each_frame(
        self,
        frame_indices: np.ndarray,
        predictions: list[str],
        probs: np.ndarray,
        frames: np.ndarray,
        num_frames_to_show: int = 5,
        use_ollama: bool = True,
        ollama_model: str | None = None,
        model_name: str = "Salesforce/blip2-opt-2.7b-coco",
        max_length: int = 150,
        frames_original: np.ndarray | None = None,
    ) -> list[str]:
        """
        Describe frames (from predict_each_frame) using Ollama or BLIP.
        use_ollama=True: Ollama text prompt from prediction label.
        use_ollama=False: Hugging Face image-to-text (BLIP) per frame.
        If frames_original is provided (same length as frames), display uses
        original-resolution frames without resizing.
        """
        from PIL import Image  # noqa: PLC0415

        ollama_model = ollama_model or self.ollama_model
        n = len(frame_indices)
        if n == 0:
            raise ValueError("No frame indices from predict_each_frame")
        show_indices = np.linspace(0, n - 1, min(num_frames_to_show, n), dtype=int)
        frame_indices_sub = [frame_indices[i] for i in show_indices]
        predictions_sub = [predictions[i] for i in show_indices]
        frames_to_describe = [frames[fi] for fi in frame_indices_sub]
        # Use original-size frames for display when provided and length matches
        display_frames = (
            [frames_original[fi] for fi in frame_indices_sub]
            if frames_original is not None
            and len(frames_original) == len(frames)
            else frames_to_describe
        )

        descriptions: list[str] = []
        if use_ollama:
            for pred in predictions_sub:
                temp=[]
                prompt = (
                    f'In 3 short sentences, what might be happening in a video and speciffy if the action is happening  in an indoor or outdoor environment'
                    f'frame classified as "{pred}"?'
                )
                desc = self._ollama_describe(
                    prompt, model_name=ollama_model, max_tokens=max_length
                )
                descriptions.append(desc)
        else:
            from transformers import pipeline  # pyright: ignore[reportMissingImports]

            pipe = pipeline(
                "image-to-text",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
            for frame in frames_to_describe:
                img_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, "RGB")
                out = pipe(pil_img, max_new_tokens=max_length)
                desc = (
                    out[0]["generated_text"]
                    if isinstance(out[0], dict)
                    else str(out[0])
                )
                temp.append(desc)
                temp.append(frame)
                descriptions.append(desc)

        num_show = len(display_frames)
        h, w = display_frames[0].shape[:2]
        figwidth = 6 * num_show
        figheight = 6 * (h / w) if w else 6
        fig, axes = plt.subplots(1, num_show, figsize=(figwidth, figheight))
        if num_show == 1:
            axes = [axes]
        for i, (frame, fi, pred, desc) in enumerate(
            zip(display_frames, frame_indices_sub, predictions_sub, descriptions)
        ):
            axes[i].imshow(frame)
            #wrapped_desc = textwrap.fill(desc, width=18)
            axes[i].set_title(
               # f"Frame {fi}\nPred: {pred}\n{wrapped_desc}",
                f"Frame {fi}",
                fontsize=9,
            )
            axes[i].axis("off")
        plt.suptitle(
            "Frames after predict_each_frame: prediction + VLM description",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()
        return descriptions
