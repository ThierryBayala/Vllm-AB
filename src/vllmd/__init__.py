"""
VLLMD — Video action recognition with LLM-based description.

Multimodal support for early identification of substance-related risk
behaviors in school environments.
"""

import warnings

warnings.filterwarnings("ignore")

from vllmd.llm import (
    MODEL_NAMES,
    ExternalLLMImageDescriber,
    describe_image,
    describe_image_gemini,
    FrameDescriber,
    OLLAMA_URL,
    OLLAMA_MODEL,
)
from vllmd.video_processing import (
    ActionRecognitionPipeline,
    VideoDataProcessor,
    VideoDataset,
    describe_frames_after_predict_each_frame,
)
from vllmd.utils import (
    Entity,
    CustomEntityRule,
    CustomEntityExtractor,
    load_rules_from_file,
)

__all__ = [
    "MODEL_NAMES",
    "ExternalLLMImageDescriber",
    "describe_image",
    "describe_image_gemini",
    "FrameDescriber",
    "OLLAMA_URL",
    "OLLAMA_MODEL",
    "ActionRecognitionPipeline",
    "VideoDataProcessor",
    "VideoDataset",
    "describe_frames_after_predict_each_frame",
    "Entity",
    "CustomEntityRule",
    "CustomEntityExtractor",
    "load_rules_from_file",
]

__version__ = "0.1.0"
