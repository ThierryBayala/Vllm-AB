# LLM-based image/frame description: external APIs (Gemini) and local (Ollama, BLIP)
import warnings
warnings.filterwarnings("ignore")
from vllmd.llm.llm_based import (
    MODEL_NAMES,
    ExternalLLMImageDescriber,
    describe_image,
    describe_image_gemini,
)
from vllmd.llm.local import FrameDescriber, OLLAMA_MODEL, OLLAMA_URL

__all__ = [
    "MODEL_NAMES",
    "ExternalLLMImageDescriber",
    "describe_image",
    "describe_image_gemini",
    "FrameDescriber",
    "OLLAMA_URL",
    "OLLAMA_MODEL",
]
