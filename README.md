# VLLMD — Video Action Recognition

A multimodal system supporting early identification of substance-related risk behaviors in school environments (CNN + LSTM + LLM-based frame description).

## Project structure (best practices)

```
Vllm/
├── src/
│   └── vllmd/                 # Main package (installable)
│       ├── llm/               # LLM-based description (Gemini, Ollama, BLIP)
│       ├── video_processing/  # Data loading, models, pipeline
│       └── utils/             # Entity extraction and helpers
├── data/                      # Datasets and config
│   ├── rules/                 # Entity rules (entity_rules.txt)
│   ├── abnormal/              # Abnormal behavior videos by class
│   ├── normal/                # Normal videos
│   └── test/                  # Test videos
├── models/                    # Saved model checkpoints
├── notebooks/                 # Jupyter notebooks
│   └── vllmd.ipynb
├── tests/                     # Pytest tests
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment and install in editable mode:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -e .
   ```

2. **Optional dependencies:**

   - Gemini API: `pip install google-genai`
   - BLIP / Hugging Face: `pip install transformers`

3. **Environment:** Copy `.env.example` to `.env` and set `GEMINI_API_KEY` if using Gemini. To avoid any cache folders in the project (e.g. `__pycache__`, `.ruff_cache`, `.pytest_cache`), see the optional variables in `.env.example`.

## Usage

- **Notebooks:** Open `notebooks/vllmd.ipynb`. The first cell sets `PROJECT_ROOT` and imports from `vllmd`. Place your videos under `data/abnormal/`, `data/normal/`, `data/test/` (see `data/README.md`).
- **From code:** `from vllmd import VideoDataProcessor, ActionRecognitionPipeline, ...`

## Naming conventions

- **Packages and modules:** lowercase, underscores (`video_processing`, `entity_extractor`).
- **Scripts and notebooks:** lowercase (`vllmd.ipynb`).
- **Data and artifacts:** `data/`, `models/` at project root.
