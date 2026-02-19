# VLLMD — Video Action Recognition

A multimodal system supporting early identification of substance-related risk behaviors in school environments (CNN + LSTM + LLM-based frame description).

## Project structure

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
├── results/                   # Outputs: figures, predictions, descriptions (from notebook)
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

2. **Dependencies:**

   - Gemini API: `pip install google-genai`
   - Hugging Face: `pip install transformers`

3. **Environment:** Set `GEMINI_API_KEY` if using Gemini in `.env`. 
## Usage

- **Notebooks:** Open `notebooks/vllmd.ipynb`. The first cell sets `PROJECT_ROOT` and imports from `vllmd`. Place your videos under `data/abnormal/`, `data/normal/`, `data/test/` (see `data/README.md`). All outputs (figures, training history, predictions, descriptions) are saved under `results/`.
- **From code:** `from vllmd import VideoDataProcessor, ActionRecognitionPipeline, ...`
