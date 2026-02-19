"""
Model architecture, training, and prediction for video action recognition.
"""

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import torchvision.models as models  # pyright: ignore[reportMissingImports]
from torch.optim import Adam  # pyright: ignore[reportMissingImports]

from vllmd.llm.local import (
    FrameDescriber,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

_default_describer = FrameDescriber()


def describe_frames_after_predict_each_frame(
    frame_indices: np.ndarray,
    predictions: list[str],
    probs: np.ndarray,
    frames: np.ndarray,
    num_frames_to_show: int = 5,
    use_ollama: bool = True,
    ollama_model: str = OLLAMA_MODEL,
    model_name: str = "Salesforce/blip2-opt-2.7b-coco",
    max_length: int = 150,
    frames_original: np.ndarray | None = None,
) -> list[str]:
    """Describe frames using Ollama or BLIP. Delegates to vllmd.llm.local.FrameDescriber.
    Pass frames_original to display predicted frames at original video resolution."""
    return _default_describer.describe_frames_after_predict_each_frame(
        frame_indices=frame_indices,
        predictions=predictions,
        probs=probs,
        frames=frames,
        num_frames_to_show=num_frames_to_show,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        model_name=model_name,
        max_length=max_length,
        frames_original=frames_original,
    )


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.embed(positions)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int = 40,
        num_heads: int = 2,
        ff_dim: int = 32,
    ):
        super().__init__()
        self.proj = nn.Linear(dim, 64)
        self.pos = PositionalEmbedding(seq_len, 64)
        self.norm1 = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, ff_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos(x)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        x = self.norm2(x)
        return self.ff(x)


class VideoActionModel(nn.Module):
    """CNN (EfficientNet) + Temporal Transformer + LSTM for video action classification."""

    def __init__(
        self,
        num_classes: int,
        frame_size: int = 64,
        num_frames: int = 40,
    ):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        backbone.classifier = nn.Sequential(nn.Identity())
        self.backbone = backbone
        self.feature_dim = 1280
        self.temporal = TemporalTransformer(
            self.feature_dim, seq_len=num_frames, num_heads=2, ff_dim=8
        )
        self.lstm = nn.LSTM(8, 64, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)
        x = self.temporal(features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class ActionRecognitionPipeline:
    """Wraps model, training, and prediction for video action recognition."""

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        frame_size: int = 64,
        num_frames: int = 40,
    ):
        self.num_classes = num_classes
        self.device = device
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.model = VideoActionModel(
            num_classes=num_classes,
            frame_size=frame_size,
            num_frames=num_frames,
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        save_path: str,
        lr: float = 1e-3,
    ) -> None:
        """Train the model and save best checkpoint by validation loss."""
        optimizer = Adam(self.model.parameters(), lr=lr)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for videos, labels in train_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(videos)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_total += labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for videos, labels in val_loader:
                    videos, labels = videos.to(self.device), labels.to(self.device)
                    logits = self.model(videos)
                    loss = self.criterion(logits, labels)
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_val = val_loss / len(val_loader)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), save_path)
            print(
                f"Epoch {epoch+1}/{epochs}  train_loss={train_loss/len(train_loader):.4f}  "
                f"train_acc={train_acc:.4f}  val_loss={avg_val:.4f}  val_acc={val_acc:.4f}"
            )

    def load(self, path: str) -> None:
        """Load model state from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path: str) -> None:
        """Save model state to path."""
        torch.save(self.model.state_dict(), path)

    def predict_each_frame(
        self,
        video_path: str,
        processor: Any,
        class_names: list[str],
        window_size: int = 40,
        frame_size: Optional[int] = None,
        max_frames: int = 300,
        batch_size: int = 8,
    ) -> tuple:
        """
        Run model on each sliding window of a single video.

        Returns:
            frame_indices: 1D array, last frame index of each window
            predictions: list of class names per window
            probs: (num_windows, num_classes) probabilities
            frames: (num_frames, H, W, 3) array of resized frames used by the model
            frames_original: (num_frames, H_orig, W_orig, 3) array of frames at original size
        """
        fs = frame_size if frame_size is not None else self.frame_size
        frames = processor.extract_frames_for_sliding(
            video_path, frame_size=fs, max_frames=max_frames
        )
        frames_original = processor.extract_frames_for_sliding_original(
            video_path, max_frames=max_frames
        )
        if len(frames) == 0:
            raise ValueError(
                f"Video produced no frames (path={video_path}). "
                "Check that the path is correct and the video is readable."
            )
        if len(frames) < window_size:
            pad = np.tile(
                frames[-1:], (window_size - len(frames), 1, 1, 1)
            )
            frames = np.concatenate([frames, pad], axis=0)
        if len(frames_original) > 0 and len(frames_original) < len(frames):
            pad_orig = np.tile(
                frames_original[-1:], (len(frames) - len(frames_original), 1, 1, 1)
            )
            frames_original = np.concatenate([frames_original, pad_orig], axis=0)

        num_windows = len(frames) - window_size + 1
        if num_windows <= 0:
            raise ValueError(
                f"Not enough frames for a single window: got {len(frames)} frames, "
                f"need at least {window_size} (window_size={window_size})."
            )
        all_logits = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, num_windows, batch_size):
                end = min(start + batch_size, num_windows)
                batch = np.stack(
                    [frames[i : i + window_size] for i in range(start, end)]
                )
                x = (
                    torch.from_numpy(batch)
                    .float()
                    .permute(0, 1, 4, 2, 3)
                    .to(self.device)
                )
                logits = self.model(x)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        pred_indices = logits.argmax(dim=1).numpy()
        frame_indices = np.arange(
            window_size - 1, window_size - 1 + num_windows, dtype=np.int64
        )
        predictions = [class_names[i] for i in pred_indices]
        return frame_indices, predictions, probs, frames, frames_original

    def show_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: list[str],
        num_samples: int = 5,
    ) -> None:
        """Plot random test samples with true vs predicted labels."""
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[assignment, misc]

        self.model.eval()
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        fig, axes = plt.subplots(1, num_samples, figsize=(25, 8))
        if num_samples == 1:
            axes = [axes]
        with torch.no_grad():
            for i, idx in enumerate(indices):
                video = X_test[idx]
                true_label = class_names[y_test[idx]]
                x = (
                    torch.from_numpy(video)
                    .float()
                    .permute(0, 3, 1, 2)
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.model(x)
                pred_idx = logits.argmax(1).item()
                predicted_label = class_names[pred_idx]
                mid_frame = (video[len(video) // 2] * 255).astype(np.uint8)
                if Image is not None:
                    image = Image.fromarray(mid_frame, "RGB")
                   # image.save("saved_image.png")
                axes[i].imshow(mid_frame)
                axes[i].set_title(
                    f"True: {true_label}\nPred: {predicted_label}", fontsize=30
                )
                axes[i].axis("off")
        plt.suptitle("Model Predictions on Test Videos", fontsize=30)
        plt.tight_layout()
        plt.show()
