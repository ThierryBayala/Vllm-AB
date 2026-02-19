"""
Plotting utilities for learning curves and evaluation metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Use Times New Roman for all figure text
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def plot_learning_curves(
    history: Union[Dict[str, List[float]], List[Dict[str, float]]],
    metrics: Optional[List[Tuple[str, str]]] = None,
    figsize: tuple[float, float] = (10, 4),
    title: Optional[str] = "Learning curves",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot training/validation metrics over epochs.

    Args:
        history: Either a dict of lists (e.g. {"train_loss": [...], "val_loss": [...], ...})
                 or a list of per-epoch dicts (e.g. [{"train_loss": 0.5, "val_loss": 0.6}, ...]).
        metrics: Pairs to plot as (train_key, val_key). Default: loss and accuracy.
                 E.g. [("train_loss", "val_loss"), ("train_acc", "val_acc")].
        figsize: Figure size (width, height).
        title: Figure title.
        save_path: If set, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    # Normalize to dict of lists
    if isinstance(history, list) and history and isinstance(history[0], dict):
        keys = list(history[0].keys())
        history = {k: [h[k] for h in history] for k in keys}
    elif not isinstance(history, dict):
        raise TypeError("history must be a dict of lists or a list of dicts")

    if metrics is None:
        metrics = []
        if "train_loss" in history and "val_loss" in history:
            metrics.append(("train_loss", "val_loss"))
        if "train_acc" in history and "val_acc" in history:
            metrics.append(("train_acc", "val_acc"))
        if not metrics:
            # Plot whatever keys we have (assume first half = train, second = val by convention)
            keys = [k for k in history if not k.startswith("_")]
            if keys:
                metrics = [(k, k.replace("train_", "val_")) for k in keys if k.startswith("train_")]
            if not metrics:
                metrics = [(k, k) for k in list(history.keys())[:2]]

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots, figsize[1]))
    if n_plots == 1:
        axes = [axes]
    epochs = range(1, len(next(iter(history.values()))) + 1)

    for ax, (train_key, val_key) in zip(axes, metrics):
        if train_key in history:
            ax.plot(epochs, history[train_key], label=train_key.replace("_", " ").title(), marker="o", markersize=3)
        if val_key in history:
            ax.plot(epochs, history[val_key], label=val_key.replace("_", " ").title(), marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_key.replace("train_", "").replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory of path if it has one and does not exist."""
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_metrics(
    metrics_dict: Dict[str, Any],
    metric_keys: Optional[List[str]] = None,
    figsize: tuple[float, float] = (6, 4),
    title: Optional[str] = "Classification metrics",
    bar_color: str = "steelblue",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot classification metrics (e.g. from classification_metrics or evaluate_predictions) as a bar chart.

    Args:
        metrics_dict: Dict with keys such as "accuracy", "precision", "recall", "f1".
        metric_keys: Which keys to plot; default: ["accuracy", "precision", "recall", "f1"].
        figsize: Figure size.
        title: Figure title.
        bar_color: Bar color.
        save_path: If set, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1"]
    keys = [k for k in metric_keys if k in metrics_dict and isinstance(metrics_dict[k], (int, float))]
    if not keys:
        raise ValueError(f"No plottable metrics in {metric_keys}; got keys: {list(metrics_dict.keys())}")
    values = [float(metrics_dict[k]) for k in keys]
    labels = [k.replace("_", " ").title() for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[float]]],
    class_names: Optional[List[str]] = None,
    figsize: tuple[float, float] = (8, 6),
    title: str = "Confusion matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (rows = true, columns = predicted).
        class_names: Labels for classes; default "0", "1", ...
        figsize: Figure size.
        title: Figure title.
        cmap: Colormap name.
        save_path: If set, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    if len(class_names) != n:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            thresh = (cm.max() + cm.min()) / 2.0 if cm.max() > 0 else 0.5
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
