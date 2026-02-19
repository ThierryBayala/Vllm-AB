# Utilities: entity extraction, display helpers, evaluation metrics, plotting
from vllmd.utils.entity_extractor import (
    Entity,
    CustomEntityRule,
    CustomEntityExtractor,
    load_rules_from_file,
)
from vllmd.utils.display import (
    frame_to_base64,
    frames_descriptions_to_html,
    display_frames_with_descriptions,
)
from vllmd.utils.metrics import (
    classification_metrics,
    top_k_accuracy,
    get_confusion_matrix,
    evaluate_predictions,
)
from vllmd.utils.plotting import (
    plot_learning_curves,
    plot_metrics,
    plot_confusion_matrix,
)

__all__ = [
    "Entity",
    "CustomEntityRule",
    "CustomEntityExtractor",
    "load_rules_from_file",
    "frame_to_base64",
    "frames_descriptions_to_html",
    "display_frames_with_descriptions",
    "classification_metrics",
    "top_k_accuracy",
    "get_confusion_matrix",
    "evaluate_predictions",
    "plot_learning_curves",
    "plot_metrics",
    "plot_confusion_matrix",
]
