# Utilities: entity extraction, display helpers, evaluation metrics, plotting, video frames, frame describe
from vllmd.utils.video_frames import (
    extract_frames_from_video,
    split_abnormal_videos_into_frames,
)
from vllmd.utils.video_describe import describe_videos_in_abnormal
from vllmd.utils.json_parser import (
    load_video_descriptions_json,
    load_all_video_descriptions,
)
from vllmd.utils.frame_describe import (
    VISUAL_DESCRIPTION_PROMPT,
    describe_frame_images_in_folder,
    extract_named_entities_spacy,
)
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
    "extract_frames_from_video",
    "split_abnormal_videos_into_frames",
    "describe_videos_in_abnormal",
    "load_video_descriptions_json",
    "load_all_video_descriptions",
    "VISUAL_DESCRIPTION_PROMPT",
    "describe_frame_images_in_folder",
    "extract_named_entities_spacy",
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
