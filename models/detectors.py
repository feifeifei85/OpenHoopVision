from typing import Any

import numpy as np
import supervision as sv


def predict_shape(frame: np.ndarray, short_side: int | None = None) -> tuple[int, int]:
    """Choose detector input dimensions while preserving the frame aspect ratio."""
    short_side = 512 if short_side is None else short_side
    stride = 32
    height, width = frame.shape[:2]
    scale = min(1.0, short_side / min(height, width), 1333 / max(height, width))
    return max(stride, int(height * scale) // stride * stride), max(
        stride, int(width * scale) // stride * stride
    )


def resolve_detector_class_ids(model: Any) -> dict[str, tuple[int, ...]]:
    """Map normalized detector class names to all matching model class IDs."""
    names = getattr(model, "class_names", {})
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    class_map = {}
    for class_id, name in entries:
        key = str(name).lower().replace("_", " ").strip()
        class_map.setdefault(key, []).append(int(class_id))
    return {name: tuple(ids) for name, ids in class_map.items()}


def _class_mask(class_ids, ids, name):
    target_ids = class_ids.get(name, ())
    if not target_ids:
        return np.zeros(ids.shape, dtype=bool)
    return ids == target_ids[0] if len(target_ids) == 1 else np.isin(ids, target_ids)


def detect(
    model: Any,
    frame_rgb: np.ndarray,
    shape: tuple[int, int],
    class_map: dict[str, tuple[int, ...]],
    player_threshold: float,
    ball_threshold: float,
    rim_threshold: float = 0.5,
    holder_threshold: float = 0.38,
    scorer_threshold: float = 0.38,
    referee_threshold: float = 0.54,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run detector inference and partition results by basketball class."""
    detections = model.predict(
        frame_rgb, threshold=0.1, shape=shape, include_source_image=False
    )
    if detections is None or len(detections) == 0 or detections.class_id is None:
        empty = sv.Detections.empty()
        return empty, empty, empty, empty, empty, empty
    detected_ids = np.asarray(detections.class_id, dtype=np.int32)
    scores = (
        np.ones(len(detected_ids), dtype=np.float32)
        if detections.confidence is None
        else np.asarray(detections.confidence, dtype=np.float32)
    )
    player_mask = _class_mask(class_map, detected_ids, "player") & (scores >= float(player_threshold))
    ball_mask = _class_mask(class_map, detected_ids, "ball") & (scores >= float(ball_threshold))
    rim_mask = _class_mask(class_map, detected_ids, "rim") & (scores >= float(rim_threshold))
    holder_mask = _class_mask(class_map, detected_ids, "holder") & (scores >= float(holder_threshold))
    scorer_mask = _class_mask(class_map, detected_ids, "scorer") & (scores >= float(scorer_threshold))
    referee_mask = _class_mask(class_map, detected_ids, "referee") & (scores >= float(referee_threshold))
    return (
        detections[player_mask],
        detections[ball_mask],
        detections[rim_mask],
        detections[holder_mask],
        detections[scorer_mask],
        detections[referee_mask],
    )


__all__ = ["detect", "predict_shape", "resolve_detector_class_ids"]

