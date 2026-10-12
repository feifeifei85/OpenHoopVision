"""Build validated player crops for appearance matching."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from tracking.pose import validated_pose


def _build_pose_foreground_crop(crop, points, valid, box, origin):
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    points = points[:, :2] - np.asarray(origin, dtype=np.float32)
    width, height = np.asarray(box[2:]) - np.asarray(box[:2])
    torso = np.asarray([5, 6, 11, 12])
    if len(valid) > int(torso.max()) and np.all(valid[torso]):
        polygon = points[torso]
        center = polygon.mean(axis=0)
        polygon = center + 1.18 * (polygon - center)
        polygon = np.clip(polygon, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1))
        cv2.fillConvexPoly(mask, np.rint(polygon[[0, 1, 3, 2]]).astype(np.int32), 255)
    if len(valid) and valid[0]:
        cv2.ellipse(
            mask,
            tuple(np.rint(points[0]).astype(int)),
            (max(3, int(0.105 * width)), max(4, int(0.055 * height))),
            0,
            0,
            360,
            255,
            -1,
        )
    for first, second, radius in (
        (5, 7, 0.055), (7, 9, 0.045), (6, 8, 0.055), (8, 10, 0.045),
        (11, 13, 0.075), (13, 15, 0.06), (12, 14, 0.075), (14, 16, 0.06),
    ):
        if second < len(valid) and valid[first] and valid[second]:
            cv2.line(
                mask,
                tuple(np.rint(points[first]).astype(int)),
                tuple(np.rint(points[second]).astype(int)),
                255,
                max(2, round(width * radius * 2)),
            )
    kernel = max(3, round(width * 0.035))
    kernel += 1 - kernel % 2
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)),
        iterations=1,
    )
    if cv2.countNonZero(mask) < max(30, int(0.03 * mask.size)):
        return crop
    foreground = crop.copy()
    foreground[mask == 0] = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    return foreground


def build_player_crop(
    resolver: Any,
    frame: np.ndarray,
    track: Any,
    pose: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    """Create a validated appearance crop, using pose foreground when available."""
    box = np.asarray(track.box, dtype=np.float32)
    if float(track.score) < float(resolver.parameters.get("min_score", 0.6)):
        return None
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    if (
        height < float(resolver.parameters.get("min_height_px", 56))
        or width < float(resolver.parameters.get("min_width_px", 24))
    ):
        return None
    bounds = (frame.shape[1], frame.shape[0])
    clipped_x1 = max(0, min(bounds[0], int(math.floor(x1))))
    clipped_y1 = max(0, min(bounds[1], int(math.floor(y1))))
    clipped_x2 = max(0, min(bounds[0], int(math.ceil(x2))))
    clipped_y2 = max(0, min(bounds[1], int(math.ceil(y2))))
    visible_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
    if visible_area / max(width * height, 1.0) < float(resolver.parameters.get('min_visible_box_fraction', 0.85)):
        return None

    pose_valid = False
    pose_data = None
    if pose is not None:
        pose_data = validated_pose(
            pose[0],
            pose[1],
            box,
            frame_shape=frame.shape,
        )
        if pose_data is not None:
            points, _, valid = pose_data
            torso_ids = np.asarray([5, 6, 11, 12], dtype=np.int64)
            pose_valid = (
                len(points) > 0
                and float(valid.mean()) >= float(resolver.parameters.get('min_pose_visibility', 0.35))
                and len(valid) > int(torso_ids.max())
                and int(valid[torso_ids].sum()) >= int(resolver.parameters.get('min_visible_torso_points', 2))
            )

    # Pose estimates can disappear on fast, occluded or distant players.
    # For a strong, mostly visible person box, use the full box crop as a
    # fallback so that the local track still has an appearance descriptor.
    if not pose_valid and (
        float(track.score) < float(resolver.parameters.get("fallback_crop_min_score", 0.55))
        or height < float(resolver.parameters.get("fallback_crop_min_height_px", 100))
        or width < float(resolver.parameters.get("fallback_crop_min_width_px", 35))
    ):
        return None
    padding_ratio = float(resolver.parameters.get('crop_padding_ratio', 0.05))
    padding_x = round(width * padding_ratio)
    padding_y = round(height * padding_ratio)
    clipped_x1 = max(0, clipped_x1 - padding_x)
    clipped_y1 = max(0, clipped_y1 - padding_y)
    clipped_x2 = min(bounds[0], clipped_x2 + padding_x)
    clipped_y2 = min(bounds[1], clipped_y2 + padding_y)
    crop = frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    if crop.size and pose_valid and len(pose_data[0]) >= 17:
        crop = _build_pose_foreground_crop(
            crop,
            pose_data[0],
            pose_data[2],
            box,
            (clipped_x1, clipped_y1),
        )
    return crop if crop.size else None


def jersey_crop_overlaps_referee(
    resolver: Any,
    frame: np.ndarray,
    track: Any,
    pose: tuple[np.ndarray, np.ndarray] | None,
    referee_boxes: Sequence[Sequence[float]],
) -> bool:
    """Check whether the configured jersey-number crop intersects a referee box."""
    if len(referee_boxes) == 0:
        return False
    jersey_parameters = resolver.parameters.get("jersey_ocr", {})
    crops = jersey_torso_crop(
        frame, track.box, pose, jersey_parameters, include_bounds=True
    )
    if crops is None:
        return False
    pad = int(jersey_parameters.get("referee_clearance_px", 8))
    return crop_bounds_overlap(crops["bounds"], referee_boxes, pad)


def crop_bounds_overlap(
    bounds: Sequence[float],
    boxes: Sequence[Sequence[float]],
    padding: float = 0,
) -> bool:
    """Return whether crop bounds overlap any detection box after padding."""
    x1, y1, x2, y2 = map(float, bounds)
    pad = float(padding)
    return any(
        x1 < rx2 + pad and x2 > rx1 - pad and y1 < ry2 + pad and y2 > ry1 - pad
        for rx1, ry1, rx2, ry2 in boxes
    )


def jersey_torso_crop(
    frame: np.ndarray,
    box: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray] | None,
    parameters: Mapping[str, Any] | None = None,
    include_bounds: bool = False,
) -> dict[str, Any] | None:
    """Crop the central jersey number using shoulder and hip keypoints."""
    parameters = parameters or {}
    box = np.asarray(box, dtype=np.float32)
    image_h, image_w = frame.shape[:2]
    min_width = int(parameters.get("min_crop_width", 14))
    min_height = int(parameters.get("min_crop_height", 14))
    min_area = int(parameters.get("min_crop_area", 196))

    def number_crop(center_x, center_y, side):
        side = max(min_width, int(round(side)))
        x1 = int(round(center_x - side / 2))
        y1 = int(round(center_y - side / 2))
        x1 = min(max(0, x1), max(0, image_w - side))
        y1 = min(max(0, y1), max(0, image_h - side))
        x2, y2 = min(image_w, x1 + side), min(image_h, y1 + side)
        crop_width, crop_height = x2 - x1, y2 - y1
        if crop_width < min_width or crop_height < min_height or crop_width * crop_height < min_area:
            return None
        result = {"full": frame[y1:y2, x1:x2]}
        if include_bounds:
            result["bounds"] = (x1, y1, x2, y2)
        return result

    def box_crop():
        height = box[3] - box[1]
        torso_height = height * 0.30
        return number_crop(
            float((box[0] + box[2]) / 2),
            float(box[1] + height * 0.35),
            torso_height * 0.8,
        )

    if pose is None:
        return box_crop()
    points = np.asarray(pose[0], dtype=np.float32)
    scores = np.asarray(pose[1], dtype=np.float32).reshape(-1)
    joints = (5, 6, 11, 12)
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or len(scores) <= max(joints)
        or len(points) <= max(joints)
        or not np.isfinite(points[list(joints), :2]).all()
        or not np.isfinite(scores[list(joints)]).all()
        or np.any(scores[list(joints)] < float(parameters.get("pose_confidence", 0.2)))
    ):
        return box_crop()
    torso = points[list(joints), :2]
    shoulder, hips = points[[5, 6], :2], points[[11, 12], :2]
    torso_height = max(1.0, float(hips[:, 1].max() - shoulder[:, 1].min()))
    number_center_x = float(torso[:, 0].mean())
    number_center_y = float(shoulder[:, 1].min() + torso_height * 0.5)
    return number_crop(number_center_x, number_center_y, torso_height * 0.8) or box_crop()
