"""Pose matching and cached pose geometry for local player tracks."""

from typing import Any

import numpy as np


def match_pose_to_players(
    players: list[Any],
    pose_boxes: np.ndarray,
    pose_points: np.ndarray,
    pose_scores: np.ndarray,
    iou_metric: Any,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Associate pose detections with player tracks using one-to-one IoU matches."""
    minimum_iou = 0.1
    if not players or len(pose_boxes) == 0:
        return {}
    player_boxes = np.asarray([track.box for track in players], dtype=np.float32)
    iou_scores = iou_metric.compute(player_boxes, np.asarray(pose_boxes, dtype=np.float32))
    pairs = np.argwhere(iou_scores >= minimum_iou)
    if len(pairs) == 0:
        return {}
    order = np.argsort(-iou_scores[pairs[:, 0], pairs[:, 1]], kind="stable")
    matches = {}
    used_players = set()
    used_poses = set()
    for pair_index in order:
        player_index, pose_index = map(int, pairs[pair_index])
        if player_index in used_players or pose_index in used_poses:
            continue
        matches[players[player_index].track_id] = (pose_points[pose_index], pose_scores[pose_index])
        used_players.add(player_index)
        used_poses.add(pose_index)
    return matches


def validated_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    box: np.ndarray,
    frame_shape: tuple[int, ...] | None = None,
    *,
    jersey_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate pose coordinates against their player box and optional image bounds."""
    if jersey_pose:
        threshold = 0.2
        x_margin = 0.1
        y_margin = 0.1
        min_margin = 4.0
    else:
        threshold = 0.25
        x_margin = 0.25
        y_margin = 0.15
        min_margin = 8.0
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 2 or len(confidence) < len(points):
        return None
    box = np.asarray(box, dtype=np.float32)
    margin = np.maximum(
        min_margin,
        (x_margin, y_margin) * np.maximum(box[2:] - box[:2], 1.0),
    )
    valid = np.isfinite(points[:, :2]).all(axis=1) & np.isfinite(confidence) & (confidence >= threshold)
    valid &= np.all(points[:, :2] >= box[:2] - margin, axis=1)
    valid &= np.all(points[:, :2] <= box[2:] + margin, axis=1)
    if frame_shape is not None:
        bounds = np.asarray((frame_shape[1], frame_shape[0]), dtype=np.float32)
        valid &= np.all(points[:, :2] >= 0, axis=1) & np.all(points[:, :2] < bounds, axis=1)
    return points, confidence, valid


def get_player_ground_point(track: Any, pose: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Estimate the player's court-contact point from visible ankles or box feet."""
    x1, y1, x2, y2 = map(float, track.box)
    box_point = np.asarray(((x1 + x2) * 0.5, y2), dtype=np.float32)
    if pose is None:
        return box_point
    pose_data = validated_pose(pose[0], pose[1], track.box)
    if pose_data is None or len(pose_data[0]) < 17:
        return box_point
    points, _, ankle_mask = pose_data
    ankle_ids = [15, 16]
    ankle_mask = ankle_mask[ankle_ids]
    if not np.any(ankle_mask):
        return box_point
    ankles = points[ankle_ids, :2][ankle_mask]
    contact = np.asarray((ankles[:, 0].mean(), ankles[:, 1].max()), dtype=np.float32)
    contact[1] += min(
        24.0,
        0.12 * max(1.0, y2 - y1),
    )
    return contact


def resolve_pose(
    track: Any,
    state: Any,
    matched_pose: tuple[np.ndarray, np.ndarray] | None,
    frame_index: int,
    max_age: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Use a fresh pose match or rescale a recent cached pose to the current box."""
    current_box = np.asarray(track.box, dtype=np.float32)
    if matched_pose is not None:
        points = np.asarray(matched_pose[0], dtype=np.float32).copy()
        scores = np.asarray(matched_pose[1], dtype=np.float32).copy()
        state.cached_pose = (points, scores, current_box.copy(), int(frame_index))
        return points, scores
    if state.cached_pose is None or frame_index - state.cached_pose[3] > max_age:
        state.cached_pose = None
        return None
    points, scores, old_box, cached_frame = state.cached_pose
    min_dimension = 1.0
    old_size = np.maximum(old_box[2:] - old_box[:2], min_dimension)
    new_size = np.maximum(current_box[2:] - current_box[:2], min_dimension)
    points = current_box[:2] + (points - old_box[:2]) * new_size / old_size
    state.cached_pose = (points, scores, current_box.copy(), cached_frame)
    return points, scores


__all__ = ["get_player_ground_point", "match_pose_to_players", "resolve_pose", "validated_pose"]
