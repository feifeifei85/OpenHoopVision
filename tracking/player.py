from dataclasses import dataclass
from typing import Any

import numpy as np

from tracking.mcbyte import McByteTracker


@dataclass(slots=True)
class TrackView:
    """Minimal player detection/track view shared by local tracking stages."""

    box: np.ndarray
    score: float
    track_id: int


@dataclass(slots=True)
class PlayerState:
    """Cached court, color, and pose observations for one local track."""

    court_observation: np.ndarray | None = None
    court_position: np.ndarray | None = None
    color: tuple[int, int, int] | None = None
    court_frame: int = -(10**9)
    cached_pose: tuple | None = None


class HolderState:
    """Debounce local ball-holder changes and retain short-lived ownership."""

    def __init__(self, hold_frames: int) -> None:
        self.hold_frames = max(0, int(hold_frames))
        self.reset()

    def reset(self) -> None:
        self.track_id = None
        self.hold_until = -1
        self.score = 0.0
        self.switch_candidate = None

    def active(self, track_id: int | None, frame_index: int) -> bool:
        return self.track_id == track_id and frame_index <= self.hold_until

    def update(
        self,
        match: tuple[int, float] | None,
        active_ids: set[int],
        frame_index: int,
    ) -> None:
        if self.track_id is not None and (self.track_id not in active_ids or frame_index > self.hold_until):
            self.track_id = None
            self.score = 0.0
        if match is None:
            self.switch_candidate = None
            return
        track_id, score = match
        previous = self.switch_candidate
        same_candidate = previous is not None and track_id == previous[0] and previous[3] == frame_index - 1
        switch_score = max(previous[1], score) if same_candidate else score
        switch_streak = previous[2] + 1 if same_candidate else 1
        self.switch_candidate = (track_id, switch_score, switch_streak, frame_index)
        if track_id == self.track_id:
            self.score = (
                0.7 * self.score
                + 0.3 * score
            )
        elif switch_streak >= 4 and (
            self.track_id is None or switch_score >= self.score + 0.08
        ):
            self.track_id = int(track_id)
            self.score = float(switch_score)
            self.switch_candidate = None
        else:
            return
        self.hold_until = frame_index + self.hold_frames


def make_tracker(fps: float, args: Any) -> McByteTracker:
    """Create the local multi-object tracker with project defaults."""
    return McByteTracker(
        lost_track_buffer=args.track_buffer,
        frame_rate=float(fps),
        track_activation_threshold=0.15,
        high_conf_det_threshold=0.15,
        cmc_method='orb',
        cmc_downscale=4,
    )


def detection_scores(
    detections: Any,
    missing_value: float,
    *,
    normalize_nan=True,
    nan_value=0.0,
    flatten=False,
) -> np.ndarray:
    """Normalize model confidence values to a one-dimensional float array."""
    if detections.confidence is None:
        return np.full(len(detections), missing_value, dtype=np.float32)
    scores = np.asarray(detections.confidence, dtype=np.float32)
    if flatten:
        scores = scores.reshape(-1)
    return np.nan_to_num(scores, nan=nan_value) if normalize_nan else scores


def detection_tracks(
    detections: Any,
    *,
    score_default=0.0,
    minimum_score=None,
    tracked=False,
) -> list[TrackView]:
    """Adapt detections to local track views, assigning temporary IDs if needed."""
    if detections is None or len(detections) == 0 or (tracked and detections.tracker_id is None):
        return []
    scores = detection_scores(detections, score_default)
    ids = detections.tracker_id if tracked else range(-1, -len(detections) - 1, -1)
    tracks = []
    for box, score, track_id in zip(detections.xyxy, scores, ids):
        if tracked and (track_id is None or int(track_id) < 0):
            continue
        if minimum_score is not None and float(score) < float(minimum_score):
            continue
        tracks.append(TrackView(np.asarray(box, dtype=np.float32).copy(), float(score), int(track_id)))
    return sorted(tracks, key=lambda track: track.track_id) if tracked else tracks


def match_holder_track(players: list[TrackView], holders: Any, iou_metric: Any) -> tuple[int, float] | None:
    """Match holder detections to local player tracks by IoU and confidence."""
    if not players or holders is None or len(holders) == 0:
        return None
    player_boxes = np.asarray([track.box for track in players], dtype=np.float32)
    iou_scores = iou_metric.compute(player_boxes, holders.xyxy)
    if iou_scores.size == 0:
        return None
    confidence = detection_scores(holders, 1.0, normalize_nan=False)
    scores = np.where(iou_scores >= 0.45, iou_scores * confidence[None, :], -1.0)
    flat_index = int(np.argmax(scores))
    if scores.flat[flat_index] < 0:
        return None
    player_index, holder_index = np.unravel_index(flat_index, scores.shape)
    return players[player_index].track_id, float(scores[player_index, holder_index])


def recover_display_tracks(
    detections: list[TrackView],
    visible_tracks: list[TrackView],
    display_only_tracks: list[TrackView],
    previous_tracks: list[TrackView],
    iou_metric: Any,
) -> list[TrackView]:
    """Reuse prior display IDs for unmatched detections that still overlap."""
    if not detections or not previous_tracks:
        return display_only_tracks
    rendered_tracks = [*visible_tracks, *display_only_tracks]
    rendered_ids = {track.track_id for track in rendered_tracks}
    detection_boxes = np.asarray([track.box for track in detections], dtype=np.float32)
    rendered_boxes = np.asarray([track.box for track in rendered_tracks], dtype=np.float32).reshape(-1, 4)
    overlaps = (
        iou_metric.compute(rendered_boxes, detection_boxes)
        if len(rendered_boxes)
        else np.zeros((0, len(detection_boxes)), dtype=np.float32)
    )
    unmatched_ids = {
        index for index in range(len(detections))
        if not overlaps.size or float(np.max(overlaps[:, index])) < 0.15
    }
    candidates = [track for track in previous_tracks if track.track_id not in rendered_ids]
    if not candidates or not unmatched_ids:
        return display_only_tracks
    detection_ids = sorted(unmatched_ids)
    history_overlap = iou_metric.compute(
        np.asarray([track.box for track in candidates], dtype=np.float32),
        detection_boxes[detection_ids],
    )
    pairs = sorted(
        (
            (float(history_overlap[history_id, column]), history_id, detection_id)
            for history_id in range(len(candidates))
            for column, detection_id in enumerate(detection_ids)
            if float(history_overlap[history_id, column]) >= 0.15
        ),
        reverse=True,
    )
    used_history = set()
    used_detections = set()
    for _, history_id, detection_id in pairs:
        if history_id in used_history or detection_id in used_detections:
            continue
        previous = candidates[history_id]
        detection = detections[detection_id]
        display_only_tracks.append(TrackView(detection.box, detection.score, previous.track_id))
        used_history.add(history_id)
        used_detections.add(detection_id)
    return display_only_tracks


__all__ = [
    "HolderState",
    "PlayerState",
    "TrackView",
    "detection_scores",
    "detection_tracks",
    "make_tracker",
    "match_holder_track",
    "recover_display_tracks",
]
