import math
from collections import deque
from dataclasses import dataclass


def normalize_fps(fps: float | None) -> float:
    """Normalize missing, invalidly low, or measured video rates to a usable FPS."""
    return max(float(fps or 30.0), 1.0)


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """One detected ball or rim box in image coordinates."""

    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) * 0.5


@dataclass(frozen=True, slots=True)
class BallTrack:
    """Smoothed ball position, velocity, observation, and interpolation state."""

    x: float
    y: float
    vx: float
    vy: float
    interpolated: bool
    bbox: tuple[float, float, float, float] | None = None
    velocity_valid: bool = True
    confidence: float | None = None
    track_id: int = 0


def _reselect_shooter(tracker, detections, t, shooter_context):
    confirm_seconds = tracker.parameters.get('shooter_reselect_confirm_seconds', 0.2)
    if not shooter_context or not detections:
        if (
            tracker.shooter_candidate is not None
            and t - tracker.shooter_candidate[1] > confirm_seconds
        ):
            tracker.shooter_candidate = None
        return None

    player_id = shooter_context.get("player_id")
    box = shooter_context.get("box")
    if player_id is None or box is None or len(box) < 4:
        tracker.shooter_candidate = None
        return None
    x1, y1, x2, y2 = map(float, box[:4])
    max_distance = tracker.parameters.get('shooter_reselect_player_distance_px', 80.0)
    candidates = []
    for detection in detections:
        box_distance = math.hypot(
            max(x1 - detection.cx, 0.0, detection.cx - x2),
            max(y1 - detection.cy, 0.0, detection.cy - y2),
        )
        if (
            detection.confidence >= tracker.parameters.get('shooter_reselect_min_confidence', 0.4)
            and detection.cy < y1 - tracker.parameters.get('shooter_reselect_head_margin_px', 10.0)
            and box_distance <= max_distance
        ):
            candidates.append(detection)
    if not candidates:
        if (
            tracker.shooter_candidate is not None
            and t - tracker.shooter_candidate[1] > confirm_seconds
        ):
            tracker.shooter_candidate = None
        return None

    candidate = max(candidates, key=lambda item: item.confidence)
    pending = tracker.shooter_candidate
    tracker.shooter_candidate = (str(player_id), t, candidate)
    if pending is None or pending[0] != str(player_id):
        return None

    elapsed = t - pending[1]
    if elapsed <= 0 or elapsed > confirm_seconds:
        return None
    previous = pending[2]
    vx = (candidate.cx - previous.cx) / elapsed
    vy = (candidate.cy - previous.cy) / elapsed
    step = math.hypot(candidate.cx - previous.cx, candidate.cy - previous.cy)
    if (
        step > tracker.parameters.get('shooter_reselect_max_step_px', 90.0)
        or vy > -tracker.parameters.get('shooter_reselect_min_up_px_s', 120.0)
        or abs(vx) > tracker.parameters.get('shooter_reselect_max_horiz_px_s', 1600.0)
    ):
        return None

    tracker.shooter_candidate = None
    if tracker.track is not None and tracker.last_t is not None:
        dt = max(t - tracker.last_t, 0.0)
        step = min(dt, 0.25)
        predicted = (tracker.track.x + tracker.track.vx * step, tracker.track.y + tracker.track.vy * step)
        if math.dist(predicted, (candidate.cx, candidate.cy)) <= tracker.parameters.get(
            'shooter_reselect_switch_distance_px', 80.0
        ):
            return None

    tracker.track = BallTrack(
        candidate.cx, candidate.cy, vx, vy, False, candidate.bbox, True, candidate.confidence, tracker._new_id()
    )
    tracker.last_observed = (candidate.cx, candidate.cy, t)
    tracker.last_t = t
    tracker.gap = 0
    tracker.reseed_anchor = None
    tracker.trail.clear()
    tracker.trail.append((round(candidate.cx), round(candidate.cy)))
    return tracker.track


class ShotBallTracker:
    """Track detected ball boxes and bridge short occlusions."""

    def __init__(
        self,
        fps: float,
        trail_length: int | None = None,
        max_gap_frames: int | None = None,
        parameters: dict | None = None,
    ) -> None:
        self.fps = normalize_fps(fps)
        self.parameters = parameters or {}
        self.max_gap = max(
            1,
            int(max_gap_frames if max_gap_frames is not None else self.parameters.get("max_gap_frames", 5)),
        )
        self.track = None
        self.last_t = None
        self.last_observed = None
        self.gap = 0
        self.reseed_anchor = None
        self.next_id = 1
        trail_length = 20 if trail_length is None else trail_length
        self.trail = deque(maxlen=max(1, int(trail_length)))
        self.shooter_candidate = None

    def reset(self):
        self.track = None
        self.last_t = None
        self.last_observed = None
        self.gap = 0
        self.reseed_anchor = None
        self.trail.clear()
        self.shooter_candidate = None

    def update(self, detections, t, shooter_context=None):
        t = float(t)
        velocity_history_seconds = self.parameters.get('velocity_history_max_seconds', 1.0)
        dt = None if self.last_t is None else t - self.last_t
        if dt is not None and dt <= 0:
            dt = None
        detections = list(detections or [])
        reselected = self._shooter_reselect(detections, t, shooter_context)
        if reselected is not None:
            return reselected
        matched = self._match(detections, dt)

        if matched is not None:
            previous = self.track
            x, y = matched.cx, matched.cy
            bbox = matched.bbox
            if previous is not None and dt is not None:
                step = min(dt, 0.25)
                predicted_x = previous.x + previous.vx * step
                predicted_y = previous.y + previous.vy * step
                weight = 0.7
                x = predicted_x + weight * (matched.cx - predicted_x)
                y = predicted_y + weight * (matched.cy - predicted_y)
                dx, dy = x - matched.cx, y - matched.cy
                bbox = (
                    matched.bbox[0] + dx,
                    matched.bbox[1] + dy,
                    matched.bbox[2] + dx,
                    matched.bbox[3] + dy,
                )
            vx = vy = 0.0
            valid = False
            if self.last_observed is not None:
                ox, oy, ot = self.last_observed
                elapsed = t - ot
                if 0 < elapsed <= velocity_history_seconds:
                    raw_vx = (x - ox) / elapsed
                    raw_vy = (y - oy) / elapsed
                    if previous is not None and previous.velocity_valid and self.gap == 0:
                        alpha = self.parameters.get('velocity_smoothing', 0.5)
                        vx = alpha * raw_vx + (1.0 - alpha) * previous.vx
                        vy = alpha * raw_vy + (1.0 - alpha) * previous.vy
                    else:
                        vx, vy = raw_vx, raw_vy
                    valid = True
            track_id = previous.track_id if previous is not None else self._new_id()
            self.track = BallTrack(x, y, vx, vy, False, bbox, valid, matched.confidence, track_id)
            self.last_observed = (x, y, t)
            self.gap = 0
            self.reseed_anchor = None
            self.last_t = t
            self.trail.append((round(x), round(y)))
            return self.track

        if self.track is not None:
            self.gap += 1
            step = dt if dt is not None else 1.0 / self.fps
            if self.gap <= self.max_gap:
                self.track = BallTrack(
                    self.track.x + self.track.vx * step,
                    self.track.y + self.track.vy * step,
                    self.track.vx,
                    self.track.vy,
                    True,
                    None,
                    self.track.velocity_valid,
                    None,
                    self.track.track_id,
                )
                self.last_t = t
                self.trail.append((round(self.track.x), round(self.track.y)))
                return self.track
            ax, ay, at = self.last_observed
            self.reseed_anchor = (
                ax,
                ay,
                at,
                self.track.vx if self.track.velocity_valid else 0.0,
                self.track.vy if self.track.velocity_valid else 0.0,
            )
            self.track = None
            self.last_t = t
            self.trail.clear()
            return None

        self.last_t = t
        predicted = None
        if self.reseed_anchor is not None:
            ax, ay, at, avx, avy = self.reseed_anchor
            elapsed = t - at
            if elapsed <= 0 or elapsed > velocity_history_seconds:
                self.reseed_anchor = None
            else:
                radius = self.parameters.get('gate_base_px', 45.0) + self.parameters.get(
                    'reseed_speed_px_s', 2500.0
                ) * elapsed
                predicted = (ax + avx * elapsed, ay + avy * elapsed)
                detections = [d for d in detections if math.hypot(d.cx - ax, d.cy - ay) <= radius]
                if not detections:
                    return None
        if not detections:
            return None
        seed = (
            max(detections, key=lambda item: item.confidence)
            if predicted is None
            else min(detections, key=lambda item: math.hypot(item.cx - predicted[0], item.cy - predicted[1]))
        )
        self.track = BallTrack(seed.cx, seed.cy, 0.0, 0.0, False, seed.bbox, False, seed.confidence, self._new_id())
        self.last_observed = (seed.cx, seed.cy, t)
        self.gap = 0
        self.reseed_anchor = None
        self.trail.append((round(seed.cx), round(seed.cy)))
        return self.track

    def _shooter_reselect(self, detections, t, shooter_context):
        return _reselect_shooter(self, detections, t, shooter_context)

    def _new_id(self):
        value = self.next_id
        self.next_id += 1
        return value

    def _match(self, detections, dt):
        if not detections or self.track is None:
            return None
        step = min(dt, 0.25) if dt else 0.0
        px = self.track.x + self.track.vx * step
        py = self.track.y + self.track.vy * step
        speed = math.hypot(self.track.vx, self.track.vy)
        gate = (
            self.parameters.get('gate_base_px', 45.0)
            + self.parameters.get('gate_speed_factor', 1.0) * speed * step
            + self.parameters.get('gate_gap_growth_px', 0.0) * self.gap
        )
        best = min(detections, key=lambda d: math.hypot(d.cx - px, d.cy - py))
        return best if math.hypot(best.cx - px, best.cy - py) <= gate else None


__all__ = ["BallTrack", "DetectionBox", "ShotBallTracker", "normalize_fps"]
