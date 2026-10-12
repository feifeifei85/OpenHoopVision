"""Low-level drawing primitives for annotated frames."""

import cv2
import numpy as np

from tracking.pose import validated_pose
from render.label import text_dimensions


_POSE_CONNECTIONS = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)
_POSE_JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def drawing_scale(frame: np.ndarray) -> float:
    """Scale drawing sizes relative to a 720-pixel-tall frame."""
    return frame.shape[0] / 720.0


def color(track_id: int) -> tuple[int, int, int]:
    """Return a stable display color for a local track ID."""
    hsv = np.uint8([[[int(track_id) * 37 % 180, 220, 240]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_ball_trail(frame: np.ndarray, trail_points) -> None:
    """Smooth and draw the tracked ball's motion trail in place."""
    scale = drawing_scale(frame)
    width = max(1, round(4 * scale))
    trail_points = np.asarray(list(trail_points), dtype=np.float32)
    if len(trail_points) < 2:
        return
    window = 7
    smooth_window = min(window, (len(trail_points) if len(trail_points) % 2 else len(trail_points) - 1))
    if smooth_window >= 3:
        trail_points = cv2.blur(
            trail_points.reshape(-1, 1, 2), (1, smooth_window), borderType=cv2.BORDER_REPLICATE
        ).reshape(-1, 2)
    trail_points = np.rint(trail_points).astype(np.int32)
    hsv_colors = np.full((len(trail_points), 1, 3), 255, dtype=np.uint8)
    hsv_colors[:, 0, 0] = np.linspace(0, 179, len(trail_points), dtype=np.uint8)
    line_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)[:, 0]
    mix = 0.3
    marker_colors = np.clip(line_colors * (1 - mix) + 255 * mix, 0, 255).astype(np.uint8)
    markers = [(0, trail_points[0])]
    spacing = 0.0
    for index in range(1, len(trail_points)):
        spacing += float(np.linalg.norm(trail_points[index] - trail_points[index - 1]))
        cv2.line(
            frame,
            tuple(trail_points[index - 1]),
            tuple(trail_points[index]),
            tuple(map(int, line_colors[index])),
            width,
            cv2.LINE_AA,
        )
        if spacing >= 14.0 * scale:
            markers.append((index, trail_points[index]))
            spacing = 0.0
    if np.linalg.norm(trail_points[-1] - markers[-1][1]) >= 4.0 * scale:
        markers.append((len(trail_points) - 1, trail_points[-1]))
    for index, point in markers:
        cv2.circle(
            frame,
            tuple(map(int, point)),
            max(1, round(3 * scale)),
            tuple(map(int, marker_colors[index])),
            -1,
            cv2.LINE_AA,
        )


def draw_box(frame: np.ndarray, box, label: str, draw_color, style: str = "player") -> None:
    """Draw a labeled detection box in place using a named visual style."""
    scale = drawing_scale(frame)
    if style == "rim":
        font_scale, y_offset = 0.5, 7
    elif style == "ball":
        font_scale, y_offset = 0.45, 2
    elif style == "player":
        font_scale, y_offset = 0.45, 4
    else:
        raise KeyError(style)
    font_scale *= scale
    y_offset = round(y_offset * scale)
    thickness = max(1, round(2 * scale))
    x1, y1, x2, y2 = map(round, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)
    (_, text_height), baseline = text_dimensions(label, font_scale, thickness)
    origin = (x1, max(round(18 * scale), text_height + baseline, y1 - y_offset))
    cv2.putText(
        frame,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        draw_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_player_skeleton(frame: np.ndarray, pose, box, draw_color) -> None:
    """Draw visible player pose joints and connections in place."""
    scale = drawing_scale(frame)
    pose_data = validated_pose(pose[0], pose[1], box, frame.shape)
    if pose_data is None or len(pose_data[0]) < 17:
        return
    points, _, valid = pose_data
    lines = [
        np.rint(points[[first, second], :2]).astype(np.int32)
        for first, second in _POSE_CONNECTIONS
        if valid[first] and valid[second]
    ]
    if lines:
        cv2.polylines(
            frame,
            lines,
            False,
            (8, 12, 18),
            max(1, round(2 * scale)),
            cv2.LINE_AA,
        )
        cv2.polylines(
            frame,
            lines,
            False,
            tuple(map(int, draw_color)),
            max(1, round(scale)),
            cv2.LINE_AA,
        )
    for index in _POSE_JOINTS:
        if valid[index]:
            center = tuple(np.rint(points[index, :2]).astype(np.int32))
            cv2.circle(
                frame,
                center,
                max(1, round(2 * scale)),
                (8, 12, 18),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                center,
                max(1, round(2 * scale)),
                tuple(map(int, draw_color)),
                -1,
                cv2.LINE_AA,
            )
