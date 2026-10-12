"""Compose scoreboards and status overlays on rendered frames."""

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from render.primitives import drawing_scale


def _rounded_rect(image, top_left, bottom_right, fill, radius):
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)
    radius = max(
        1,
        min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2),
    )
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), fill, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), fill, -1)
    cv2.circle(image, (x1 + radius, y1 + radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, fill, -1, cv2.LINE_AA)


def _scoreboard_text(frame, text, center_x, center_y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    glyph_padding = max(2, thickness * 2)
    glyph = np.zeros((text_height + baseline + glyph_padding, text_width + glyph_padding), dtype=np.uint8)
    glyph_origin = glyph_padding // 2
    cv2.putText(
        glyph,
        text,
        (
            glyph_origin,
            text_height + glyph_origin,
        ),
        font,
        scale,
        255,
        thickness,
        cv2.LINE_AA,
    )
    _, glyph_top, _, glyph_height = cv2.boundingRect(cv2.findNonZero(glyph))
    glyph_offset = glyph_top + (glyph_height - 1) / 2 - (text_height + glyph_origin)
    origin = (round(center_x - text_width / 2), round(center_y - glyph_offset))
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _scoreboard_marker(frame, x, marker_y, marker_width, marker_border, team_color):
    _rounded_rect(
        frame,
        (round(x), marker_y[0] - marker_border),
        (round(x + marker_width), marker_y[1] + marker_border),
        (225, 230, 238),
        marker_border * 2,
    )
    _rounded_rect(
        frame,
        (round(x + marker_border), marker_y[0]),
        (round(x + marker_width - marker_border), marker_y[1]),
        team_color,
        marker_border,
    )


def _paint_scoreboard(frame, layout):
    panel = frame.copy()
    _rounded_rect(
        panel,
        layout["bounds"][:2],
        layout["bounds"][2:],
        (29, 36, 47),
        round((layout["bounds"][3] - layout["bounds"][1]) / 2),
    )
    alpha = 0.94
    cv2.addWeighted(panel, alpha, frame, 1 - alpha, 0, frame)

    cursor = layout["bounds"][0] + round(layout["padding"])
    _scoreboard_marker(
        frame, cursor, layout["marker_y"], layout["marker_width"], layout["marker_border"], layout["colors"][0]
    )
    cursor += layout["marker_width"] + layout["gap"]
    _scoreboard_text(
        frame,
        layout["labels"][0],
        cursor + layout["label_width"] / 2,
        layout["center_y"],
        layout["label_scale"],
        (171, 181, 194),
        layout["label_thickness"],
    )
    cursor += layout["label_width"] + layout["gap"]

    for index, score in enumerate((layout["dark_score"], layout["light_score"])):
        _scoreboard_text(
            frame,
            score,
            cursor + layout["score_width"] / 2,
            layout["center_y"],
            layout["score_scale"],
            (248, 250, 252),
            layout["score_thickness"],
        )
        cursor += layout["score_width"] + layout["gap"]
        if index == 0:
            _scoreboard_text(
                frame,
                '-',
                cursor + layout["separator_width"] / 2,
                layout["center_y"],
                layout["score_scale"],
                (171, 181, 194),
                layout["score_thickness"],
            )
            cursor += layout["separator_width"] + layout["gap"]

    _scoreboard_text(
        frame,
        layout["labels"][1],
        cursor + layout["label_width"] / 2,
        layout["center_y"],
        layout["label_scale"],
        (171, 181, 194),
        layout["label_thickness"],
    )
    cursor += layout["label_width"] + layout["gap"]
    _scoreboard_marker(
        frame, cursor, layout["marker_y"], layout["marker_width"], layout["marker_border"], layout["colors"][1]
    )
    cursor += layout["marker_width"] + layout["gap"]
    cv2.line(
        frame,
        (round(cursor), round(layout["center_y"] - layout["line_height"] / 2)),
        (round(cursor), round(layout["center_y"] + layout["line_height"] / 2)),
        (86, 97, 111),
        layout["score_thickness"],
        cv2.LINE_AA,
    )
    cursor += layout["score_thickness"] + layout["gap"]
    _scoreboard_text(
        frame,
        layout["time_text"],
        cursor + layout["time_width"] / 2,
        layout["center_y"],
        layout["time_scale"],
        (190, 198, 210),
        layout["time_thickness"],
    )


def draw_scoreboard(
    frame: np.ndarray,
    scores: Mapping[str, int],
    video_time: float | None = None,
    team_colors: Sequence[Sequence[int | None]] | None = None,
    right_clearance: float | None = None,
) -> None:
    """Draw a compact, content-sized scoreboard above the court."""
    video_time = 0.0 if video_time is None else video_time
    height, width = frame.shape[:2]
    scale = drawing_scale(frame)
    available_width = width - 2 * max(
        round(16 * scale),
        round(width * 0.012),
    )
    if right_clearance is not None:
        available_width = min(
            available_width,
            round(2 * (float(right_clearance) - width * 0.5) - 4 * scale),
        )
    if (
        available_width < 60
        or height < 30
    ):
        return

    fallback_colors = (
        (0, 0, 0),
        (255, 255, 255),
    )
    colors = tuple(
        tuple(
            map(
                int,
                (
                    team_colors[index]
                    if team_colors is not None and len(team_colors) > index and team_colors[index] is not None
                    else fallback
                ),
            )
        )
        for index, fallback in enumerate(fallback_colors)
    )

    total_seconds = max(0, int(round(float(video_time))))
    if total_seconds >= 3600:
        time_text = f"{total_seconds // 3600}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
    else:
        time_text = f"{total_seconds // 60}:{total_seconds % 60:02d}"

    max_score = 999
    dark_score = str(max(0, min(max_score, int(scores.get("dark", 0)))))
    light_score = str(max(0, min(max_score, int(scores.get("light", 0)))))
    unit = min(scale, available_width / 442.0)
    font = cv2.FONT_HERSHEY_DUPLEX
    label_scale, score_scale, time_scale = (
        0.42 * unit,
        unit,
        0.55 * unit,
    )
    label_thickness = score_thickness = max(1, round(unit))

    def text_size(text, scale, thickness):
        return cv2.getTextSize(text, font, scale, thickness)[0]

    labels = ['DARK', 'LIGHT']
    label_sizes = [text_size(label, label_scale, label_thickness) for label in labels]
    score_size = text_size(str(max_score), score_scale, score_thickness)
    time_thickness = max(1, round(unit))
    time_width, time_height = text_size(time_text, time_scale, time_thickness)
    separator_width = text_size('-', score_scale, score_thickness)[0]
    label_width = max(size[0] for size in label_sizes)
    score_width = score_size[0]
    line_height = max(max(size[1] for size in label_sizes), score_size[1], time_height)
    padding = max(
        1,
        round(line_height * 0.5),
    )
    gap = max(
        1,
        round(padding * 0.5),
    )
    marker_border = max(1, round(unit))
    marker_width = max(
        round(5 * unit),
        round(line_height / 6.0) + 2 * marker_border,
    )
    team_width = marker_width + gap + label_width
    board_width = round(
        2 * padding
        + 2 * team_width
        + 2 * score_width
        + separator_width
        + time_width
        + 6 * gap
        + score_thickness
    )

    board_height = line_height + 2 * padding
    y1 = max(
        round(4 * unit),
        round(padding * 0.5),
    )
    if y1 + board_height >= height:
        return
    x1 = (width - board_width) // 2
    x2, y2 = x1 + board_width, y1 + board_height
    center_y = y1 + board_height / 2
    marker_y = (
        round(center_y - line_height / 3.0),
        round(center_y + line_height / 3.0),
    )

    _paint_scoreboard(
        frame,
        {
            "bounds": (x1, y1, x2, y2),
            "center_y": center_y,
            "line_height": line_height,
            "padding": padding,
            "gap": gap,
            "marker_y": marker_y,
            "marker_width": marker_width,
            "marker_border": marker_border,
            "colors": colors,
            "labels": labels,
            "label_width": label_width,
            "label_scale": label_scale,
            "label_thickness": label_thickness,
            "dark_score": dark_score,
            "light_score": light_score,
            "score_width": score_width,
            "score_scale": score_scale,
            "score_thickness": score_thickness,
            "separator_width": separator_width,
            "time_text": time_text,
            "time_width": time_width,
            "time_scale": time_scale,
            "time_thickness": time_thickness,
        },
    )


def draw_event_status(frame: np.ndarray, event: Mapping[str, Any] | None, x: int, baseline_y: int) -> int:
    """Draw a compact event label and return its pixel width."""
    if not event:
        return 0
    event_type = str(event.get("event") or "").lower()
    result = str(event.get("result") or "").strip().lower()
    if event_type == "ball_motion" or result in ("rising", "descending"):
        return 0
    if event_type == "dribble":
        label = "DRIBBLE"
    elif event_type == "shot":
        label = f"SHOT {result.upper()}".strip()
        points = event.get("points")
        if result == "made" and points in (2, 3):
            label += f" {points}PT"
    else:
        outcome = event.get("rebound_type") if event_type == "rebound" else result
        label = f"{event_type.upper()} {str(outcome or '').upper()}".strip()
    color = (
        (80, 255, 120)
        if result == "made"
        else (40, 220, 255)
    )
    scale = drawing_scale(frame)
    font_scale = 0.5 * scale
    thickness = max(1, round(2 * scale))
    cv2.putText(
        frame,
        label,
        (int(x), int(baseline_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]


def draw_fps_and_event_status(
    frame: np.ndarray,
    fps_label: str,
    event: Mapping[str, Any] | None,
) -> None:
    """Draw the FPS and current event labels in place."""
    scale = drawing_scale(frame)
    x = round(20 * scale)
    baseline_y = round(32 * scale)
    font_scale = 0.5 * scale
    thickness = max(1, round(2 * scale))
    cv2.putText(
        frame,
        fps_label,
        (x, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    fps_width = cv2.getTextSize(
        fps_label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][0]
    cursor = x + fps_width + round(14 * scale)
    draw_event_status(frame, event, x=cursor, baseline_y=baseline_y)


def draw_video_overlays(
    frame: np.ndarray,
    scores: Mapping[str, int],
    video_time: float,
    team_colors: Sequence[Sequence[int | None]] | None,
    right_clearance: float | None,
    fps_label: str,
    event_display: Mapping[str, Any] | None,
) -> None:
    """Draw the scoreboard and event status overlays on one frame."""
    draw_scoreboard(
        frame,
        scores,
        video_time=video_time,
        team_colors=team_colors,
        right_clearance=right_clearance,
    )
    draw_fps_and_event_status(frame, fps_label, event_display)
