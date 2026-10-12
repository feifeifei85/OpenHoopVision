import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import cv2

from render.primitives import draw_box
from render.label import player_overlay_label, text_dimensions
from render.writer import FfmpegVideoWriter


def rewrite_identity_overlay(
    source_video: Path,
    output_video: Path,
    fallback_fps: float,
    team_colors: Any,
    records: list[dict[str, Any]],
    previous_labels: dict | None = None,
) -> None:
    """Rewrite output frames with reconciled team colors and player labels."""
    previous_labels = previous_labels or {}
    source = cv2.VideoCapture(str(source_video))
    rendered = cv2.VideoCapture(str(output_video))
    output_mode = None
    output_tmp = None
    writer = None
    written = 0
    frames_complete = False
    writer_closed = False
    cleanup_ok = True
    try:
        if output_video.exists():
            output_mode = stat.S_IMODE(output_video.stat().st_mode)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_video.stem}.labels.",
            suffix=output_video.suffix,
            dir=output_video.parent,
        )
        output_tmp = Path(temporary_name)
        os.close(file_descriptor)
        fps = float(rendered.get(cv2.CAP_PROP_FPS) or fallback_fps)
        size = (
            int(rendered.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(rendered.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        scale = size[1] / 720.0
        label_font_scale = 0.45 * scale
        label_thickness = max(1, round(2 * scale))
        writer = FfmpegVideoWriter(output_tmp, fps, size)
        if source.isOpened() and rendered.isOpened() and writer.is_opened():
            for record in records:
                source_ok, source_frame = source.read()
                rendered_ok, rendered_frame = rendered.read()
                if not source_ok or not rendered_ok:
                    break
                frame_index = int(record.get("frame", written + 1))
                for player_index, player in enumerate(record.get("players", ())):
                    x1, y1, _, _ = map(round, player["box"])
                    x1 = min(max(x1, 0), size[0] - 1)
                    y1 = min(max(y1, 0), size[1] - 1)
                    team_id = player.get("team")
                    draw_color = (
                        tuple(map(int, team_colors[int(team_id)]))
                        if team_id is not None and team_colors is not None
                        else (170, 170, 170)
                    )
                    text = player_overlay_label(player)
                    old_text = previous_labels.get(
                        (frame_index, player.get("local_track_id"), player_index), text
                    )
                    dimensions = [
                        text_dimensions(label, label_font_scale, label_thickness)
                        for label in {text, old_text}
                    ]
                    text_height = max(item[0][1] for item in dimensions)
                    baseline = max(item[1] for item in dimensions)
                    text_y = max(
                        round(18 * scale), text_height + baseline, y1 - round(4 * scale)
                    )
                    padding = max(1, round(2 * scale))
                    left = max(0, x1 - padding)
                    right = min(size[0], x1 + max(item[0][0] for item in dimensions) + 2 * padding)
                    top = max(0, text_y - text_height - baseline - padding)
                    bottom = min(size[1], text_y + text_height + baseline + padding)
                    rendered_frame[top:bottom, left:right] = source_frame[top:bottom, left:right]
                    draw_box(rendered_frame, player["box"], text, draw_color, style="player")
                writer.write(rendered_frame)
                written += 1
            frames_complete = written == len(records)
    finally:
        try:
            if writer is not None:
                writer.release()
                writer_closed = True
        finally:
            try:
                source.release()
            except BaseException:
                cleanup_ok = False
                raise
            finally:
                try:
                    rendered.release()
                except BaseException:
                    cleanup_ok = False
                    raise
                finally:
                    if output_tmp is not None and not (frames_complete and writer_closed and cleanup_ok):
                        output_tmp.unlink(missing_ok=True)

    completed = output_tmp is not None and frames_complete and writer_closed and cleanup_ok
    if completed:
        try:
            if output_mode is not None:
                os.chmod(output_tmp, output_mode)
            os.replace(output_tmp, output_video)
        except BaseException:
            output_tmp.unlink(missing_ok=True)
            raise
    else:
        print("warning: identity labels updated in tracks, but result video could not be rewritten")


__all__ = ["rewrite_identity_overlay"]
