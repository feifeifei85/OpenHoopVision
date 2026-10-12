"""Compose per-frame tracking annotations on the source image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from render.label import player_overlay_label
from render.minimap import CourtMiniMap
from render.primitives import draw_ball_trail, draw_box, draw_player_skeleton

if TYPE_CHECKING:
    from pipeline.context import FrameContext


class FrameRenderer:
    """Draw the tracking and court layers for one analyzed frame."""

    def __init__(self, court_map: CourtMiniMap) -> None:
        self.court_map = court_map

    @staticmethod
    def draw_rim(frame: np.ndarray, context: FrameContext) -> None:
        if context.rim_overlay_box is None or context.rim_overlay_score is None:
            return
        draw_box(
            frame,
            context.rim_overlay_box,
            f"RIM {context.rim_overlay_score:.2f}",
            (0, 220, 255),
            style="rim",
        )

    def draw_frame(self, frame: np.ndarray, context: FrameContext) -> None:
        for player in context.feature_players:
            local_id = int(player["local_track_id"])
            track = context.visible_by_id.get(local_id)
            pose = context.player_poses.get(local_id)
            if track is not None and pose is not None:
                pose_color = (255, 255, 255) if player["holder"] else (180, 170, 160)
                draw_player_skeleton(frame, pose, track.box, pose_color)

        for player in context.feature_players:
            local_id = int(player["local_track_id"])
            track = context.visible_by_id.get(local_id)
            if track is None:
                continue
            draw_box(
                frame,
                track.box,
                player_overlay_label(player),
                context.player_box_colors.get(local_id, (170, 170, 170)),
                style="player",
            )

        overlay_labels = {
            int(player["local_track_id"]): player["identity_label"]
            for player in context.overlay_players
        }
        for track in context.display_only_tracks:
            draw_box(
                frame,
                track.box,
                overlay_labels.get(track.track_id, f"R{track.track_id}"),
                context.display_box_colors.get(track.track_id, (170, 170, 170)),
                style="player",
            )

        draw_ball_trail(frame, context.ball_trail)
        ball_track = context.ball_track
        if ball_track is not None and ball_track.bbox is not None:
            draw_box(
                frame,
                ball_track.bbox,
                f"BALL {ball_track.confidence or 0.0:.2f}",
                (20, 190, 255),
                style="ball",
            )

        self.court_map.draw(frame, context.map_entries)


__all__ = ["FrameRenderer"]
