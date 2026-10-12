import numpy as np

from pipeline.context import FrameContext, VideoAnalysisContext
from tracking.court import update_court_state
from tracking.team import normalize_team


def _round_geometry(value, size, digits):
    if value is None:
        return None
    return [round(float(item), digits) for item in np.asarray(value).reshape(-1)[:size]]


class FrameFeatureBuilder:

    def __init__(self, state: VideoAnalysisContext) -> None:
        self.state = state

    def build_rim_feature(self, ctx: FrameContext) -> None:
        ctx.rim_feature = None
        ctx.rim_overlay_box = None
        ctx.rim_overlay_score = None
        if ctx.rims is not None and len(ctx.rims):
            best_rim = int(np.argmax(ctx.rims.confidence))
            rim_box = np.asarray(ctx.rims.xyxy[best_rim], dtype=np.float32)
            ctx.rim_feature = {
                "box": _round_geometry(rim_box, 4, 2),
                "center": _round_geometry(
                    (rim_box[:2] + rim_box[2:]) * 0.5,
                    2,
                    3,
                ),
                "width": round(float(rim_box[2] - rim_box[0]), 2),
                "height": round(float(rim_box[3] - rim_box[1]), 2),
                "score": round(float(ctx.rims.confidence[best_rim]), 4),
            }
            ctx.rim_overlay_box = rim_box
            ctx.rim_overlay_score = float(ctx.rims.confidence[best_rim])

    def build_frame_features(self, frame: np.ndarray, ctx: FrameContext) -> FrameContext:
        map_stale_frames = max(
            3,
            round(0.3 * self.state.fps),
        )
        ctx.player_box_colors.clear()
        ctx.display_box_colors.clear()
        ctx.feature_players = []
        ctx.overlay_players = []
        staged_features = []
        for track_id, track in sorted(ctx.active_by_id.items()):
            visible_track = ctx.visible_by_id.get(track_id)
            if visible_track is None:
                continue
            state = self.state.player_states[track_id]
            pose = ctx.player_poses.get(track_id)
            update_court_state(track, state, self.state.court_mapper, self.state.frame_index, self.state.fps, pose)
            reid_eligible = track_id in ctx.reid_track_ids
            team_id = self.state.team_classifier.predict_track(track_id) if reid_eligible else None
            team_color = (
                None
                if team_id is None or self.state.team_classifier.team_colors is None
                else self.state.team_classifier.team_colors[team_id]
            )
            if team_color is not None:
                state.color = tuple(map(int, team_color))
            is_holder = self.state.holder_state.active(track_id, self.state.frame_index)
            box_color = (
                tuple(map(int, team_color)) if team_color is not None else (170, 170, 170)
            )
            court_fresh = (
                state.court_position is not None
                and self.state.frame_index - state.court_frame <= 5
            )
            court_xy = (
                None
                if not court_fresh
                else [round(float(v), 3) for v in state.court_position]
            )
            pose_feature = None
            if pose is not None:
                points, scores = pose
                points = np.asarray(points, dtype=np.float32)
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                pose_feature = [
                    [
                        round(float(x), 2),
                        round(float(y), 2),
                        round(float(score), 4),
                    ]
                    for (x, y), score in zip(points[:, :2], scores)
                ]
            player_feature = {
                "id": int(track_id),
                "box": _round_geometry(visible_track.box, 4, 2),
                "score": round(float(visible_track.score), 4),
                "team": team_id,
                "team_semantic": normalize_team(team_id),
                "holder": bool(is_holder),
                "court_xy": court_xy,
                "pose": pose_feature,
            }
            color_features = (
                self.state.team_classifier.color_features_at(track_id, self.state.frame_index)
                if reid_eligible
                else None
            )
            if color_features is not None:
                player_feature["jersey_l"] = round(
                    float(color_features[0]), 4
                )
                player_feature["jersey_s"] = round(
                    float(color_features[1]), 4
                )
            staged_features.append((player_feature, box_color))
        referee_boxes = (
            ()
            if ctx.referees is None or len(ctx.referees) == 0
            else np.asarray(ctx.referees.xyxy, dtype=np.float32)
        )
        jersey_samples = self.state.jersey_sampler.sample(
            frame,
            ctx.reid_tracks,
            ctx.player_poses,
            referee_boxes,
            self.state.identity_resolver,
            self.state.frame_index,
            self.state.fps,
        )
        global_ids = self.state.identity_resolver.update(
            frame=frame,
            tracks=ctx.visible_tracks,
            active_ids=ctx.active_ids,
            poses=ctx.player_poses,
            frame_index=self.state.frame_index,
            referee_boxes=referee_boxes,
            jersey_samples=jersey_samples,
            reid_eligible_ids=ctx.reid_track_ids,
        )
        for player_feature, box_color in staged_features:
            local_id = int(player_feature["id"])
            player_feature["local_track_id"] = local_id
            global_id = global_ids.get(local_id)
            eligible_for_reid = local_id in ctx.reid_track_ids
            player_feature["reid_eligible"] = bool(eligible_for_reid)
            player_feature["id"] = (
                int(global_id)
                if eligible_for_reid and global_id is not None
                else None
            )
            team_id = self.state.identity_resolver.get_team_id(local_id) if eligible_for_reid else None
            player_feature["team"] = team_id
            player_feature["team_semantic"] = normalize_team(team_id)
            jersey_number = self.state.identity_resolver.get_jersey_number(local_id) if eligible_for_reid else None
            player_feature["jersey_number"] = jersey_number
            player_feature["jersey_number_candidate"] = jersey_number
            player_feature["jersey_number_locked"] = (
                self.state.identity_resolver.is_jersey_number_locked(local_id) if eligible_for_reid else False
            )
            player_feature["identity_label"] = (
                self.state.identity_resolver.get_identity_label(local_id, global_id)
                if eligible_for_reid
                else f"R{local_id}"
            )
            if team_id is not None and self.state.team_classifier.team_colors is not None:
                box_color = tuple(map(int, self.state.team_classifier.team_colors[team_id]))
            ctx.player_box_colors[local_id] = box_color
            ctx.feature_players.append(player_feature)
        for track in ctx.display_only_tracks:
            identity_label = f"R{track.track_id}"
            team_id = self.state.team_classifier.predict_track(track.track_id)
            team_colors = self.state.team_classifier.team_colors
            box_color = (
                tuple(map(int, team_colors[team_id]))
                if team_id is not None and team_colors is not None
                else (170, 170, 170)
            )
            ctx.display_box_colors[track.track_id] = box_color
            ctx.overlay_players.append(
                {
                    "id": None,
                    "local_track_id": int(track.track_id),
                    "box": _round_geometry(track.box, 4, 2),
                    "score": round(float(track.score), 4),
                    "team": None,
                    "team_semantic": None,
                    "holder": False,
                    "court_xy": None,
                    "pose": None,
                    "reid_eligible": False,
                    "jersey_number": None,
                    "jersey_number_candidate": None,
                    "jersey_number_locked": False,
                    "identity_label": identity_label,
                }
            )
        for track_id in tuple(self.state.player_states):
            if track_id not in ctx.active_ids:
                del self.state.player_states[track_id]
        ctx.map_entries = []
        for track_id, state in sorted(self.state.player_states.items()):
            if (
                track_id not in ctx.visible_by_id
                or state.court_position is None
                or self.state.frame_index - state.court_frame > map_stale_frames
            ):
                continue
            entry = {
                "track_id": track_id,
                "court_xy": state.court_position,
                "holder": self.state.holder_state.active(track_id, self.state.frame_index),
            }
            if state.color is not None:
                entry["color"] = state.color
            ctx.map_entries.append(entry)
        ctx.holder_id = (
            None
            if self.state.holder_state.track_id is None
            or not self.state.holder_state.active(self.state.holder_state.track_id, self.state.frame_index)
            else global_ids.get(int(self.state.holder_state.track_id))
        )
        return ctx
