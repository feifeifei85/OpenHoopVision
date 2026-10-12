import numpy as np

from pipeline.context import (
    FrameContext,
    FrameEventResult,
    VideoAnalysisContext,
    VideoFinalizationResult,
)
from events.context import scorer_feature, shooter_ball_context
from events.scoring import apply_shot_events_to_scoreboard
from events.attribution import refine_shot_attribution
from reid.records import translate_event_identity_references
from tracking.team import normalize_team


def _detections_feature(detections):
    if detections is None or len(detections) == 0:
        return []
    confidence = detections.confidence
    confidence = (
        np.full(len(detections), 1.0, dtype=np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32).reshape(-1)
    )
    return [
        {
            "box": (
                None
                if box is None
                else [round(float(item), 2) for item in np.asarray(box).reshape(-1)[:4]]
            ),
            "score": round(float(score), 4),
        }
        for box, score in zip(detections.xyxy, confidence)
    ]


class FrameEventProcessor:

    def __init__(self, state: VideoAnalysisContext) -> None:
        self.state = state

    def _event_display_status(self, status_fields, completed_events):
        if completed_events:
            self.state.last_event_display = completed_events[-1]
            self.state.last_event_display_frame = self.state.frame_index
            return self.state.last_event_display
        if status_fields["ball_status"] is not None:
            self.state.last_event_display = None
            self.state.last_event_display_frame = None
            return None
        if (
            self.state.last_event_display is not None
            and self.state.frame_index - self.state.last_event_display_frame <= round(2 * self.state.fps)
        ):
            status = status_fields["event_status"]
            if status is not None and status.get("event") == "dribble" and status.get("end_frame") is None:
                return status
            return self.state.last_event_display
        if status_fields["event_status"] is not None:
            return status_fields["event_status"]
        self.state.last_event_display = None
        self.state.last_event_display_frame = None
        return None

    def process_frame(self, ctx: FrameContext) -> FrameEventResult:
        event_players = []
        for player in ctx.feature_players:
            local_id = int(player["local_track_id"] if player.get("local_track_id") is not None else player["id"])
            event_player = {**player, "id": local_id}
            # Keep event inference on the original per-frame team prediction.
            raw_team = (
                self.state.team_classifier.predict_raw_track(local_id)
                if local_id in ctx.reid_track_ids
                else None
            )
            event_player["team"] = raw_team
            event_player["team_semantic"] = normalize_team(raw_team)
            event_players.append(event_player)
        scorer_candidates = [*event_players]
        for player in ctx.overlay_players:
            local_id = int(player["local_track_id"])
            team_id = self.state.team_classifier.predict_track(local_id)
            scorer_candidates.append(
                {
                    **player,
                    "id": local_id,
                    "team": team_id,
                    "team_semantic": normalize_team(team_id),
                }
            )
        scorer_detections = _detections_feature(ctx.scorers)
        scorer_tag = scorer_feature(
            {
                "players": [*ctx.feature_players, *ctx.overlay_players],
                "detections": {"scorer": scorer_detections},
            }
        )
        event_holder_id = (
            int(self.state.holder_state.track_id)
            if self.state.holder_state.track_id is not None
            and self.state.holder_state.active(self.state.holder_state.track_id, self.state.frame_index)
            else None
        )
        event_scorer_tag = scorer_feature(
            {
                "players": event_players,
                "scorer_candidates": scorer_candidates,
                "detections": {"scorer": scorer_detections},
            }
        )
        shooter_context = shooter_ball_context(
            scorer_candidates,
            event_holder_id,
            event_scorer_tag,
            self.state.event_pipeline.machine.active_player_id,
            self.state.event_pipeline.event_parameters,
        )
        frame_time = (self.state.frame_index - 1) / self.state.fps
        ball_track = self.state.ball_tracker.update(ctx.ball_detections, frame_time, shooter_context)
        ctx.ball_track = ball_track
        ctx.ball_trail = list(self.state.ball_tracker.trail)
        ball_feature = None
        if ball_track is not None:
            ball_feature = {
                "id": int(ball_track.track_id),
                "box": (
                    None
                    if ball_track.bbox is None
                    else [
                        round(float(item), 2)
                        for item in np.asarray(ball_track.bbox).reshape(-1)[:4]
                    ]
                ),
                "center": [
                    round(float(ball_track.x), 3),
                    round(float(ball_track.y), 3),
                ],
                "score": (
                    None
                    if ball_track.confidence is None
                    else round(float(ball_track.confidence), 4)
                ),
                "observed": not ball_track.interpolated,
                "interpolated": bool(ball_track.interpolated),
                "velocity_valid": bool(ball_track.velocity_valid),
                "velocity_px_frame": [
                    round(ball_track.vx / self.state.fps, 3),
                    round(ball_track.vy / self.state.fps, 3),
                ],
                "velocity_px_s": [
                    round(ball_track.vx, 3),
                    round(ball_track.vy, 3),
                ],
            }
        frame_record = {
            "schema": "basketball_frame_v2",
            "frame": int(self.state.frame_index),
            "time": round(float(frame_time), 6),
            "fps": round(float(self.state.fps), 6),
            "rim": ctx.rim_feature,
            "ball": ball_feature,
            "holder_id": ctx.holder_id,
            "players": [*ctx.feature_players, *ctx.overlay_players],
            "scorer_tag": scorer_tag,
            "detections": {
                "player": _detections_feature(ctx.players),
                "ball": _detections_feature(ctx.balls),
                "rim": _detections_feature(ctx.rims),
                "holder": _detections_feature(ctx.holders),
                "scorer": scorer_detections,
                "referee": _detections_feature(ctx.referees),
            },
            "events": [],
        }
        if self.state.frame_index == 1:
            identity_parameters = self.state.identity_resolver.parameters
            jersey_ocr = identity_parameters.get("jersey_ocr", {})
            min_ocr_confidence = max(0.65, float(jersey_ocr.get("min_read_confidence", 0.55)))
            sample_interval_frames = int(jersey_ocr.get("sample_interval_frames", 1))
            confirmations = int(jersey_ocr.get("confirmations", 4))
            frame_record["run_config"] = {
                "frame_time_origin": "frame_zero",
                "shot_event_detector": self.state.event_pipeline.event_parameters.get("schema", "shot_tracker_fsm_v1"),
                "shot_event_parameters": dict(self.state.event_pipeline.event_parameters),
                "ball_tracker": dict(self.state.ball_tracker.parameters),
                "scorer_detector_threshold": float(self.state.args.scorer_threshold),
                "detection_thresholds": {
                    "player": float(self.state.args.player_threshold),
                    "ball": float(self.state.args.ball_threshold),
                    "rim": float(self.state.args.rim_threshold),
                    "holder": float(self.state.args.holder_threshold),
                    "scorer": float(self.state.args.scorer_threshold),
                    "referee": float(self.state.args.referee_threshold),
                },
                "scorer_player_min_iou": 0.3,
                "player_detector": str(self.state.args.model),
                "auxiliary_detector": str(self.state.args.pose_model),
                "player_identity": {
                    "schema": "global_player_identity_v1",
                    "local_id_field": "local_track_id",
                    "global_id_field": "id",
                    "reid_model": "OSNet x1.0 sports_model.pth.tar-60",
                    "matching_score": "multi-frame OSNet cosine similarity with team and locked jersey evidence",
                    "reid_crop": "pose-derived foreground mask with full-box fallback",
                    "reid_eligibility": (
                        "verified player; referee excluded; court projection or conservative screen zone; "
                        "small crops require visible ankle keypoints"
                    ),
                    "reid_similarity_floor": float(
                        identity_parameters.get("reid_similarity_floor", 0.76)
                    ),
                    "association": "one-to-one mutual nearest neighbor",
                    "provisional_label": "R",
                    "stable_label": "G",
                    "stabilization_window_frames": int(
                        identity_parameters.get("reid_new_max_len", 10)
                    ),
                    "reid_match_stabilizes_immediately": True,
                    "similarity_threshold": float(
                        identity_parameters.get("similarity_threshold", 0.80)
                    ),
                    "match_min_features": int(identity_parameters.get("match_min_features", 2)),
                    "clip_cut": dict(identity_parameters.get("clip_cut", {})),
                    "jersey_number_binding": {
                        "online": True,
                        "global_identity_fusion": True,
                        "crop_region": "square crop centered between shoulder and hip keypoints",
                        "input_preprocessing": "BGR to RGB, bicubic resize to PARSeq size, normalize to [-1, 1]",
                        "provisional_min_confidence": min_ocr_confidence,
                        "candidate_change": {
                            "minimum_votes": max(4, confirmations),
                            "minimum_span_frames": max(12, 2 * sample_interval_frames),
                            "purity": 0.75,
                            "mean_confidence": max(0.70, min_ocr_confidence),
                        },
                        "sample_interval_frames": sample_interval_frames,
                        "confirmations": confirmations,
                        "minimum_vote_purity": float(jersey_ocr.get("purity", 0.85)),
                        "minimum_mean_confidence": float(jersey_ocr.get("confidence", 0.80)),
                        "min_span_frames": int(jersey_ocr.get("min_span_frames", 12)),
                        "min_crop_area": int(jersey_ocr.get("min_crop_area", 196)),
                    },
                },
            }
        event_frame_record = dict(frame_record)
        event_frame_record["players"] = event_players
        event_frame_record["scorer_candidates"] = scorer_candidates
        event_frame_record["holder_id"] = event_holder_id
        event_frame_record["scorer_tag"] = event_scorer_tag
        self.state.last_event_players = event_players
        self.state.context_rows.append(event_frame_record)
        completed_events = self.state.event_pipeline.update(ball_track, event_frame_record)
        completed_shots = [event for event in completed_events if event.get("event") == "shot"]
        refine_shot_attribution(
            completed_shots,
            self.state.team_classifier.color_history,
            self.state.context_rows,
            self.state.team_history,
        )
        self.state.event_pipeline.machine.attach_dribble_context(completed_shots)
        apply_shot_events_to_scoreboard(
            self.state.scores,
            completed_events,
            team_observations=event_players,
            pending_events=self.state.score_queue,
        )
        self.state.event_count += len(completed_events)
        frame_record["events"] = translate_event_identity_references(completed_events, self.state.identity_resolver)
        frame_record["shot_tracking"] = {
            "phase": self.state.event_pipeline.machine.phase,
            "rim_box": (
                None
                if self.state.event_pipeline.last_rim_box is None
                else [
                    round(float(item), 2)
                    for item in np.asarray(
                        self.state.event_pipeline.last_rim_box
                    ).reshape(-1)[:4]
                ]
            ),
            "px_per_m": self.state.event_pipeline.last_scale,
            "px_per_m_hint": self.state.event_pipeline.last_scale_hint,
            "rim_calibration_samples": len(self.state.event_pipeline.rims.samples),
        }
        local_status_fields = self.state.event_pipeline.status_fields_at(self.state.frame_index)
        status_fields = translate_event_identity_references(
            local_status_fields,
            self.state.identity_resolver,
            parent_frame=self.state.frame_index,
        )
        frame_record.update(status_fields)
        status_event = status_fields["event_status"] or status_fields["ball_status"]
        event_display = self._event_display_status(status_fields, completed_events)
        ball_status = local_status_fields["ball_status"]
        reminders = (
            [
                {
                    "event": "shot_reminder",
                    "phase": ball_status["result"],
                    "player_id": ball_status.get("player_id"),
                    "local_track_id": ball_status.get("local_track_id"),
                }
            ]
            if ball_status is not None
            and ball_status.get("result") in ("rising", "descending")
            else []
        )
        frame_record["shot_reminders"] = translate_event_identity_references(
            reminders, self.state.identity_resolver, parent_frame=self.state.frame_index
        )
        self.state.reminder_count += len(reminders)
        frame_record["scoreboard"] = {
            "dark": int(self.state.scores["dark"]),
            "light": int(self.state.scores["light"]),
            "status": None if status_event is None else status_event.get("result") or status_event.get("event"),
        }
        return FrameEventResult(
            frame_record=frame_record,
            frame_time=frame_time,
            event_display=event_display,
        )

    def finalize_video(self) -> VideoFinalizationResult:
        flushed_events = self.state.event_pipeline.flush()
        flushed_shots = [event for event in flushed_events if event.get("event") == "shot"]
        refine_shot_attribution(
            flushed_shots,
            self.state.team_classifier.color_history,
            self.state.context_rows,
            self.state.team_history,
        )
        self.state.event_pipeline.machine.attach_dribble_context(flushed_shots)
        apply_shot_events_to_scoreboard(
            self.state.scores,
            flushed_events,
            team_observations=self.state.last_event_players,
            pending_events=self.state.score_queue,
        )
        self.state.event_count += len(flushed_events)
        local_status_fields = self.state.event_pipeline.status_fields_at(self.state.frame_index)
        final_status_fields = translate_event_identity_references(
            local_status_fields,
            self.state.identity_resolver,
            parent_frame=self.state.frame_index,
        )
        final_status = final_status_fields["event_status"] or final_status_fields["ball_status"]
        event_display = self._event_display_status(final_status_fields, flushed_events)
        reminders = (
            [
                {
                    "event": "shot_reminder",
                    "phase": local_status_fields["ball_status"]["result"],
                    "player_id": local_status_fields["ball_status"].get("player_id"),
                    "local_track_id": local_status_fields["ball_status"].get("local_track_id"),
                }
            ]
            if local_status_fields["ball_status"] is not None
            and local_status_fields["ball_status"].get("result") in ("rising", "descending")
            else []
        )
        translated_events = translate_event_identity_references(
            flushed_events, self.state.identity_resolver
        )
        translated_reminders = translate_event_identity_references(
            reminders,
            self.state.identity_resolver,
            parent_frame=self.state.frame_index,
        )
        scoreboard = {
            "dark": int(self.state.scores["dark"]),
            "light": int(self.state.scores["light"]),
            "status": None if final_status is None else final_status.get("result") or final_status.get("event"),
        }
        return VideoFinalizationResult(
            flushed_events=translated_events,
            status_fields=final_status_fields,
            scoreboard=scoreboard,
            shot_phase=self.state.event_pipeline.machine.phase,
            shot_reminders=translated_reminders,
            event_display=event_display,
        )
