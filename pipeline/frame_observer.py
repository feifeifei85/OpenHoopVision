import cv2
import numpy as np

from pipeline.context import FrameContext, VideoAnalysisContext
from events.detector import BallEventPipeline
from models.detectors import detect, predict_shape
from tracking.player import (
    PlayerState,
    TrackView,
    detection_tracks,
    match_holder_track,
    recover_display_tracks,
)
from tracking.pose import match_pose_to_players, resolve_pose
from tracking.team import build_jersey_mask, extract_jersey_color
from tracking.ball import DetectionBox
from tracking.court import is_player_on_court


class FrameObserver:

    def __init__(self, state: VideoAnalysisContext) -> None:
        self.state = state

    def _refresh_visible_players(
        self,
        tracks: list[TrackView],
        active_by_id: dict[int, TrackView],
        referee_track_ids: set[int],
        pose_matches: dict[int, tuple[np.ndarray, np.ndarray]],
        frame_shape: tuple[int, ...],
    ) -> tuple[list[TrackView], dict[int, TrackView]]:
        on_court_track_ids = {
            track.track_id
            for track in tracks
            if track.track_id not in referee_track_ids
            and is_player_on_court(
                track,
                pose_matches.get(track.track_id),
                self.state.court_mapper,
                self.state.frame_index,
                frame_shape,
                self.state.identity_resolver.parameters.get("fallback_crop_min_height_px", 100),
            )
        }
        self.state.verified_player_ids.update(
            track.track_id
            for track in tracks
            if track.track_id in on_court_track_ids and track.score >= 0.55
        )
        self.state.verified_player_ids.intersection_update(active_by_id)
        visible_tracks = [
            track
            for track in tracks
            if track.track_id in self.state.verified_player_ids and track.track_id in on_court_track_ids
        ]
        visible_by_id = {track.track_id: track for track in visible_tracks}
        for track_id in visible_by_id:
            self.state.player_states.setdefault(track_id, PlayerState())
        return visible_tracks, visible_by_id

    def observe_frame(self, frame: np.ndarray) -> FrameContext:
        if self.state.identity_resolver.detect_clip_cut(frame, self.state.frame_index):
            self.state.ball_tracker.reset()
            self.state.holder_state.reset()
            self.state.event_pipeline = BallEventPipeline(self.state.fps)
            self.state.court_mapper.reset()
            self.state.context_rows.clear()
            self.state.team_history.clear()
            self.state.score_queue.clear()
            self.state.jersey_sampler.reset_interval()
            self.state.previous_display_tracks.clear()
            self.state.verified_player_ids.clear()
            self.state.last_event_players = []
            self.state.last_event_display = None
            self.state.last_event_display_frame = None
            for state in self.state.player_states.values():
                state.color = None
                state.cached_pose = None
                state.court_observation = None
                state.court_position = None
                state.court_frame = -(10**9)
        court_future = None
        if (self.state.frame_index - 1) % self.state.args.court_interval == 0:
            court_future = self.state.court_pose.submit(frame)
        pose_scheduled = (self.state.frame_index - 1) % self.state.args.pose_interval == 0
        pose_future = None
        if pose_scheduled:
            pose_future = self.state.player_pose.submit(frame)
        if self.state.detector_shape is None:
            self.state.detector_shape = predict_shape(frame, self.state.args.short_side)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector_players, balls, rims, holders, scorers, referees = detect(
            self.state.detector,
            frame_rgb,
            self.state.detector_shape,
            self.state.class_ids,
            player_threshold=self.state.args.player_threshold,
            ball_threshold=self.state.args.ball_threshold,
            rim_threshold=self.state.args.rim_threshold,
            holder_threshold=self.state.args.holder_threshold,
            scorer_threshold=self.state.args.scorer_threshold,
            referee_threshold=self.state.args.referee_threshold,
        )
        if not pose_scheduled:
            pose_boxes = np.empty((0, 4), dtype=np.float32)
            pose_points = np.empty((0, 17, 2), dtype=np.float32)
            pose_scores = np.empty((0, 17), dtype=np.float32)
        else:
            pose_boxes, _, pose_points, pose_scores = self.state.player_pose.result(pose_future, frame)
        players = detector_players
        detector_pose_matches = match_pose_to_players(
            detection_tracks(players, score_default=1.0),
            pose_boxes,
            pose_points,
            pose_scores,
            self.state.iou_metric,
        )
        tracker_pose_points = np.zeros((len(players), *pose_points.shape[1:]), dtype=np.float32)
        tracker_pose_scores = np.zeros((len(players), *pose_scores.shape[1:]), dtype=np.float32)
        for detection_id, (points, scores) in detector_pose_matches.items():
            detection_index = -detection_id - 1
            tracker_pose_points[detection_index] = points
            tracker_pose_scores[detection_index] = scores
        all_visible_tracks = detection_tracks(
            self.state.player_tracker.update(
                players,
                frame=frame,
                pose_data=(tracker_pose_points, tracker_pose_scores),
            ),
            tracked=True,
        )
        pose_matches = match_pose_to_players(
            all_visible_tracks, pose_boxes, pose_points, pose_scores, self.state.iou_metric
        )
        all_active_by_id = {
            track.track_id: track
            for track in detection_tracks(self.state.player_tracker.tracked_objects, tracked=True)
        }
        unconfirmed_track_ids = {
            track.track_id
            for track in self.state.player_tracker.tracked_stracks
            if not track.activated
        }
        display_candidates = [
            *all_visible_tracks,
            *(all_active_by_id[track_id] for track_id in unconfirmed_track_ids if track_id in all_active_by_id),
        ]
        referee_boxes = (
            np.empty((0, 4), dtype=np.float32)
            if referees is None or len(referees) == 0
            else np.asarray(referees.xyxy, dtype=np.float32)
        )
        referee_track_ids = set()
        if referee_boxes.size and display_candidates:
            referee_overlap = self.state.iou_metric.compute(
                np.asarray([track.box for track in display_candidates], dtype=np.float32),
                referee_boxes,
            )
            referee_track_ids = {
                track.track_id
                for index, track in enumerate(display_candidates)
                if float(np.max(referee_overlap[index])) >= 0.35
            }
        visible_tracks, visible_by_id = self._refresh_visible_players(
            all_visible_tracks, all_active_by_id, referee_track_ids, pose_matches, frame.shape
        )
        if (self.state.frame_index - 1) % self.state.args.court_interval == 0:
            court_points, court_scores = self.state.court_pose.result(court_future, frame)
            continuity_data = []
            for track in visible_tracks:
                state = self.state.player_states[track.track_id]
                court_age = (self.state.frame_index - state.court_frame) / self.state.fps
                if (
                    state.court_position is None
                    or court_age < 0
                    or court_age > 1.0
                ):
                    continue
                x1, _, x2, y2 = map(float, track.box)
                continuity_data.append(
                    (
                        (x1 + x2) * 0.5,
                        y2,
                        *state.court_position,
                        1.5
                        + 12.0 * court_age,
                    )
                )
            if self.state.court_mapper.update(
                court_points,
                court_scores,
                self.state.frame_index,
                frame.shape,
                continuity_data,
            ):
                self.state.court_mapper.refine(
                    frame,
                    self.state.frame_index,
                    (track.box for track in visible_tracks),
                )
                visible_tracks, visible_by_id = self._refresh_visible_players(
                    all_visible_tracks, all_active_by_id, referee_track_ids, pose_matches, frame.shape
                )
        active_by_id = {
            track_id: track
            for track_id, track in all_active_by_id.items()
            if track_id in self.state.verified_player_ids
        }
        active_ids = set(active_by_id)
        self.state.team_classifier.prune(
            active_ids,
            self.state.frame_index,
            self.state.fps,
            max(0.0, float(self.state.args.team_cache_ttl_seconds)),
        )
        pose_matches = {
            track_id: pose
            for track_id, pose in pose_matches.items()
            if track_id in visible_by_id
        }
        max_pose_age = max(2, 2 * self.state.args.pose_interval)
        player_poses = {}
        reid_track_ids = set(visible_by_id)
        for track in visible_tracks:
            state = self.state.player_states[track.track_id]
            pose = resolve_pose(
                track, state, pose_matches.get(track.track_id), self.state.frame_index, max_pose_age
            )
            if pose is None:
                continue
            player_poses[track.track_id] = pose
            jersey_region = build_jersey_mask(frame.shape, pose[0], pose[1], track.box)
            descriptor, jersey_bgr, palette_sample = extract_jersey_color(
                frame, jersey_region
            )
            self.state.team_classifier.observe(
                track.track_id, descriptor, jersey_bgr, self.state.frame_index, palette_sample
            )
        self.state.team_classifier.maybe_fit(self.state.frame_index)
        self.state.holder_state.update(
            match_holder_track(visible_tracks, holders, self.state.iou_metric),
            active_ids,
            self.state.frame_index,
        )
        ball_detections = [
            DetectionBox(float(item.score), tuple(map(float, item.box)))
            for item in detection_tracks(
                balls,
                score_default=1.0,
            )
        ]
        display_only_tracks = [
            track
            for track in display_candidates
            if track.track_id not in visible_by_id
            and track.track_id not in referee_track_ids
        ]
        display_only_tracks = recover_display_tracks(
            detection_tracks(players, score_default=1.0),
            visible_tracks,
            display_only_tracks,
            self.state.previous_display_tracks,
            self.state.iou_metric,
        )
        self.state.previous_display_tracks = [*visible_tracks, *display_only_tracks]
        return FrameContext(
            players=players,
            balls=balls,
            rims=rims,
            holders=holders,
            scorers=scorers,
            referees=referees,
            visible_tracks=visible_tracks,
            display_only_tracks=display_only_tracks,
            reid_tracks=[track for track in visible_tracks if track.track_id in reid_track_ids],
            reid_track_ids=reid_track_ids,
            visible_by_id=visible_by_id,
            active_by_id=active_by_id,
            active_ids=active_ids,
            player_poses=player_poses,
            ball_detections=ball_detections,
        )
