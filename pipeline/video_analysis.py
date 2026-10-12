import json
import os
import time
import warnings
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from rfdetr import from_checkpoint
from tqdm import tqdm

from events.detector import BallEventPipeline
from models.court import CourtPoseDetector
from models.detectors import resolve_detector_class_ids
from models.jersey_number import JerseyNumberReader
from models.player_pose import PlayerPoseDetector
from pipeline.context import (
    FrameEventResult,
    VideoAnalysisContext,
    VideoFinalizationResult,
)
from pipeline.frame_events import FrameEventProcessor
from pipeline.frame_features import FrameFeatureBuilder
from pipeline.frame_observer import FrameObserver
from pipeline.identity_finalization import finalize_identity_records
from reid.jersey import JerseyNumberSampler
from reid.resolver import IdentityResolver
from render.frame import FrameRenderer
from render.identity import rewrite_identity_overlay
from render.label import player_overlay_label
from render.minimap import CourtMiniMap
from render.overlays import (
    draw_fps_and_event_status,
    draw_scoreboard,
    draw_video_overlays,
)
from render.writer import FfmpegVideoWriter
from tracking.ball import ShotBallTracker
from tracking.court import CourtMapper, default_nba_court_vertices
from tracking.mcbyte import GuardedCMC, IoU
from tracking.player import HolderState, make_tracker
from tracking.team import JerseyTeamClassifier

_RUNTIME_CONFIGURED = False


def configure_runtime() -> None:
    """Configure inference thread counts when a video runner is explicitly created."""
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        # Torch does not allow changing interop threads after parallel work begins.
        pass
    _RUNTIME_CONFIGURED = True


def _same_path(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _print_video_summary(run: VideoAnalysisContext) -> None:
    elapsed = time.perf_counter() - run.start_time
    frame_processing_elapsed = run.frame_processing_elapsed
    print(f"runtime events: {run.event_count}; shot reminders: {run.reminder_count}; scoreboard: {run.scores}")
    print(f"output: {run.args.output}")
    print(f"tracks: {run.args.tracks}")
    if run.team_classifier.team_colors is not None:
        print(f"team colors (BGR): {run.team_classifier.team_colors}; locked={run.team_classifier.team_colors_locked}")
    cmc = getattr(run.player_tracker, "cmc", None)
    if isinstance(cmc, GuardedCMC):
        print(
            "player CMC: "
            f"method={'orb'}, downscale={4}, "
            f"interval={cmc.interval}, applied={cmc.applied}, "
            f"rejected={cmc.rejected}, skipped={cmc.skipped}"
        )
    print(
        "Pose association masks: "
        f"detections={run.player_tracker.pose_mask_detections}, "
        f"candidate pairs scored={run.player_tracker.pose_mask_pairs_scored}"
    )
    print(
        f"frames: {run.frame_index}, processing: {frame_processing_elapsed:.2f}s, "
        f"fps: {run.frame_index / max(frame_processing_elapsed, 1e-06):.2f}; "
        f"total: {elapsed:.2f}s, end-to-end fps: {run.frame_index / max(elapsed, 1e-06):.2f}"
    )
    encoder_status = (
        f"unavailable ({run.identity_resolver.encoder.error})"
        if run.identity_resolver.encoder.error
        else "loaded" if run.identity_resolver.encoder.model is not None else "not needed"
    )
    print(
        "OSNet sports ReID: "
        f"{encoder_status}; crops={run.identity_resolver.total_crops}, "
        f"matches={run.identity_resolver.match_count}, "
        f"history relinks={run.identity_resolver.identity_relink_count}, "
        f"team switch splits={run.identity_resolver.team_switch_split_count}, "
        f"OCR lock resets={run.identity_resolver.jersey_conflict_reset_count}, "
        f"total={run.identity_resolver.total_reid_ms:.1f}ms"
    )
    jersey_reader = run.jersey_sampler.reader
    jersey_status = (
        f"unavailable ({jersey_reader.error})"
        if jersey_reader.error
        else "loaded" if jersey_reader.model is not None else "not needed"
    )
    print(
        f"PARSeq jersey OCR: {jersey_status}; samples={jersey_reader.observations}, "
        f"accepted reads={jersey_reader.accepted}, "
        f"confirmed G labels={run.identity_resolver.jersey_confirmation_count}"
        f", provisional numbers={run.identity_resolver.jersey_candidate_count}, "
        f"candidate changes={run.identity_resolver.jersey_candidate_change_count}, "
        f"locks={run.identity_resolver.jersey_lock_count}, "
        f"referee skips={run.identity_resolver.jersey_referee_rejects}"
    )
    print(
        f"global player IDs: {len(run.identity_resolver.identities)} created; "
        f"visible tracks get an ID; "
        f"clip boundaries: {run.identity_resolver.clip_cut_count}"
    )


class FrameArtifactWriter:
    """Keep annotated video frames and their JSONL records in sync."""

    def __init__(self, video_writer: FfmpegVideoWriter, track_file: TextIO) -> None:
        self.video_writer = video_writer
        self.track_file = track_file
        self.last_frame_record: dict | None = None
        self.last_render_base: np.ndarray | None = None
        self.last_frame_time = 0.0
        self.last_fps_label: str | None = None
        self.pending_frame: np.ndarray | None = None

    def write_frame(
        self,
        frame: np.ndarray,
        frame_record: dict,
        frame_time: float,
        draw_scoreboard: Callable[[np.ndarray], None],
        make_fps_label: Callable[[], str],
        draw_status: Callable[[np.ndarray, str], None],
    ) -> None:
        self.last_render_base = frame.copy()
        self.last_frame_time = frame_time
        draw_scoreboard(frame)
        if self.last_frame_record is not None:
            self.track_file.write(json.dumps(self.last_frame_record, ensure_ascii=False) + "\n")
        self.last_frame_record = frame_record
        fps_label = make_fps_label()
        draw_status(frame, fps_label)
        self.last_fps_label = fps_label
        if self.pending_frame is not None:
            self.video_writer.write(self.pending_frame)
        self.pending_frame = frame

    def finish(self, draw_overlays: Callable[[np.ndarray], None]) -> None:
        if self.last_render_base is not None:
            final_frame = self.last_render_base.copy()
            draw_overlays(final_frame)
            self.video_writer.write(final_frame)
        if self.last_frame_record is not None:
            self.track_file.write(json.dumps(self.last_frame_record, ensure_ascii=False) + "\n")


class VideoAnalysisRunner:
    """Compose runtime dependencies and coordinate video processing stages."""

    def __init__(self, args: Any) -> None:
        configure_runtime()
        video_path = Path(args.video).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        tracks_path = Path(args.tracks).expanduser().resolve()
        if _same_path(video_path, output_path):
            raise ValueError("--output must not refer to the input video")
        if _same_path(video_path, tracks_path):
            raise ValueError("--tracks must not refer to the input video")
        if _same_path(output_path, tracks_path):
            raise ValueError("--tracks must not refer to the output video")
        if args.max_frames is not None and args.max_frames < 0:
            raise ValueError("--max-frames must be non-negative")

        args.court_interval = max(1, int(args.court_interval))
        args.pose_interval = max(1, int(args.pose_interval))
        self._processing_cleanup = ExitStack()
        self._writer_cleanup = ExitStack()
        try:
            self._initialize_runtime(args)
        except BaseException:
            self.close()
            raise

    def _initialize_runtime(self, args: Any) -> None:
        detector = from_checkpoint(str(args.model))
        class_ids = resolve_detector_class_ids(detector)
        device = 0 if torch.cuda.is_available() else "cpu"
        player_pose = PlayerPoseDetector(args.pose_model, args.pose_imgsz, device)
        self._processing_cleanup.callback(player_pose.close)
        court_pose = CourtPoseDetector(args.court_model, args.court_imgsz, device)
        self._processing_cleanup.callback(court_pose.close)
        team_classifier = JerseyTeamClassifier()
        iou_metric = IoU()
        court_mapper = CourtMapper(default_nba_court_vertices(), max_pending_gap=args.court_interval)
        capture = cv2.VideoCapture(str(args.video))
        self._processing_cleanup.callback(capture.release)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        resolver_device = f"cuda:{device}" if isinstance(device, int) else device
        player_tracker = make_tracker(fps, args)
        writer = FfmpegVideoWriter(args.output, fps, (width, height))
        self._writer_cleanup.callback(writer.release)
        if not writer.is_opened():
            raise RuntimeError(f"cannot write video: {args.output}")
        court_map = CourtMiniMap(width, height)
        ball_tracker = ShotBallTracker(fps, args.trail_length)
        holder_state = HolderState(args.holder_hold_frames)
        identity_resolver = IdentityResolver({"weights": str(args.reid_model)}, fps, resolver_device, team_classifier)
        jersey_reader = JerseyNumberReader({"checkpoint": args.jersey_ocr_model}, resolver_device)
        jersey_sampler = JerseyNumberSampler(jersey_reader)
        event_pipeline = BallEventPipeline(fps)
        start_time = time.perf_counter()
        progress_total = (
            min(total_frames, args.max_frames)
            if total_frames > 0 and args.max_frames is not None
            else total_frames
        )
        progress = tqdm(
            total=progress_total if total_frames > 0 else None,
            desc=f"Processing {args.video.name}",
            unit="frame",
            mininterval=0.5,
        )
        self._processing_cleanup.callback(progress.close)
        player_pose.start()
        court_pose.start()
        state = VideoAnalysisContext(
            args=args,
            detector=detector,
            class_ids=class_ids,
            device=device,
            player_pose=player_pose,
            court_pose=court_pose,
            court_mapper=court_mapper,
            team_classifier=team_classifier,
            player_states={},
            verified_player_ids=set(),
            previous_display_tracks=[],
            iou_metric=iou_metric,
            capture=capture,
            fps=fps,
            width=width,
            player_tracker=player_tracker,
            writer=writer,
            court_map=court_map,
            ball_tracker=ball_tracker,
            holder_state=holder_state,
            identity_resolver=identity_resolver,
            jersey_sampler=jersey_sampler,
            event_pipeline=event_pipeline,
            start_time=start_time,
            progress=progress,
            scores={"dark": 0, "light": 0},
            score_queue=[],
            event_count=0,
            reminder_count=0,
            context_rows=deque(maxlen=180),
            team_history=deque(maxlen=50),
            last_event_display=None,
            last_event_display_frame=None,
            last_event_players=[],
            artifact_writer=None,
            frame_index=0,
            detector_shape=None,
            frame_processing_elapsed=0.0,
        )
        self.state = state
        self.observer = FrameObserver(state)
        self.feature_builder = FrameFeatureBuilder(state)
        self.event_processor = FrameEventProcessor(state)
        self.renderer = FrameRenderer(court_map)

    def close(self) -> None:
        """Release all resources, including those opened before a setup error."""
        try:
            self._processing_cleanup.close()
        finally:
            self._writer_cleanup.close()

    def _write_frame(self, frame: np.ndarray, result: FrameEventResult) -> None:
        state = self.state
        if state.artifact_writer is None:
            raise RuntimeError("video artifact writer is not initialized")
        state.artifact_writer.write_frame(
            frame,
            result.frame_record,
            result.frame_time,
            partial(
                draw_scoreboard,
                scores=state.scores,
                video_time=result.frame_time,
                team_colors=(
                    state.team_classifier.team_colors
                    if state.team_classifier.team_colors_locked
                    else None
                ),
                right_clearance=state.width - state.court_map.margin - state.court_map.width,
            ),
            lambda: f"{state.frame_index / max(time.perf_counter() - state.start_time, 1e-06):.1f} FPS",
            partial(draw_fps_and_event_status, event=result.event_display),
        )

    def _finish_artifacts(self, result: VideoFinalizationResult) -> None:
        state = self.state
        output = state.artifact_writer
        if output is None:
            raise RuntimeError("video artifact writer is not initialized")
        if output.last_frame_record is not None:
            output.last_frame_record.setdefault("events", []).extend(result.flushed_events)
            output.last_frame_record.update(result.status_fields)
            output.last_frame_record["scoreboard"] = result.scoreboard
            output.last_frame_record["shot_tracking"]["phase"] = result.shot_phase
            output.last_frame_record["shot_reminders"] = result.shot_reminders
        output.finish(
            partial(
                draw_video_overlays,
                scores=state.scores,
                video_time=output.last_frame_time,
                team_colors=(
                    state.team_classifier.team_colors
                    if state.team_classifier.team_colors_locked
                    else None
                ),
                right_clearance=state.width - state.court_map.margin - state.court_map.width,
                fps_label=output.last_fps_label or "",
                event_display=result.event_display,
            )
        )

    def _run_video(self) -> None:
        state = self.state
        while True:
            has_frame, frame = state.capture.read()
            if not has_frame or (state.args.max_frames is not None and state.frame_index >= state.args.max_frames):
                break
            state.frame_index += 1
            frame_context = self.observer.observe_frame(frame)
            self.feature_builder.build_rim_feature(frame_context)
            self.feature_builder.build_frame_features(frame, frame_context)
            result = self.event_processor.process_frame(frame_context)
            self.renderer.draw_rim(frame, frame_context)
            self.renderer.draw_frame(frame, frame_context)
            self._write_frame(frame, result)
            state.progress.update(1)
        state.frame_processing_elapsed = time.perf_counter() - state.start_time

    def run(self) -> None:
        state = self.state
        try:
            with open(state.args.tracks, "w", encoding="utf-8") as track_file:
                state.artifact_writer = FrameArtifactWriter(state.writer, track_file)
                try:
                    self._run_video()
                finally:
                    self._processing_cleanup.close()
                finalization = self.event_processor.finalize_video()
                self._finish_artifacts(finalization)
        finally:
            self.close()
        finalize_identity_records(
            state.args.tracks,
            state.jersey_sampler,
            state.identity_resolver,
            partial(
                rewrite_identity_overlay,
                state.args.video,
                state.args.output,
                state.fps,
                state.team_classifier.team_colors,
            ),
            label_for_player=player_overlay_label,
        )
        _print_video_summary(state)


__all__ = ["VideoAnalysisRunner"]
