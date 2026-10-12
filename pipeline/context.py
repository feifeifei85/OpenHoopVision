from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameContext:
    """Detections and derived features for one video frame."""

    players: Any
    balls: Any
    rims: Any
    holders: Any
    scorers: Any
    referees: Any
    visible_tracks: list[Any]
    display_only_tracks: list[Any]
    reid_tracks: list[Any]
    reid_track_ids: set[int]
    visible_by_id: dict[int, Any]
    active_by_id: dict[int, Any]
    active_ids: set[int]
    player_poses: dict[int, Any]
    ball_detections: list[Any]
    rim_feature: dict | None = None
    rim_overlay_box: Any | None = None
    rim_overlay_score: float | None = None
    feature_players: list[dict[str, Any]] = field(default_factory=list)
    overlay_players: list[dict[str, Any]] = field(default_factory=list)
    map_entries: list[dict[str, Any]] = field(default_factory=list)
    player_box_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    display_box_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    ball_track: Any = None
    ball_trail: list[tuple[float, float]] = field(default_factory=list)
    holder_id: int | None = None


@dataclass(frozen=True)
class FrameEventResult:
    """Event and record data produced for one processed frame."""

    frame_record: dict[str, Any]
    frame_time: float
    event_display: Any


@dataclass(frozen=True)
class VideoFinalizationResult:
    """Event data needed to update and render the final decoded frame."""

    flushed_events: list[dict[str, Any]]
    status_fields: dict[str, Any]
    scoreboard: dict[str, Any]
    shot_phase: str
    shot_reminders: list[dict[str, Any]]
    event_display: Any


@dataclass
class VideoAnalysisContext:
    """Shared runtime state passed between video-analysis stages."""

    args: Any
    detector: Any
    class_ids: Any
    device: Any
    player_pose: Any
    court_pose: Any
    court_mapper: Any
    team_classifier: Any
    player_states: dict[int, Any]
    verified_player_ids: set[int]
    previous_display_tracks: list[Any]
    iou_metric: Any
    capture: Any
    fps: float
    width: int
    player_tracker: Any
    writer: Any
    court_map: Any
    ball_tracker: Any
    holder_state: Any
    identity_resolver: Any
    jersey_sampler: Any
    event_pipeline: Any
    start_time: float
    progress: Any
    scores: dict[str, int]
    score_queue: list[Any]
    event_count: int
    reminder_count: int
    context_rows: Any
    team_history: Any
    last_event_display: Any
    last_event_display_frame: int | None
    last_event_players: list[dict[str, Any]]
    artifact_writer: Any
    frame_index: int
    detector_shape: Any
    frame_processing_elapsed: float
