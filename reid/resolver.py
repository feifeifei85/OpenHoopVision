"""Resolve short-lived tracker IDs to stable player identities."""

import bisect
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from models.appearance import SportsReIDEncoder
from tracking.ball import normalize_fps
from reid.crops import build_player_crop, jersey_crop_overlaps_referee
from reid.jersey import get_trusted_jersey_key, update_jersey_votes
from reid.matching import match_new_tracks


@dataclass(slots=True)
class GlobalIdentity:
    global_id: int
    features: deque
    prototype: np.ndarray | None = None
    last_seen_frame: int = 0
    active_local_id: int | None = None
    team_counts: dict[int, int] = field(default_factory=dict)
    feature_version: int = 0
    jersey_counts: dict[tuple[int, str], int] = field(default_factory=dict)


@dataclass(slots=True)
class LocalIdentityTrack:
    created_frame: int
    features: list | None = None
    identity_feature_count: int = 0
    team_counts: dict[int, int] | None = None
    reid_resolved: bool = False
    reid_start_frame: int | None = None
    reid_last_frame: int = -1
    jersey_identity_key_added: tuple[int, str] | None = None
    jersey_lock_support: int = 0
    jersey_lock_confidence: float = 0.0
    jersey_lock_purity: float = 0.0
    jersey_lock_frame: int | None = None
    jersey_lock_span: int = 0
    match_similarity_cache: dict = field(default_factory=dict)
    jersey_votes: deque = field(default_factory=deque)
    jersey_conflict_votes: deque = field(default_factory=deque)
    jersey_team_history: deque = field(default_factory=deque)
    jersey_confirmed_key: tuple[int, str] | None = None
    jersey_number: str | None = None
    jersey_candidate: str | None = None
    jersey_locked: bool = False
    referee_overlap_streak: int = 0

    @property
    def feature_count(self):
        return len(self.features or ())


class IdentityResolver:
    """Map short-lived McByte IDs to stable IDs used by event and output records."""

    def __init__(self, parameters, fps, device, team_classifier):
        self.parameters = dict(parameters or {})
        weights = self.parameters.get("weights")
        if weights is None:
            raise ValueError("ReID model path must be configured")
        self.parameters["weights"] = str(weights)
        self.fps = normalize_fps(fps)
        self.team_classifier = team_classifier
        self.encoder = SportsReIDEncoder(device, self.parameters["weights"])
        self.local_to_global = {}
        self.local_id_to_global = {}
        self.local_identity_segments = defaultdict(list)
        self.identity_aliases = {}
        self.identity_relink_count = 0
        self.jersey_confirmation_count = 0
        self.jersey_candidate_count = 0
        self.jersey_candidate_change_count = 0
        self.jersey_lock_count = 0
        self.team_switch_split_count = 0
        self.jersey_conflict_reset_count = 0
        self.local_track_birth_frames = {}
        self.local_tracks = {}
        self.identities = {}
        self.feature_gallery = {}
        # Keep the matchable gallery in the same global-ID order as identities.
        self.feature_gallery_ids = []
        self.next_global_id = 1
        self.total_reid_ms = 0.0
        self.total_crops = 0
        self.match_count = 0
        self.clip_cut_frame = None
        self.clip_cut_title = None
        self.last_clip_cut_frame = -(10**9)
        self.clip_cut_count = 0
        self.clip_cut_frame_index = None
        self.jersey_referee_rejects = 0
        self.visible_local_ids = set()

    def _reset_clip_identities(self):
        for identity in self.identities.values():
            identity.active_local_id = None
        self.local_to_global.clear()
        self.local_id_to_global.clear()
        self.local_tracks.clear()
        self.local_track_birth_frames.clear()
        self.visible_local_ids.clear()
        # A camera cut starts new local tracks, but both same-video indexes
        # remain useful when players reappear in the new shot.
        self.team_classifier.reset_segment_locks()

    def _detect_clip_cut(self, frame, frame_index):
        clip_cut = self.parameters.get("clip_cut", {})
        if not clip_cut.get("enabled", True):
            return False
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = clip_cut.get("title_roi", (0.015, 0.847, 0.085, 0.91))
        title = cv2.cvtColor(
            frame[
                int(y1 * height) : max(int(y1 * height) + 1, int(y2 * height)),
                int(x1 * width) : max(int(x1 * width) + 1, int(x2 * width)),
            ],
            cv2.COLOR_BGR2GRAY,
        )
        title = (title > int(clip_cut.get("title_threshold", 190))).astype(np.float32)
        scene = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        cut = False
        if self.clip_cut_frame is not None and self.clip_cut_title is not None:
            scene_delta = float(np.mean(cv2.absdiff(scene, self.clip_cut_frame)))
            title_delta = float(np.mean(np.abs(title - self.clip_cut_title)))
            cut = (
                scene_delta >= float(clip_cut.get("scene_diff", 45.0))
                and title_delta >= float(clip_cut.get("title_diff", 0.02))
                and int(frame_index) - self.last_clip_cut_frame
                >= int(clip_cut.get("min_gap_frames", 60))
            )
        self.clip_cut_frame = scene
        self.clip_cut_title = title
        if cut:
            self.last_clip_cut_frame = int(frame_index)
            self.clip_cut_count += 1
            self._reset_clip_identities()
        return cut

    def detect_clip_cut(self, frame, frame_index):
        frame_index = int(frame_index)
        clip_cut = self._detect_clip_cut(frame, frame_index)
        self.clip_cut_frame_index = frame_index
        return clip_cut

    def get_jersey_number(self, local_id):
        local_state = self.local_tracks.get(int(local_id))
        if local_state is not None and local_state.jersey_candidate is not None:
            return str(local_state.jersey_candidate)
        return None if local_state is None else local_state.jersey_number

    def is_jersey_number_locked(self, local_id):
        local_state = self.local_tracks.get(int(local_id))
        if local_state is None:
            return False
        return local_state.jersey_locked

    def jersey_sampling_interval(self, local_id, frame_index, base_interval, fps):
        """Choose OCR sampling frequency from track age and lock state."""
        local_state = self.local_tracks.get(int(local_id))
        interval = max(1, int(base_interval))
        if local_state is None:
            return interval
        sample_start = local_state.reid_start_frame
        if sample_start is None:
            sample_start = local_state.created_frame
        age = int(frame_index) - int(sample_start)
        if local_state.jersey_locked:
            return max(interval, 5)
        if age > float(fps):
            return max(interval, 2)
        return interval

    def note_jersey_referee_rejection(self):
        """Count an OCR crop rejected because it overlaps a referee."""
        self.jersey_referee_rejects += 1

    def get_identity_label(self, local_id, global_id):
        local_id = int(local_id)
        local_state = self.local_tracks.get(local_id)
        if global_id is None:
            return f"R{local_id}"
        prefix = "G" if local_state is not None and local_state.reid_resolved else "R"
        return f"{prefix}{global_id}"

    def get_team_id(self, local_id):
        local_id = int(local_id)
        local_state = self.local_tracks.get(local_id)
        if local_state is not None and local_state.jersey_locked and local_state.jersey_confirmed_key is not None:
            return int(local_state.jersey_confirmed_key[0])
        global_id = self.local_to_global.get(local_id)
        identity = self.identities.get(global_id) if global_id is not None else None
        identity_team = None if identity is None else self.get_identity_team(global_id)
        if identity_team is not None:
            return identity_team
        team_id = self.team_classifier.predict_identity_track(local_id)
        return None if team_id is None else int(team_id)

    def get_identity_team(self, global_id):
        identity = self.identities.get(self.canonicalize_global_id(global_id))
        if identity is None:
            return None
        jersey_teams = {
            int(team_id)
            for (team_id, _), support in identity.jersey_counts.items()
            if support > 0
        }
        if len(jersey_teams) == 1:
            return next(iter(jersey_teams))
        return self.get_confident_team(identity.team_counts)

    @staticmethod
    def _normalize_feature(feature):
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        return feature / (float(np.linalg.norm(feature)) + 1e-12)

    def canonicalize_global_id(self, global_id):
        if global_id is None:
            return None
        try:
            global_id = int(global_id)
        except (TypeError, ValueError):
            return None
        path = []
        seen = set()
        while global_id in self.identity_aliases and global_id not in seen:
            seen.add(global_id)
            path.append(global_id)
            global_id = int(self.identity_aliases[global_id])
        for alias in path:
            self.identity_aliases[alias] = global_id
        return global_id

    def alias_global_identity(self, previous_id: int | None, canonical_id: int | None) -> None:
        """Merge one obsolete global ID into its canonical identity."""
        previous_id = self.canonicalize_global_id(previous_id)
        canonical_id = self.canonicalize_global_id(canonical_id)
        if previous_id is None or canonical_id is None or previous_id == canonical_id:
            return
        previous = self.identities.get(previous_id)
        canonical = self.identities.get(canonical_id)
        if previous is not None and canonical is not None:
            for feature in previous.features:
                self.append_feature(canonical, feature)
            for key, count in previous.jersey_counts.items():
                canonical.jersey_counts[key] = canonical.jersey_counts.get(key, 0) + count
            for team_id, count in previous.team_counts.items():
                canonical.team_counts[team_id] = canonical.team_counts.get(team_id, 0) + count
        self.identity_aliases[previous_id] = canonical_id
        self.identity_relink_count += 1

    def append_feature(self, identity: GlobalIdentity, feature: np.ndarray) -> None:
        """Add one normalized appearance vector to a global identity's feature bank."""
        feature = self._normalize_feature(feature)
        identity.features.append(feature)
        identity.prototype = self._normalize_feature(np.mean(np.stack(identity.features), axis=0))
        identity.feature_version += 1
        global_id = int(identity.global_id)
        minimum_updates = max(2, int(self.parameters.get("reid_min_updates", 2)))
        if len(identity.features) >= minimum_updates and global_id not in self.feature_gallery:
            self.feature_gallery[global_id] = identity
            bisect.insort(self.feature_gallery_ids, global_id)

    def assign_local_identity(self, local_id: int, global_id: int, frame_index: int) -> None:
        """Bind a local track to a global identity and record the segment boundary."""
        local_id, global_id, frame_index = int(local_id), int(global_id), int(frame_index)
        previous = self.local_to_global.get(local_id)
        segments = self.local_identity_segments[local_id]
        if previous == global_id:
            if not segments or segments[-1][1] != global_id:
                segments.append((frame_index, global_id))
            return
        if previous is not None:
            old_identity = self.identities.get(previous)
            if old_identity is not None and old_identity.active_local_id == local_id:
                old_identity.active_local_id = None
        self.local_to_global[local_id] = global_id
        self.local_id_to_global[local_id] = global_id
        if not segments or segments[-1][1] != global_id:
            segments.append((frame_index, global_id))
        identity = self.identities.get(global_id)
        if identity is not None:
            identity.active_local_id = local_id

    def _create_identity(self, local_id, frame_index, feature=None):
        global_id = self.next_global_id
        self.next_global_id += 1
        identity = GlobalIdentity(
            global_id=global_id,
            features=deque(maxlen=int(self.parameters.get('feature_bank_size', 50))),
            last_seen_frame=int(frame_index),
            active_local_id=int(local_id),
        )
        self.identities[global_id] = identity
        self.assign_local_identity(local_id, global_id, frame_index)
        if feature is not None:
            self.append_feature(identity, feature)
        return global_id


    def _unbind_local_identity(self, local_id, frame_index):
        local_id = int(local_id)
        global_id = self.local_to_global.pop(local_id, None)
        self.local_id_to_global.pop(local_id, None)
        if global_id is not None:
            identity = self.identities.get(global_id)
            if identity is not None and identity.active_local_id == local_id:
                identity.active_local_id = None
        segments = self.local_identity_segments[local_id]
        if not segments or segments[-1][1] is not None:
            segments.append((int(frame_index), None))
        local_state = self.local_tracks.get(local_id)
        if local_state is not None:
            # The gallery stores historical identity evidence across local
            # track loss; an occlusion must not erase a locked team-number pair.
            local_state.jersey_identity_key_added = None
            local_state.jersey_confirmed_key = None
            local_state.jersey_number = None
            local_state.jersey_candidate = None
            local_state.jersey_locked = False
            local_state.jersey_lock_support = 0
            local_state.jersey_lock_confidence = 0.0
            local_state.jersey_lock_purity = 0.0
            local_state.jersey_lock_frame = None
            local_state.jersey_lock_span = 0
            local_state.jersey_votes.clear()
            local_state.jersey_conflict_votes.clear()
            local_state.reid_resolved = False
            local_state.identity_feature_count = 0
            local_state.match_similarity_cache.clear()

    def _split_local_identity(self, local_id, frame_index, feature=None, *, preserve_team=False):
        local_id, frame_index = int(local_id), int(frame_index)
        self._unbind_local_identity(local_id, frame_index)
        local_state = self.local_tracks[local_id]
        local_state.created_frame = frame_index
        local_state.features = [] if feature is None else [feature.copy()]
        local_state.identity_feature_count = 0
        local_state.reid_start_frame = frame_index
        local_state.reid_last_frame = frame_index if feature is not None else -1
        local_state.team_counts = {}
        local_state.jersey_votes.clear()
        local_state.jersey_conflict_votes.clear()
        local_state.jersey_team_history.clear()
        local_state.referee_overlap_streak = 0
        local_state.reid_resolved = False
        local_state.match_similarity_cache.clear()
        self.local_track_birth_frames[local_id] = frame_index
        if not preserve_team:
            self.team_classifier.forget_track(local_id)


    def get_confident_team(self, counts: dict[int, int] | None) -> int | None:
        """Return a team only when its accumulated identity votes pass the gate."""
        counts = counts or {}
        total = sum(counts.values())
        minimum = int(self.parameters.get("team_gate_min_observations", 5))
        if total < minimum:
            return None
        team, votes = max(counts.items(), key=lambda item: item[1])
        if votes / total < float(self.parameters.get("team_gate_min_share", 0.6)):
            return None
        return int(team)


    @staticmethod
    def compute_similarity(query: np.ndarray, identity: GlobalIdentity) -> float | None:
        """Compare a normalized query vector with a gallery identity prototype."""
        if identity.prototype is None or not identity.features:
            return None
        return float(query @ identity.prototype)


    def _sync_jersey_identity(self, local_id):
        local_state = self.local_tracks.get(int(local_id))
        global_id = self.local_to_global.get(int(local_id))
        identity = self.identities.get(global_id) if global_id is not None else None
        if local_state is None or identity is None:
            return
        current_key = local_state.jersey_confirmed_key if local_state.jersey_locked else None
        previous_key = local_state.jersey_identity_key_added
        if current_key == previous_key:
            return
        if previous_key is not None:
            remaining = identity.jersey_counts.get(previous_key, 0) - 1
            if remaining > 0:
                identity.jersey_counts[previous_key] = remaining
            else:
                identity.jersey_counts.pop(previous_key, None)
        if current_key is not None:
            identity.jersey_counts[current_key] = identity.jersey_counts.get(current_key, 0) + 1
        local_state.jersey_identity_key_added = current_key

    def update(
        self,
        frame,
        tracks,
        active_ids,
        poses,
        frame_index,
        referee_boxes=(),
        jersey_samples=(),
        reid_eligible_ids=None,
    ):
        """Update visible IDs while limiting appearance matching to eligible court tracks."""
        team_classifier = self.team_classifier
        start_time = time.perf_counter()
        active_ids = {int(track_id) for track_id in active_ids}
        frame_index = int(frame_index)
        jersey_parameters = self.parameters.get("jersey_ocr", {})
        feature_gap = max(1, int(self.parameters.get("reid_min_frame_gap", 5)))
        maximum_features = max(2, int(self.parameters.get("feature_bank_size", 50)))
        match_window = max(0, int(self.parameters.get("reid_new_max_len", 10)))
        reid_eligible_ids = (
            {int(track.track_id) for track in tracks}
            if reid_eligible_ids is None
            else {int(track_id) for track_id in reid_eligible_ids}
        )
        if self.clip_cut_frame_index != frame_index:
            self.detect_clip_cut(frame, frame_index)
        self.visible_local_ids = {int(track.track_id) for track in tracks}
        for local_id, global_id in list(self.local_to_global.items()):
            identity = self.identities.get(global_id)
            if local_id not in active_ids:
                if identity is not None and identity.active_local_id == local_id:
                    identity.active_local_id = None
                self.local_to_global.pop(local_id, None)
                self.local_tracks.pop(local_id, None)
            elif identity is not None:
                identity.active_local_id = local_id

        candidates = []
        for track in tracks:
            local_id = int(track.track_id)
            reid_eligible = local_id in reid_eligible_ids
            birth_frame = self.local_track_birth_frames.setdefault(local_id, frame_index)
            local_state = self.local_tracks.get(local_id)
            if local_state is None:
                local_state = LocalIdentityTrack(
                    created_frame=birth_frame,
                    reid_resolved=frame_index == 1 and reid_eligible,
                    reid_start_frame=frame_index if reid_eligible else None,
                )
                self.local_tracks[local_id] = local_state
            if reid_eligible and local_state.reid_start_frame is None:
                local_state.reid_start_frame = frame_index
            if local_state.features is None:
                local_state.features = []
            if local_state.team_counts is None:
                local_state.team_counts = {}
            if reid_eligible and (
                local_state.jersey_confirmed_key is not None or local_state.jersey_number is not None
            ):
                referee_overlap = jersey_crop_overlaps_referee(
                    self, frame, track, poses.get(local_id), referee_boxes
                )
                local_state.referee_overlap_streak = (
                    local_state.referee_overlap_streak + 1 if referee_overlap else 0
                )
                reset_after = max(1, int(jersey_parameters.get("referee_reset_frames", 12)))
                if local_state.referee_overlap_streak >= reset_after:
                    local_state.jersey_confirmed_key = None
                    local_state.jersey_number = None
                    local_state.jersey_candidate = None
                    local_state.jersey_locked = False
                    local_state.jersey_lock_support = 0
                    local_state.jersey_lock_confidence = 0.0
                    local_state.jersey_lock_purity = 0.0
                    local_state.jersey_lock_frame = None
                    local_state.jersey_lock_span = 0
                    local_state.jersey_votes.clear()
                    local_state.jersey_conflict_votes.clear()
                    local_state.referee_overlap_streak = 0
            current_team = team_classifier.predict_identity_track(local_id) if reid_eligible else None
            team_history = local_state.team_counts or {}
            if current_team is not None and team_history:
                previous_team, previous_count = max(team_history.items(), key=lambda item: item[1])
                if (
                    int(current_team) != int(previous_team)
                    and previous_count >= 12
                    and previous_count / max(1, sum(team_history.values())) >= 0.9
                ):
                    self._split_local_identity(local_id, frame_index, preserve_team=True)
                    self.team_switch_split_count += 1
                    local_state = self.local_tracks[local_id]
            if current_team is not None:
                local_state.jersey_team_history.append((frame_index, int(current_team)))
                team_window = max(1, int(jersey_parameters.get("team_window_frames", 90)))
                while (
                    local_state.jersey_team_history
                    and frame_index - local_state.jersey_team_history[0][0] > team_window
                ):
                    local_state.jersey_team_history.popleft()
            if not reid_eligible:
                continue
            if local_state.feature_count >= maximum_features:
                continue
            if local_state.reid_last_frame >= 0 and frame_index - local_state.reid_last_frame < feature_gap:
                continue
            crop = build_player_crop(self, frame, track, poses.get(local_id))
            if crop is not None:
                candidates.append({"track": track, "crop": crop})

        jersey_state_before = {
            local_id: (state.jersey_locked, state.jersey_confirmed_key)
            for local_id, state in self.local_tracks.items()
            if local_id in self.visible_local_ids
        }
        update_jersey_votes(self, frame_index, jersey_samples, self.visible_local_ids)
        for local_id in self.visible_local_ids & reid_eligible_ids:
            self._sync_jersey_identity(local_id)
        new_feature_local_ids = set()
        if candidates:
            try:
                embeddings = self.encoder.encode([row["crop"] for row in candidates])
            except Exception as exc:
                self.encoder.error = f"{type(exc).__name__}: {exc}"
                self.encoder.model = None
                print(f"warning: OSNet ReID inference failed; identity matching is disabled: {self.encoder.error}")
                embeddings = []
            if len(embeddings) == len(candidates):
                for row, feature in zip(candidates, embeddings):
                    local_id = int(row["track"].track_id)
                    local_state = self.local_tracks[local_id]
                    local_state.features.append(self._normalize_feature(feature))
                    local_state.reid_last_frame = frame_index
                    new_feature_local_ids.add(local_id)
                self.total_crops += len(candidates)

        match_min_features = max(2, int(self.parameters.get("reid_min_updates", 2)))
        match_rows = []
        track_by_local_id = {int(track.track_id): track for track in tracks}
        newly_resolved_local_ids = set()
        for local_id, track in track_by_local_id.items():
            if local_id not in reid_eligible_ids:
                continue
            local_state = self.local_tracks[local_id]
            reid_start_frame = local_state.reid_start_frame
            if reid_start_frame is None:
                continue
            if (
                local_state.reid_resolved
                or local_state.feature_count < match_min_features
                or frame_index - reid_start_frame > match_window
            ):
                continue
            features = list(local_state.features or ())
            match_rows.append(
                {
                    "track": track,
                    "team": team_classifier.predict_identity_track(local_id),
                    "feature": features[-1],
                    "track_features": features,
                    "query_feature": self._normalize_feature(np.mean(np.stack(features), axis=0)),
                }
            )

        matched_local_ids, _ = match_new_tracks(
            self, match_rows, frame_index, active_ids
        )
        for local_id in matched_local_ids:
            local_state = self.local_tracks.get(local_id)
            if local_state is not None:
                local_state.reid_resolved = True
                local_state.match_similarity_cache.clear()

        for local_id, track in track_by_local_id.items():
            local_state = self.local_tracks[local_id]
            reid_eligible = local_id in reid_eligible_ids
            if reid_eligible:
                reid_start_frame = local_state.reid_start_frame
                if (
                    not local_state.reid_resolved
                    and reid_start_frame is not None
                    and frame_index - reid_start_frame > match_window
                ):
                    local_state.reid_resolved = True
                    newly_resolved_local_ids.add(local_id)
            if local_id not in self.local_to_global:
                global_id = self._create_identity(local_id, frame_index)
                self.identities[global_id].team_counts = dict(local_state.team_counts or {})
            identity = self.identities.get(self.local_to_global[local_id])
            if identity is None:
                continue
            if reid_eligible:
                if local_state.reid_resolved and local_state.identity_feature_count < len(local_state.features or ()):
                    for feature in (local_state.features or ())[local_state.identity_feature_count :]:
                        self.append_feature(identity, feature)
                    local_state.identity_feature_count = len(local_state.features or ())
                team_id = team_classifier.predict_identity_track(local_id)
                if team_id is not None:
                    local_state.team_counts[int(team_id)] = local_state.team_counts.get(int(team_id), 0) + 1
                    identity.team_counts[int(team_id)] = identity.team_counts.get(int(team_id), 0) + 1
            identity.active_local_id = local_id
            identity.last_seen_frame = frame_index
            if reid_eligible:
                self._sync_jersey_identity(local_id)

        late_match_rows = []
        for local_id, track in track_by_local_id.items():
            if local_id not in reid_eligible_ids:
                continue
            local_state = self.local_tracks[local_id]
            reid_start_frame = local_state.reid_start_frame
            jersey_key = get_trusted_jersey_key(local_state)
            previous_jersey_state = jersey_state_before.get(local_id)
            current_jersey_state = (local_state.jersey_locked, local_state.jersey_confirmed_key)
            # Try on a new lock immediately, otherwise retry only after two fresh crops.
            new_feature_pair = (
                local_id in new_feature_local_ids and local_state.feature_count % 2 == 0
            )
            if (
                jersey_key is None
                or not local_state.reid_resolved
                or local_state.feature_count < match_min_features
                or reid_start_frame is None
                or frame_index - reid_start_frame <= match_window
                or (
                    not new_feature_pair
                    and local_id not in newly_resolved_local_ids
                    and previous_jersey_state == current_jersey_state
                )
            ):
                continue
            features = list(local_state.features or ())
            late_match_rows.append(
                {
                    "track": track,
                    "feature": features[-1],
                    "track_features": [],
                    "query_feature": self._normalize_feature(np.mean(np.stack(features), axis=0)),
                    "late_jersey_key": jersey_key,
                }
            )
        late_matched_ids = set()
        if late_match_rows:
            late_matched_ids, _ = match_new_tracks(
                self, late_match_rows, frame_index, active_ids
            )
        for local_id in late_matched_ids:
            self.local_tracks[local_id].match_similarity_cache.clear()

        self.total_reid_ms += (time.perf_counter() - start_time) * 1000.0
        assignments = {}
        active_global_ids = set()
        for track in sorted(tracks, key=lambda item: float(item.score), reverse=True):
            local_id = int(track.track_id)
            global_id = self.local_to_global.get(local_id)
            if global_id is not None:
                if global_id in active_global_ids:
                    global_id = None
                else:
                    active_global_ids.add(global_id)
            stored_global_id = self.local_to_global.get(local_id)
            if stored_global_id is not None:
                segments = self.local_identity_segments[local_id]
                if not segments or segments[-1][1] != global_id:
                    segments.append((frame_index, global_id))
            assignments[local_id] = global_id
        return assignments

    def get_global_id_at(self, local_id, frame_index=None):
        local_id = int(local_id)
        if frame_index is None:
            return self.canonicalize_global_id(self.local_id_to_global.get(local_id))
        for start_frame, global_id in reversed(self.local_identity_segments.get(local_id, ())):
            if int(start_frame) <= int(frame_index):
                return self.canonicalize_global_id(global_id)
        return None
