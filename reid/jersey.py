"""Resolve online jersey evidence and final number consensus for player identities."""

from collections import Counter, defaultdict, deque
from typing import Any

from reid.crops import crop_bounds_overlap, jersey_torso_crop


class JerseyNumberSampler:
    """Collect and retain full-number and per-digit OCR predictions per local track."""

    def __init__(self, reader: Any) -> None:
        self.reader = reader
        self.last_sample_frame: dict[int, int] = {}
        self.predictions: dict[tuple[int, int], dict[str, Any]] = {}
        self.digit_predictions: dict[tuple[int, int, str], dict[str, Any]] = {}

    def reset_interval(self) -> None:
        """Allow immediate OCR sampling after a clip boundary while retaining prior evidence."""
        self.last_sample_frame.clear()

    def sample(
        self,
        frame: Any,
        tracks: list[Any],
        poses: dict[int, Any],
        referee_boxes: Any,
        resolver: Any,
        frame_index: int,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Crop eligible jerseys, predict number/digit candidates, and return online votes."""
        parameters = self.reader.parameters
        if not parameters.get("enabled", True) or self.reader.error is not None:
            return []

        base_interval = max(1, int(parameters.get("sample_interval_frames", 1)))
        pad = int(parameters.get("referee_clearance_px", 8))
        crops, local_ids = [], []
        for track in tracks:
            local_id = int(track.track_id)
            interval = resolver.jersey_sampling_interval(
                local_id, frame_index, base_interval, fps
            )
            if frame_index - self.last_sample_frame.get(local_id, frame_index - interval) < interval:
                continue
            self.last_sample_frame[local_id] = int(frame_index)
            crop = jersey_torso_crop(
                frame, track.box, poses.get(local_id), parameters, include_bounds=True
            )
            if crop is None:
                continue
            if crop_bounds_overlap(crop["bounds"], referee_boxes, pad):
                resolver.note_jersey_referee_rejection()
                continue
            local_ids.append(local_id)
            crops.append(crop)

        predictions = self.reader.predict(crops)
        digit_crops, digit_refs, samples = [], [], []
        for local_id, crop, prediction in zip(local_ids, crops, predictions):
            ref = (int(frame_index), local_id)
            self.predictions[ref] = prediction
            number_crop = crop["full"]
            if len(str(prediction.get("number", ""))) == 2:
                middle = number_crop.shape[1] // 2
                digit_crops.extend((
                    {"full": number_crop[:, :middle]},
                    {"full": number_crop[:, middle:]},
                ))
                digit_refs.extend(((*ref, "left"), (*ref, "right")))
            samples.append({"local_id": local_id, "prediction": prediction})

        digit_predictions = self.reader.predict(digit_crops)
        for ref, prediction in zip(digit_refs, digit_predictions):
            self.digit_predictions[ref] = prediction
        return samples


def get_trusted_jersey_key(state: Any) -> tuple[int, str] | None:
    """Return a team-number key only after its online evidence passes lock limits."""
    key = state.jersey_confirmed_key
    if (
        not state.jersey_locked
        or key is None
        or state.jersey_lock_support < 4
        or state.jersey_lock_confidence < 0.80
        or state.jersey_lock_purity < 0.85
        or state.jersey_lock_span < 12
    ):
        return None
    return int(key[0]), str(key[1])


def resolve_jersey_team(resolver: Any, local_state: Any, frame_index: int) -> int | None:
    """Resolve a track's jersey team from recent, confident team observations."""
    history = local_state.jersey_team_history
    window = max(1, int(resolver.parameters.get("jersey_ocr", {}).get("team_window_frames", 90)))
    while history and int(frame_index) - history[0][0] > window:
        history.popleft()
    return resolver.get_confident_team(Counter(team for _, team in history))


def lock_jersey_number(
    resolver: Any,
    local_state: Any,
    number: str,
    frame_index: int,
    support: int,
    confidence: float,
    purity: float,
    span: int,
) -> None:
    """Commit a confirmed jersey number and its evidence metadata to a local track."""
    if not local_state.jersey_locked:
        resolver.jersey_lock_count += 1
    local_state.jersey_number = str(number)
    local_state.jersey_candidate = str(number)
    local_state.jersey_locked = True
    local_state.jersey_lock_frame = int(frame_index)
    local_state.jersey_lock_support = int(support)
    local_state.jersey_lock_confidence = float(confidence)
    local_state.jersey_lock_purity = float(purity)
    local_state.jersey_lock_span = int(span)


def register_locked_jersey(
    resolver: Any,
    local_id: int,
    number: str,
    frame_index: int,
    visible_local_ids: set[int],
) -> bool:
    """Bind a locked number to a team unless another visible player owns that pair."""
    local_id = int(local_id)
    local_state = resolver.local_tracks[local_id]
    team_id = resolve_jersey_team(resolver, local_state, frame_index)
    if team_id is None:
        return False
    key = (int(team_id), str(number))
    if any(
        owner != local_id
        and owner in visible_local_ids
        and state.jersey_confirmed_key == key
        for owner, state in resolver.local_tracks.items()
    ):
        return False
    if local_state.jersey_confirmed_key != key:
        resolver.jersey_confirmation_count += 1
    local_state.jersey_confirmed_key = key
    return True


def update_jersey_votes(
    resolver: Any,
    frame_index: int,
    samples: list[dict[str, Any]],
    visible_local_ids: set[int],
) -> None:
    """Update per-track online jersey votes and lock stable numbers."""
    if not samples:
        return
    jersey_parameters = resolver.parameters.get("jersey_ocr", {})
    required = max(4, int(jersey_parameters.get("confirmations", 4)))
    purity_threshold = float(jersey_parameters.get("purity", 0.85))
    min_span = max(12, int(jersey_parameters.get("min_span_frames", 12)))
    min_read_confidence = float(jersey_parameters.get("min_read_confidence", 0.55))
    min_vote_confidence = float(jersey_parameters.get("confidence", 0.80))
    window = max(1, int(jersey_parameters.get("window_frames", 120)))
    candidate_threshold = max(0.65, min_read_confidence)
    sample_interval = max(1, int(jersey_parameters.get("sample_interval_frames", 1)))
    candidate_change_confirmations = max(3, required)
    candidate_change_span = max(12, 2 * sample_interval)
    candidate_change_purity = 0.75
    candidate_change_confidence = max(0.70, candidate_threshold)
    for sample in samples:
        local_id = int(sample["local_id"])
        local_state = resolver.local_tracks[local_id]
        prediction = sample.get("prediction") or {}
        number = str(prediction.get("number", ""))
        confidence = float(prediction.get("confidence", 0.0))
        locked_number = local_state.jersey_number
        if locked_number is not None:
            conflicts = local_state.jersey_conflict_votes
            team_id = resolve_jersey_team(resolver, local_state, frame_index)
            if local_state.jersey_confirmed_key is None and team_id is not None:
                register_locked_jersey(resolver, local_id, locked_number, frame_index, visible_local_ids)
            while conflicts and frame_index - conflicts[0][0] > 24:
                conflicts.popleft()
            if number == str(locked_number) and confidence >= candidate_threshold:
                conflicts.clear()
            elif number.isdigit() and len(number) <= 2 and confidence >= 0.95:
                conflicts.append((int(frame_index), team_id, number, confidence))
                matching = [vote for vote in conflicts if vote[2] == number]
                if (
                    len(matching) >= 3
                    and matching[-1][0] - matching[-3][0] >= max(6, min_span)
                ):
                    local_state.jersey_confirmed_key = None
                    local_state.jersey_number = None
                    local_state.jersey_locked = False
                    local_state.jersey_lock_frame = None
                    local_state.jersey_lock_support = 0
                    local_state.jersey_lock_confidence = 0.0
                    local_state.jersey_lock_purity = 0.0
                    local_state.jersey_lock_span = 0
                    local_state.jersey_candidate = number
                    local_state.jersey_votes = deque(matching)
                    local_state.jersey_conflict_votes.clear()
                    resolver.jersey_conflict_reset_count += 1
            continue
        if (
            not number.isdigit()
            or len(number) > 2
            or confidence < candidate_threshold
        ):
            continue
        team_id = resolve_jersey_team(resolver, local_state, frame_index)
        votes = local_state.jersey_votes
        while votes and frame_index - votes[0][0] > window:
            votes.popleft()
        if team_id is None:
            votes = deque(vote for vote in votes if frame_index - vote[0] <= 30)
            vote_team = None
        else:
            votes = deque(vote for vote in votes if vote[1] in (None, int(team_id)))
            vote_team = int(team_id)
        local_state.jersey_votes = votes
        votes.append((int(frame_index), vote_team, number, confidence))

        weights = defaultdict(float)
        vote_frames = defaultdict(list)
        vote_counts = Counter()
        for vote_frame, _, jersey, vote_confidence in votes:
            weights[jersey] += vote_confidence
            vote_frames[jersey].append(vote_frame)
            vote_counts[jersey] += 1
        winning_number = max(weights, key=weights.get)
        support = vote_counts[winning_number]
        winning_weight = weights[winning_number]
        vote_purity = winning_weight / max(1e-6, sum(weights.values()))
        mean_confidence = winning_weight / max(1, support)
        support_frames = vote_frames[winning_number]
        vote_span = max(support_frames) - min(support_frames)

        if local_state.jersey_candidate is None:
            local_state.jersey_candidate = number
            resolver.jersey_candidate_count += 1
        elif (
            winning_number != local_state.jersey_candidate
            and support >= candidate_change_confirmations
            and vote_span >= candidate_change_span
            and vote_purity >= candidate_change_purity
            and mean_confidence >= candidate_change_confidence
        ):
            local_state.jersey_candidate = winning_number
            resolver.jersey_candidate_change_count += 1

        if team_id is None:
            confirmed = (
                support >= 4
                and vote_span >= 18
                and vote_purity >= 0.85
                and mean_confidence >= 0.80
            )
            if confirmed:
                lock_jersey_number(
                    resolver, local_state, winning_number, frame_index,
                    support, mean_confidence, vote_purity, vote_span,
                )
                votes.clear()
            continue

        confirmed = (
            support >= required
            and vote_span >= min_span
            and vote_purity >= purity_threshold
            and mean_confidence >= min_vote_confidence
        )
        if not confirmed:
            continue
        lock_jersey_number(
            resolver, local_state, winning_number, frame_index,
            support, mean_confidence, vote_purity, vote_span,
        )
        register_locked_jersey(resolver, local_id, winning_number, frame_index, visible_local_ids)
        local_state.jersey_votes.clear()




__all__ = [
    "JerseyNumberSampler",
    "get_trusted_jersey_key",
    "lock_jersey_number",
    "register_locked_jersey",
    "resolve_jersey_team",
    "update_jersey_votes",
]
