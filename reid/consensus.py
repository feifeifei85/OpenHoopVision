"""Offline jersey-number consensus over cached OCR and online evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from reid.records import assign_identity_labels

PlayerRow = dict[str, Any]
PlayerSegment = list[tuple[int, PlayerRow]]
Record = dict[str, Any]


@dataclass(frozen=True)
class JerseyConsensusResult:
    """Summary of number and label changes made to the supplied records."""

    added: int
    corrected: int
    demoted: int
    status_changed: bool
    identity_labels_changed: bool
    labeled: int
    total_players: int


def _split_on_jersey_color(segment: PlayerSegment) -> list[PlayerSegment]:
    if len(segment) < 18:
        return [segment]
    boundary = None
    largest_change = 0.22
    for index in range(8, len(segment) - 8):
        before = [
            float(player["jersey_l"])
            for _, player in segment[max(0, index - 8):index]
            if player.get("jersey_l") is not None
        ]
        after = [
            float(player["jersey_l"])
            for _, player in segment[index:index + 8]
            if player.get("jersey_l") is not None
        ]
        if len(before) < 3 or len(after) < 3:
            continue
        change = abs(float(np.median(after) - np.median(before)))
        if change > largest_change:
            boundary, largest_change = index, change
    if boundary is None:
        return [segment]
    return _split_on_jersey_color(segment[:boundary]) + _split_on_jersey_color(segment[boundary:])


def _compute_number_consensus(
    votes: list[tuple[int, str, float]],
    minimum: int,
    purity_limit: float = 0.75,
    confidence_limit: float = 0.80,
    min_span: int = 6,
) -> str | None:
    weights = defaultdict(float)
    counts = Counter()
    frames = defaultdict(list)
    for frame_index, number, confidence in votes:
        weights[number] += confidence
        counts[number] += 1
        frames[number].append(frame_index)
    if not weights:
        return None
    number, weight = max(weights.items(), key=lambda item: item[1])
    purity = weight / sum(weights.values())
    mean_confidence = weight / counts[number]
    if (
        counts[number] < minimum
        or purity < purity_limit
        or mean_confidence < confidence_limit
        or frames[number][-1] - frames[number][0] < min_span
    ):
        return None
    return number


def _is_number_locked(player: PlayerRow) -> bool:
    return bool(player.get("jersey_number_locked", player.get("jersey_number") is not None))


def _propagate_global_number_consensus(records: list[Record]) -> tuple[int, int]:
    votes = defaultdict(Counter)
    for record in records:
        for player in record.get("players", ()):
            global_id = player.get("id")
            if (
                global_id is not None
                and str(player.get("identity_label", "")).startswith("G")
                and player.get("jersey_number") is not None
                and _is_number_locked(player)
            ):
                votes[global_id][str(player["jersey_number"])] += 1

    added = corrected = 0
    for record in records:
        for player in record.get("players", ()):
            global_id = player.get("id")
            number_votes = votes.get(global_id)
            if (
                global_id is None
                or not str(player.get("identity_label", "")).startswith("G")
                or _is_number_locked(player)
                or len(number_votes or ()) != 1
                or next(iter(number_votes.values()), 0) < 3
            ):
                continue
            stable_number = next(iter(number_votes))
            existing = player.get("jersey_number")
            corrected += existing not in (None, stable_number)
            player["jersey_number"] = stable_number
            player["jersey_number_locked"] = True
            added += existing is None
    return added, corrected


def _select_best_digit(
    votes: list[tuple[int, str, float]], purity_limit: float
) -> tuple[str, set[int], float] | None:
    weights = defaultdict(float)
    frames = defaultdict(set)
    counts = Counter()
    for frame_index, digit, confidence in votes:
        weights[digit] += confidence
        frames[digit].add(frame_index)
        counts[digit] += 1
    if not weights:
        return None
    digit, weight = max(weights.items(), key=lambda item: item[1])
    total = sum(weights.values())
    count = counts[digit]
    if (
        len(frames[digit]) < 4
        or weight / max(total, 1e-6) < purity_limit
        or weight / max(count, 1) < 0.65
    ):
        return None
    return digit, frames[digit], weight / max(total, 1e-6)


def reconcile_jersey_number_records(
    records: list[Record],
    predictions: dict[tuple[int, int], dict[str, Any]],
    digit_predictions: dict[tuple[int, int, str], dict[str, Any]],
    initial_states: dict[tuple[int, Any, int], tuple[bool, Any, str, tuple[Any, Any]]],
) -> JerseyConsensusResult:
    """Apply cached OCR and locked-number evidence to frame records in place."""
    by_local_id = defaultdict(list)
    for frame_index, record in enumerate(records, 1):
        for player in record.get("players", ()):
            local_id = player.get("local_track_id")
            if local_id is not None:
                local_id = int(local_id)
                by_local_id[local_id].append((frame_index, player))

    segments = []
    for local_id, rows in by_local_id.items():
        segment = []
        for frame_index, player in rows:
            previous = segment[-1] if segment else None
            previous_id = previous[1].get("id") if previous else None
            global_id = player.get("id")
            if previous and (
                frame_index - previous[0] > 8
                or (previous_id is not None and global_id is not None and previous_id != global_id)
            ):
                segments.extend(_split_on_jersey_color(segment))
                segment = []
            segment.append((frame_index, player))
        if segment:
            segments.extend(_split_on_jersey_color(segment))

    resolved_segments = []
    for segment in segments:
        ocr_votes, online_votes = [], []
        global_ids = {
            p.get("id")
            for _, p in segment
            if p.get("id") is not None
            and str(p.get("identity_label", "")).startswith("G")
        }
        for frame_index, player in segment:
            ref = (frame_index, int(player["local_track_id"]))
            prediction = predictions.get(ref, {})
            number = str(prediction.get("number", ""))
            confidence = float(prediction.get("confidence", 0.0))
            if number.isdigit() and len(number) <= 2 and confidence >= 0.90:
                ocr_votes.append((frame_index, number, confidence))
            existing = player.get("jersey_number")
            if existing is not None and _is_number_locked(player):
                online_votes.append((frame_index, str(existing), 1.0))
        online_number = _compute_number_consensus(online_votes, 3)
        ocr_number = _compute_number_consensus(
            ocr_votes, 3, purity_limit=0.65, confidence_limit=0.90, min_span=12
        )
        conflict = (
            online_number is not None
            and ocr_number is not None
            and ocr_number != online_number
        )
        number = None if conflict else online_number or ocr_number
        global_id = next(iter(global_ids)) if len(global_ids) == 1 else None
        resolved_segments.append({
            "rows": segment,
            "number": number,
            "global_id": global_id,
            "conflict": conflict,
            "ocr_votes": ocr_votes,
        })
    need_first_digit = {
        ref for ref, prediction in predictions.items()
        if len(str(prediction.get("number", ""))) == 2
    }

    direct_numbers = {}
    for ref in need_first_digit:
        full = predictions.get(ref, {})
        full_number = str(full.get("number", ""))
        full_confidence = float(full.get("confidence", 0.0))
        left = digit_predictions.get((*ref, "left"), {})
        right = digit_predictions.get((*ref, "right"), {})
        left_number = str(left.get("number", ""))
        right_number = str(right.get("number", ""))
        left_confidence = float(left.get("confidence", 0.0))
        right_confidence = float(right.get("confidence", 0.0))
        if (
            len(full_number) == 2
            and left_number[:1].isdigit()
            and right_number[-1:].isdigit()
            and left_confidence >= 0.75
            and right_confidence >= 0.75
            and full_confidence >= 0.90
            and right_number[-1] == full_number[-1]
        ):
            direct_numbers[ref] = left_number[0] + right_number[-1]
        elif (
            len(full_number) == 2
            and left_number == full_number
            and left_confidence >= 0.90
            and full_confidence >= 0.90
        ):
            direct_numbers[ref] = full_number

    for item in resolved_segments:
        direct_votes = []
        for frame_index, player in item["rows"]:
            ref = (frame_index, int(player["local_track_id"]))
            if ref in direct_numbers:
                direct_votes.append((frame_index, direct_numbers[ref], 0.95))
        direct_number = _compute_number_consensus(direct_votes, 2, min_span=3)
        if (
            direct_number is not None
            and not item["conflict"]
            and (
                item["number"] is None
                or item["number"] == direct_number
                or direct_number.startswith(item["number"])
                or item["global_id"] is None
            )
        ):
            item["direct_number"] = direct_number
            item["number"] = direct_number

    trusted_numbers, trusted_refs = {}, set()
    for item in resolved_segments:
        first_votes, last_votes = [], []
        full_support, left_support = defaultdict(set), defaultdict(set)
        full_last_frames = defaultdict(set)
        paired_frames = defaultdict(set)
        for frame_index, player in item["rows"]:
            ref = (frame_index, int(player["local_track_id"]))
            full = predictions.get(ref, {})
            full_number = str(full.get("number", ""))
            full_confidence = float(full.get("confidence", 0.0))
            left = digit_predictions.get((*ref, "left"), {})
            right = digit_predictions.get((*ref, "right"), {})
            left_number = str(left.get("number", ""))
            right_number = str(right.get("number", ""))
            left_confidence = float(left.get("confidence", 0.0))
            right_confidence = float(right.get("confidence", 0.0))

            if len(full_number) == 2 and full_confidence >= 0.85:
                full_support[full_number].add(frame_index)
            if len(full_number) == 2 and full_confidence >= 0.55:
                last_votes.append((frame_index, full_number[-1], full_confidence))
                full_last_frames[full_number[-1]].add(frame_index)
            if len(left_number) == 2 and left_confidence >= 0.85:
                left_support[left_number].add(frame_index)
            if len(left_number) == 1 and left_confidence >= 0.55:
                first_votes.append((frame_index, left_number, left_confidence))
            if len(right_number) == 1 and right_confidence >= 0.55:
                last_votes.append((frame_index, right_number, right_confidence))
            if (
                len(left_number) == len(right_number) == 1
                and left_confidence >= 0.65
                and right_confidence >= 0.65
            ):
                paired_frames[left_number + right_number].add(frame_index)

        first = _select_best_digit(first_votes, 0.68)
        last = _select_best_digit(last_votes, 0.60)
        if first is None or last is None:
            continue
        candidate = first[0] + last[0]
        evidence = (
            full_support[candidate]
            if len(full_support[candidate]) >= 3
            else left_support[candidate]
            if len(left_support[candidate]) >= 3
            else paired_frames[candidate]
        )
        direct_evidence = len(evidence) >= 3 and max(evidence) - min(evidence) >= 3
        repeated_digits = (
            len(first[1]) >= 8
            and first[2] >= 0.80
            and len(last[1]) >= 8
            and last[2] >= 0.60
            and len(full_last_frames[candidate[-1]]) >= 8
        )
        if not direct_evidence and not repeated_digits:
            continue
        trusted_numbers[id(item)] = candidate
        item["number"] = candidate
        item["conflict"] = False
        for frame_index, player in item["rows"]:
            trusted_refs.add((frame_index, int(player["local_track_id"])))

    global_numbers = defaultdict(set)
    for item in resolved_segments:
        if item["number"] is not None and item["global_id"] is not None and not item["conflict"]:
            global_numbers[item["global_id"]].add(item["number"])
    stable_global_numbers = {
        global_id: next(iter(numbers))
        for global_id, numbers in global_numbers.items()
        if len(numbers) == 1
    }

    added = corrected = 0
    for item in resolved_segments:
        number = item["number"]
        global_id = item["global_id"]
        if number is None and not item["conflict"] and global_id in stable_global_numbers:
            number = stable_global_numbers[global_id]
        if number is not None:
            for frame_index, player in item["rows"]:
                existing = player.get("jersey_number")
                ref = (frame_index, int(player["local_track_id"]))
                if (
                    existing in (None, number)
                    or ref in trusted_refs
                    or not item["conflict"]
                ):
                    added += existing is None
                    corrected += existing is not None and existing != number
                    player["jersey_number"] = number
                    player["jersey_number_locked"] = True

    for item in resolved_segments:
        number = item.get("direct_number")
        if number is None or item["number"] != number:
            continue
        for _, player in item["rows"]:
            existing = player.get("jersey_number")
            if (
                existing is None
                or not _is_number_locked(player)
                or number.startswith(str(existing))
                or item["global_id"] is None
            ):
                added += existing is None
                corrected += existing is not None and existing != number
                player["jersey_number"] = number
                player["jersey_number_locked"] = True

    global_added, global_corrected = _propagate_global_number_consensus(records)
    added += global_added
    corrected += global_corrected

    while True:
        round_added = 0
        for item in resolved_segments:
            online_votes = [
                (frame_index, str(player["jersey_number"]), 1.0)
                for frame_index, player in item["rows"]
                if player.get("jersey_number") is not None and _is_number_locked(player)
            ]
            online_number = _compute_number_consensus(online_votes, 3)
            ocr_number = _compute_number_consensus(
                item["ocr_votes"], 3, purity_limit=0.65, confidence_limit=0.90, min_span=12
            )
            trusted_number = trusted_numbers.get(id(item))
            conflict = (
                trusted_number is None
                and online_number is not None
                and ocr_number is not None
                and ocr_number != online_number
            )
            item["conflict"] = conflict
            number = trusted_number or (
                None
                if conflict
                else item.get("direct_number")
                or online_number
                or ocr_number
            )
            item["number"] = number
            if number is not None:
                for _, player in item["rows"]:
                    if player.get("jersey_number") is None or not _is_number_locked(player):
                        existing = player.get("jersey_number")
                        corrected += existing is not None and existing != number
                        player["jersey_number"] = number
                        player["jersey_number_locked"] = True
                        added += existing is None
                        round_added += existing is None

        global_added, global_corrected = _propagate_global_number_consensus(records)
        added += global_added
        corrected += global_corrected
        round_added += global_added
        if not round_added:
            break

    local_numbers = {}
    for item in resolved_segments:
        players_by_frame = dict(item["rows"])
        votes = item["ocr_votes"]
        vote_start = vote_end = 0
        for frame_index in players_by_frame:
            while vote_end < len(votes) and votes[vote_end][0] <= frame_index:
                vote_end += 1
            while vote_start < vote_end and votes[vote_start][0] < frame_index - 6:
                vote_start += 1
            window = votes[vote_start:vote_end]
            number = _compute_number_consensus(
                window, 3, purity_limit=0.65, confidence_limit=0.90, min_span=2
            )
            if number is None:
                continue
            support = [vote[0] for vote in window if vote[1] == number]
            for row_frame in range(min(support), max(support) + 1):
                if row_frame in players_by_frame:
                    local_numbers[(row_frame, int(players_by_frame[row_frame]["local_track_id"]))] = number

    previous_local = {}
    for frame_index, record in enumerate(records, 1):
        for player in record.get("players", ()):
            local_id = player.get("local_track_id")
            number = local_numbers.get((frame_index, local_id))
            previous = previous_local.get(local_id)
            follows_same_number = previous and previous[1].get("jersey_number") == number
            follows_identity_reset = previous and previous[1].get("id") is not None
            if (
                number is not None
                and player.get("id") is None
                and not _is_number_locked(player)
                and previous is not None
                and frame_index - previous[0] <= 2
                and (follows_same_number or follows_identity_reset)
            ):
                existing = player.get("jersey_number")
                corrected += existing is not None and existing != number
                player["jersey_number"] = number
                player["jersey_number_locked"] = True
                added += existing is None
            if local_id is not None:
                previous_local[local_id] = (frame_index, player)

    global_numbers = defaultdict(Counter)
    for record in records:
        for player in record.get("players", ()):
            if (
                player.get("id") is not None
                and player.get("jersey_number") is not None
                and _is_number_locked(player)
            ):
                global_numbers[player["id"]][str(player["jersey_number"])] += 1
    demoted = 0
    for global_id, numbers in global_numbers.items():
        if len(numbers) < 2:
            continue
        number, count = numbers.most_common(1)[0]
        keep_number = number if count / sum(numbers.values()) >= 0.75 else None
        for record in records:
            for player in record.get("players", ()):
                if player.get("id") != global_id:
                    continue
                existing = player.get("jersey_number")
                if keep_number is not None and not _is_number_locked(player):
                    corrected += existing is not None and existing != keep_number
                    player["jersey_number"] = keep_number
                    player["jersey_number_locked"] = True
                elif (
                    _is_number_locked(player)
                    and (keep_number is None or str(existing) != keep_number)
                ):
                    player["jersey_number_locked"] = False
                    demoted += 1

    assign_identity_labels(records)
    identity_labels_changed = any(
        initial_states.get(
            (int(record.get("frame", index + 1)), player.get("local_track_id"), player_index),
            (False, None, "", (None, None)),
        )[1] != player.get("identity_label")
        for index, record in enumerate(records)
        for player_index, player in enumerate(record.get("players", ()))
    )
    status_changed = identity_labels_changed or any(
        initial_states.get(
            (int(record.get("frame", index + 1)), player.get("local_track_id"), player_index),
            (False, None, "", (None, None)),
        )[0] != _is_number_locked(player)
        for index, record in enumerate(records)
        for player_index, player in enumerate(record.get("players", ()))
    )

    after = sum(
        player.get("jersey_number") is not None
        for record in records for player in record.get("players", ())
    )

    return JerseyConsensusResult(
        added=added,
        corrected=corrected,
        demoted=demoted,
        status_changed=status_changed,
        identity_labels_changed=identity_labels_changed,
        labeled=after,
        total_players=sum(len(record.get("players", ())) for record in records),
    )


__all__ = ["JerseyConsensusResult", "reconcile_jersey_number_records"]
