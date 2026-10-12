"""Translate event IDs and reconcile identity references in persisted records."""

from typing import Any

from tracking.team import normalize_team

_IDENTITY_REFERENCE_FIELDS = frozenset({
    "player_id", "holder_id", "global_player_id", "active_player_id", "shooter_id",
    "gained_by", "turnover_player_id", "receiver_id",
})


def translate_event_identity_references(
    value: Any,
    resolver: Any,
    parent_key: str | None = None,
    parent_frame: int | None = None,
) -> Any:
    """Copy event output while replacing McByte references with known global IDs."""
    if isinstance(value, dict):
        reference_frame = parent_frame
        for frame_key in ("start_frame", "frame"):
            try:
                if value.get(frame_key) is not None:
                    reference_frame = int(value[frame_key])
                    break
            except (TypeError, ValueError):
                continue
        local_id = value.get("local_track_id")
        local_frame = reference_frame
        for frame_key in ("completed_frame", "end_frame"):
            try:
                if value.get(frame_key) is not None:
                    local_frame = int(value[frame_key])
                    break
            except (TypeError, ValueError):
                continue
        translated = {}
        for key, item in value.items():
            if key in _IDENTITY_REFERENCE_FIELDS or (key == "id" and parent_key == "scorer_tag"):
                if item is None:
                    translated[key] = None
                else:
                    try:
                        is_local_reference = (
                            local_id is not None
                            and str(item) == str(local_id)
                            and (key == "player_id" or (key == "id" and parent_key == "scorer_tag"))
                        )
                        translated[key] = resolver.get_global_id_at(
                            int(local_id if is_local_reference else item),
                            local_frame if is_local_reference else reference_frame,
                        )
                    except (TypeError, ValueError):
                        translated[key] = None
            else:
                translated[key] = translate_event_identity_references(
                    item, resolver, key, local_frame if local_id is not None else reference_frame
                )
        return translated
    if isinstance(value, list):
        return [translate_event_identity_references(item, resolver, parent_key, parent_frame) for item in value]
    if isinstance(value, tuple):
        return tuple(translate_event_identity_references(item, resolver, parent_key, parent_frame) for item in value)
    return value


def canonicalize_global_identity_references(
    value: Any, resolver: Any, parent_key: str | None = None
) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                resolver.canonicalize_global_id(item)
                if key in _IDENTITY_REFERENCE_FIELDS or (key == "id" and parent_key == "scorer_tag")
                else canonicalize_global_identity_references(item, resolver, key)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_global_identity_references(item, resolver, parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize_global_identity_references(item, resolver, parent_key) for item in value)
    return value


def reconcile_identity_records(records: list[dict[str, Any]], resolver: Any) -> int:
    """Apply accepted ReID merges to earlier player rows and event references."""
    changed_observations = 0
    for record in records:
        frame_index = int(record.get("frame", 0))
        for player in record.get("players", ()):
            local_id = player.get("local_track_id")
            if local_id is None:
                continue
            if player.get("reid_eligible") is False:
                player["id"] = None
                continue
            global_id = resolver.get_global_id_at(int(local_id), frame_index)
            if player.get("id") != global_id:
                player["id"] = global_id
                changed_observations += 1
            identity = resolver.identities.get(global_id)
            team_id = None if identity is None else resolver.get_identity_team(global_id)
            if team_id is not None:
                player["team"] = int(team_id)
                player["team_semantic"] = normalize_team(team_id)
        canonical = canonicalize_global_identity_references(record, resolver)
        record.clear()
        record.update(canonical)
    return changed_observations


def assign_identity_labels(records: list[dict[str, Any]]) -> None:
    """Set stable G/R display labels from global and local player IDs in place."""
    for record in records:
        for player in record.get("players", ()):
            global_id = player.get("id")
            local_id = player.get("local_track_id")
            if player.get("reid_eligible") is False:
                player["id"] = None
                player["jersey_number"] = None
                player["jersey_number_candidate"] = None
                player["jersey_number_locked"] = False
                player["identity_label"] = None if local_id is None else f"R{local_id}"
                continue
            prefix = str(player.get("identity_label", ""))[:1]
            player["identity_label"] = (
                f"{prefix if prefix in {'R', 'G'} else 'R'}{global_id}"
                if global_id is not None
                else None if local_id is None else f"R{local_id}"
            )


__all__ = ["translate_event_identity_references", "reconcile_identity_records", "assign_identity_labels"]
