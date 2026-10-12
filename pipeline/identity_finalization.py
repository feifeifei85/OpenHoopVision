"""Persist post-video identity reconciliation and refresh changed overlays."""

import json
import os
from pathlib import Path
from typing import Any, Callable

from reid.consensus import reconcile_jersey_number_records
from reid.records import assign_identity_labels, reconcile_identity_records


def _save_records(tracks_path: Path, records: list[dict[str, Any]], resolver: Any) -> None:
    tracks_tmp = tracks_path.with_name(tracks_path.name + ".tmp")
    try:
        with open(tracks_path, encoding="utf-8") as source, open(
            tracks_tmp, "w", encoding="utf-8"
        ) as destination:
            record_count = 0
            for index, line in enumerate(source):
                if index >= len(records):
                    raise ValueError("track records changed while resolving jersey numbers")
                record = json.loads(line)
                record["players"] = records[index].get("players", [])
                if resolver.identity_aliases:
                    reconcile_identity_records([record], resolver)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1
            if record_count != len(records):
                raise ValueError("track records changed while resolving jersey numbers")
        os.replace(tracks_tmp, tracks_path)
    except BaseException:
        tracks_tmp.unlink(missing_ok=True)
        raise


def finalize_identity_records(
    tracks_path: Path,
    sampler: Any,
    resolver: Any,
    rewrite_overlay: Callable[..., None],
    label_for_player: Callable[[dict[str, Any]], str],
) -> None:
    """Reconcile cached jersey evidence, persist JSONL changes, and redraw when needed."""
    has_identity_aliases = bool(resolver.identity_aliases)
    records: list[dict[str, Any]] = []
    initial_states: dict[tuple[int, Any, int], tuple[bool, Any, str, tuple[Any, Any]]] = {}
    history_label_changed = False
    reconciled_observations = 0

    with open(tracks_path, encoding="utf-8") as track_file:
        for record_index, line in enumerate(track_file):
            source_record = json.loads(line)
            record = {
                "frame": source_record.get("frame", record_index + 1),
                "players": source_record.get("players", []),
            }
            frame_index = int(record["frame"])
            for player_index, player in enumerate(record["players"]):
                player.setdefault("jersey_number_candidate", player.get("jersey_number"))
                player.setdefault("jersey_number_locked", player.get("jersey_number") is not None)
                key = (frame_index, player.get("local_track_id"), player_index)
                initial_states[key] = (
                    bool(player["jersey_number_locked"]),
                    player.get("identity_label"),
                    label_for_player(player),
                    (player.get("team"), player.get("team_semantic")),
                )

            if has_identity_aliases:
                reconciled_observations += reconcile_identity_records([record], resolver)
                assign_identity_labels([record])
                for player_index, player in enumerate(record.get("players", ())):
                    key = (frame_index, player.get("local_track_id"), player_index)
                    initial = initial_states[key]
                    history_label_changed |= (
                        initial[2] != label_for_player(player)
                        or initial[3] != (player.get("team"), player.get("team_semantic"))
                    )
            records.append(record)

    if has_identity_aliases:
        print(
            f"ReID history reconciliation: {reconciled_observations} player observations across "
            f"{resolver.identity_relink_count} relinks"
        )

    result = reconcile_jersey_number_records(
        records,
        sampler.predictions,
        sampler.digit_predictions,
        initial_states,
    )
    changed = bool(result.added or result.corrected or result.demoted or result.status_changed)
    if not changed and not has_identity_aliases:
        print(f"jersey track consensus: {result.labeled}/{result.total_players} labeled")
        return

    _save_records(tracks_path, records, resolver)
    if changed or history_label_changed:
        previous_overlay_labels = {key: state[2] for key, state in initial_states.items()}
        rewrite_overlay(records, previous_overlay_labels)
    print(
        f"jersey track consensus: {result.labeled}/{result.total_players} labeled; "
        f"added={result.added}, corrected={result.corrected}, ambiguous labels demoted={result.demoted}"
    )


__all__ = ["finalize_identity_records"]
