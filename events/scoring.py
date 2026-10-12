"""Court-location shot values and scoreboard event updates."""

from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any

import numpy as np

from events.context import event_team, player_team
from tracking.court import COURT_GEOMETRY, COURT_LENGTH_M, COURT_WIDTH_M


def shot_point_value(court_xy: Sequence[float] | np.ndarray | None) -> int | None:
    """Return a court-based two/three-point value, or None near the boundary line."""
    if court_xy is None:
        return None
    try:
        x, y = np.asarray(court_xy, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.isfinite((x, y)).all() or not 0 <= x <= COURT_LENGTH_M or not 0 <= y <= COURT_WIDTH_M:
        return None

    geometry = COURT_GEOMETRY
    basket_x = geometry["basket_offset_m"] if x < COURT_LENGTH_M * 0.5 else COURT_LENGTH_M - geometry["basket_offset_m"]
    distance = float(np.hypot(x - basket_x, y - COURT_WIDTH_M * 0.5))
    corner_y = geometry["baseline_offset_m"]
    corner_x = geometry["basket_offset_m"] + np.sqrt(
        geometry["three_point_radius_m"] ** 2 - (COURT_WIDTH_M * 0.5 - corner_y) ** 2
    )
    corner_distance = min(y, COURT_WIDTH_M - y)
    baseline_distance = min(x, COURT_LENGTH_M - x)
    if corner_distance < corner_y - 0.2 and baseline_distance < corner_x - 0.25:
        return 3
    if abs(corner_distance - corner_y) <= 0.2 and baseline_distance < corner_x + 0.25:
        return None
    if abs(distance - geometry["three_point_radius_m"]) <= 0.35:
        return None
    return 3 if distance > geometry["three_point_radius_m"] else 2


def apply_shot_events_to_scoreboard(
    scores: MutableMapping[str, int],
    events: Iterable[dict[str, Any]] | None,
    *,
    team_observations: Iterable[dict[str, Any]] = (),
    pending_events: list[dict[str, Any]] | None = None,
) -> None:
    """Apply made shots once and retain events whose team is not known yet."""
    queued = list(pending_events or ())
    if pending_events is not None:
        pending_events.clear()
    candidates = []
    seen = set()
    for event in queued + list(events or ()):
        if id(event) not in seen:
            candidates.append(event)
            seen.add(id(event))
    for event in candidates:
        if event.get("result") != "made" or event.get("score_applied"):
            continue
        team = event_team(event)
        if team is None and event.get("player_id") is not None:
            player = next(
                (item for item in team_observations if str(item.get("id")) == str(event["player_id"])),
                None,
            )
            team = None if player is None else player_team(player)
        points = event.get("points", 2)
        try:
            points = int(points)
        except (TypeError, ValueError):
            points = 2
        if points <= 0:
            points = 2
        event["points"] = points
        if team not in scores:
            event["score_pending"] = "team_unknown"
            if pending_events is not None:
                pending_events.append(event)
            continue
        scores[team] += points
        event["team"] = team
        event["score_applied"] = True
        event.pop("score_pending", None)


__all__ = ["shot_point_value", "apply_shot_events_to_scoreboard"]
