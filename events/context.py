import math
from collections.abc import Mapping, Sequence
from typing import Any

from tracking.team import normalize_team


def scorer_feature(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Choose the player box with the strongest overlap with a scorer detection."""
    best = None
    players = row.get("scorer_candidates", row.get("players", []))
    for detection in row.get("detections", {}).get("scorer", []):
        scorer_box = detection.get("box")
        if not scorer_box or len(scorer_box) < 4:
            continue
        x1, y1, x2, y2 = map(float, scorer_box[:4])
        scorer_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        for player in players:
            if player.get("id") is None and player.get("local_track_id") is None:
                continue
            player_box = player.get("box")
            if not player_box or len(player_box) < 4:
                continue
            px1, py1, px2, py2 = map(float, player_box[:4])
            intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(0.0, min(y2, py2) - max(y1, py1))
            player_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            overlap = intersection / max(scorer_area + player_area - intersection, 1.0)
            candidate = (overlap, float(detection.get("score") or 0.0), player)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < 0.3:
        return None
    player = best[2]
    return {
        "id": None if player.get("id") is None else int(player["id"]),
        "local_track_id": (
            None if player.get("local_track_id") is None else int(player["local_track_id"])
        ),
        "confidence": round(best[1], 5),
        "team": player.get("team"),
        "team_semantic": player.get("team_semantic"),
        "overlap": round(best[0], 5),
    }


def pose_wrists(pose: Sequence[Sequence[float]] | None) -> list[tuple[float, float]] | None:
    """Return finite, sufficiently confident wrist coordinates from a pose row."""
    threshold = 0.25
    if pose is None:
        return None
    wrists = []
    for index in [9, 10]:
        if len(pose) > index and pose[index] is not None and len(pose[index]) >= 3:
            x, y, conf = map(float, pose[index][:3])
            if conf >= threshold and math.isfinite(x) and math.isfinite(y):
                wrists.append((x, y))
    return wrists


def nearest_wrist_distance(point: Sequence[float], wrists: Sequence[tuple[float, float]]) -> float:
    """Measure the shortest image-space distance from a point to either wrist."""
    x, y = point[:2]
    return min(math.hypot(x - wrist_x, y - wrist_y) for wrist_x, wrist_y in wrists)


def player_team(player: Mapping[str, Any]) -> str | None:
    """Resolve the stable dark/light team label for a player feature."""
    return normalize_team(player.get("team_semantic")) or normalize_team(player.get("team"))


def event_team(event: Mapping[str, Any]) -> str | None:
    """Resolve an event team from its explicit team or scorer evidence."""
    team = normalize_team(event.get("team"))
    if team is not None:
        return team
    for tag in (event.get("peak_scorer_tag"), event.get("scorer_tag")):
        if not isinstance(tag, dict):
            continue
        team = player_team(tag)
        if team is not None:
            return team
    return None


def shooter_ball_context(
    players: Sequence[Mapping[str, Any]],
    holder_id: int | None,
    scorer_tag: Mapping[str, Any] | None,
    active_player_id: int | None = None,
    event_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the selected shooter's box for ball-action reclassification."""
    player_id = active_player_id
    if player_id is None:
        player_id = holder_id
    if (
        player_id is None
        and scorer_tag is not None
        and float(scorer_tag.get("confidence") or 0.0)
        >= (event_parameters or {}).get("late_entry_min_scorer_confidence", 0.3)
    ):
        player_id = scorer_tag.get("id")
    if player_id is None:
        return None
    player = next((item for item in players if str(item.get("id")) == str(player_id)), None)
    if player is None or not player.get("box"):
        return None
    return {"player_id": player_id, "box": player["box"]}


__all__ = [
    "event_team",
    "nearest_wrist_distance",
    "player_team",
    "pose_wrists",
    "scorer_feature",
    "shooter_ball_context",
]


