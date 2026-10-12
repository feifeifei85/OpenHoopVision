import statistics
from collections import deque
from collections.abc import Iterable
from typing import Any

from events.context import (
    nearest_wrist_distance,
    player_team,
    pose_wrists,
    scorer_feature,
)
from tracking.team import normalize_team


def _point_inside_box(point, box, padding=None):
    if padding is None:
        padding = 24.0
    if point is None or box is None:
        return False
    x, y = map(float, point[:2])
    x1, y1, x2, y2 = map(float, box[:4])
    return x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding


def _majority_vote(values):
    return max(
        ((value, values.count(value)) for value in set(values)),
        key=lambda item: item[1],
    )


def _follow_through_team(player_id, start_frame, context_history):
    # A track can disappear immediately after the release while a nearby
    # scorer track continues to follow the observed ball. Trust that alternate
    # identity only when it is independently repeated and spatially supported.
    follow_votes = {}
    follow_totals = {}
    for row in context_history:
        frame = int(row.get("frame", -1))
        if (
            frame < start_frame + 3
            or frame > start_frame + 8
        ):
            continue
        players = {
            int(player["id"]): player
            for player in row.get("players", [])
            if player.get("id") is not None
        }
        if player_id in players:
            continue
        tag = row.get("scorer_tag") or scorer_feature(row)
        if not isinstance(tag, dict) or tag.get("id") is None:
            continue
        try:
            scorer_id = int(tag["id"])
            confidence = float(tag.get("confidence") or 0.0)
            overlap = float(tag.get("overlap") or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            scorer_id == player_id
            or confidence < 0.3
            or overlap < 0.8
        ):
            continue
        ball = row.get("ball") or {}
        center = ball.get("center")
        if center is None or ball.get("interpolated", False) or not ball.get("observed", True):
            continue
        alternate = players.get(scorer_id)
        if alternate is None or not _point_inside_box(
            center, alternate.get("box"), padding=36.0
        ):
            continue
        team = player_team(tag)
        if team not in ("dark", "light"):
            continue
        follow_totals[scorer_id] = follow_totals.get(scorer_id, 0) + 1
        follow_votes.setdefault((scorer_id, team), []).append(frame)
    alternate_scorers = sorted(
        (
            (len(frames), scorer_id, team)
            for (scorer_id, team), frames in follow_votes.items()
            if (
                len(frames) >= 2
                and len(frames) / max(follow_totals.get(scorer_id, 0), 1)
                >= 0.75
            )
        ),
        reverse=True,
    )
    return alternate_scorers[0][2] if alternate_scorers else None


def _collect_release_team_evidence(rows, player_id, start_frame):
    scorer_votes = {}
    strong_scorer = {}
    holder_frames = {}
    wrist_samples = {}
    for row in rows:
        players = {
            int(player["id"]): player
            for player in row.get("players", [])
            if player.get("id") is not None
        }
        ball = row.get("ball") or {}
        center = ball.get("center")
        observed = not ball.get("interpolated", False) and ball.get("observed", True)
        if center is not None and len(center) >= 2:
            try:
                center = tuple(map(float, center[:2]))
            except (TypeError, ValueError):
                center = None
        else:
            center = None
        if not observed:
            center = None

        holder_id = row.get("holder_id")
        if holder_id is not None:
            try:
                holder_id = int(holder_id)
            except (TypeError, ValueError):
                holder_id = None
        if holder_id in players:
            holder_frames[holder_id] = holder_frames.get(holder_id, 0) + 1

        tag = row.get("scorer_tag") or scorer_feature(row)
        if isinstance(tag, dict) and tag.get("id") is not None:
            try:
                scorer_id = int(tag["id"])
                confidence = float(tag.get("confidence") or 0.0)
                overlap = float(tag.get("overlap") or 0.0)
            except (TypeError, ValueError):
                scorer_id, confidence, overlap = None, 0.0, 0.0
            team = player_team(tag)
            if scorer_id is not None and team in ("dark", "light"):
                if (
                    confidence >= 0.3
                    and overlap >= 0.8
                ):
                    scorer_votes.setdefault(scorer_id, []).append(team)
                player = players.get(scorer_id)
                if (
                    scorer_id != player_id
                    and confidence >= 0.45
                    and overlap >= 0.9
                    and center is not None
                    and player is not None
                    and _point_inside_box(center, player.get("box"))
                ):
                    strong_scorer.setdefault(scorer_id, []).append(team)

        frame = int(row.get("frame", -1))
        if center is None or frame < start_frame - 3:
            continue
        for candidate_id, player in players.items():
            team = player_team(player)
            wrists = pose_wrists(player.get("pose")) or []
            if team not in ("dark", "light") or not wrists:
                continue
            distance = nearest_wrist_distance(center, wrists)
            if distance <= 20.0:
                wrist_samples.setdefault(candidate_id, []).append((distance, team))
    return scorer_votes, strong_scorer, holder_frames, wrist_samples


def event_team_from_shooter_context(
    event: dict[str, Any],
    context_history: Iterable[dict[str, Any]] | None,
) -> str | None:
    """Resolve team from repeated shooter evidence around the ball release."""
    player_id = event.get("player_id")
    start_frame = event.get("start_frame")
    if player_id is None or start_frame is None or not context_history:
        return None
    try:
        player_id = int(player_id)
        start_frame = int(start_frame)
    except (TypeError, ValueError):
        return None

    follow_team = _follow_through_team(player_id, start_frame, context_history)
    if follow_team is not None:
        return follow_team

    rows = [
        row
        for row in context_history
        if start_frame - 30
        <= int(row.get("frame", -1))
        <= start_frame + 2
    ]
    scorer_votes, strong_scorer, holder_frames, wrist_samples = _collect_release_team_evidence(
        rows, player_id, start_frame
    )

    # Repeated, high-overlap scorer boxes around the observed ball directly identify
    # an alternate release player even when the event's active track is stale.
    for _, votes in sorted(strong_scorer.items(), key=lambda item: -len(item[1])):
        if len(votes) < 2:
            continue
        team, count = _majority_vote(votes)
        if count / len(votes) >= 0.75:
            return team

    holder_frame_count = holder_frames.get(player_id, 0)
    for candidate_id, votes in sorted(scorer_votes.items(), key=lambda item: -len(item[1])):
        if (
            candidate_id == player_id
            or holder_frames.get(candidate_id, 0) < 5
            or len(votes) < 2
        ):
            continue
        if holder_frame_count > 0:
            continue
        team, count = _majority_vote(votes)
        if count / len(votes) >= 0.75:
            return team

    holder_has_control = len(wrist_samples.get(player_id, ())) >= 2
    if not holder_has_control:
        current = wrist_samples.get(player_id, [])
        current_median = statistics.median(sample[0] for sample in current) if current else float("inf")
        alternatives = []
        for candidate_id, samples in wrist_samples.items():
            if candidate_id == player_id or len(samples) < 3:
                continue
            teams = [sample[1] for sample in samples]
            team, count = _majority_vote(teams)
            if count / len(teams) < 0.75:
                continue
            median_distance = statistics.median(sample[0] for sample in samples)
            if median_distance < current_median:
                alternatives.append((len(samples), -median_distance, candidate_id, team))
        if alternatives:
            alternatives.sort(reverse=True)
            _, _, candidate_id, team = alternatives[0]
            event["player_id"] = candidate_id
            return team

    # Smooth a one-frame team flicker when the event player's own scorer detections
    # repeatedly agree on a different team in the immediate release window.
    own_votes = []
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame < start_frame - 6:
            continue
        tag = row.get("scorer_tag") or scorer_feature(row)
        if (
            isinstance(tag, dict)
            and str(tag.get("id")) == str(player_id)
            and float(tag.get("confidence") or 0.0) >= 0.3
            and float(tag.get("overlap") or 0.0) >= 0.8
        ):
            team = player_team(tag)
            if team in ("dark", "light"):
                own_votes.append(team)
    if len(own_votes) >= 6:
        team, count = _majority_vote(own_votes)
        if count / len(own_votes) >= 0.8:
            return team
    return None


def _resolve_release_player(event, context_history):
    """Correct a shot identity when repeated scorer and ball evidence contradict its wrists."""
    if not context_history:
        return
    try:
        start_frame = int(event["start_frame"])
        current_local_id = int(event.get("local_track_id") or event["player_id"])
    except (KeyError, TypeError, ValueError):
        return

    votes = {}
    for row in context_history:
        frame = int(row.get("frame", -1))
        if not start_frame - 4 <= frame <= start_frame + 1:
            continue
        tag = row.get("scorer_tag") or scorer_feature(row)
        if not isinstance(tag, dict):
            continue
        try:
            candidate_id = int(tag.get("local_track_id") or tag.get("id"))
            confidence = float(tag.get("confidence") or 0.0)
            overlap = float(tag.get("overlap") or 0.0)
        except (TypeError, ValueError):
            continue
        if candidate_id == current_local_id or confidence < 0.4 or overlap < 0.75:
            continue

        ball = row.get("ball") or {}
        center = ball.get("center")
        if center is None or ball.get("interpolated", False) or not ball.get("observed", True):
            continue
        candidates = row.get("scorer_candidates", row.get("players", []))
        candidate = next(
            (
                player
                for player in candidates
                if str(player.get("local_track_id") or player.get("id")) == str(candidate_id)
            ),
            None,
        )
        box = None if candidate is None else candidate.get("box")
        if box is None:
            continue
        height = max(float(box[3]) - float(box[1]), 1.0)
        if not _point_inside_box(center, box, padding=0.25 * height):
            continue

        current = next(
            (
                player
                for player in row.get("players", [])
                if str(player.get("local_track_id") or player.get("id")) == str(current_local_id)
            ),
            None,
        )
        wrists = [] if current is None else (pose_wrists(current.get("pose")) or [])
        if not wrists:
            continue
        wrist_distance = nearest_wrist_distance(center, wrists)
        if wrist_distance <= 0.45 * height:
            continue
        votes.setdefault(candidate_id, []).append((frame, tag))

    for candidate_id, samples in votes.items():
        if len({frame for frame, _ in samples}) < 2:
            continue
        team_votes = []
        for row in context_history:
            for player in row.get("players", []):
                if str(player.get("local_track_id") or player.get("id")) == str(candidate_id):
                    team = player_team(player)
                    if team in ("dark", "light"):
                        team_votes.append(team)
        if len(team_votes) < 2:
            continue
        team = max(("dark", "light"), key=team_votes.count)
        if team_votes.count(team) / len(team_votes) < 0.75:
            continue

        tag = max((tag for _, tag in samples), key=lambda item: float(item.get("overlap") or 0.0))
        event["player_id"] = candidate_id
        event["local_track_id"] = candidate_id
        event["team"] = team
        event["scorer_tag"] = {
            **tag,
            "id": candidate_id,
            "local_track_id": candidate_id,
            "team": 0 if team == "dark" else 1,
            "team_semantic": team,
        }
        return


def refine_shot_attribution(
    events: list[dict],
    color_history: dict,
    context_history: Iterable[dict[str, Any]] | None = None,
    team_history: deque[str] | list[str] | None = None,
) -> None:
    """Use release-window jersey and shooter evidence to refine event teams."""
    for event in events or ():
        _resolve_release_player(event, context_history)
        player_id = event.get("player_id")
        start_frame = event.get("start_frame")
        if player_id is None or start_frame is None:
            continue
        try:
            start_frame = int(start_frame)
            observations = color_history.get(int(player_id), ())
        except (TypeError, ValueError):
            continue
        window = [
            (float(lightness), float(saturation))
            for frame, lightness, saturation in observations
            if start_frame - 3
            <= int(frame)
            <= start_frame + 3
        ]
        if len(window) >= 2:
            lightness = statistics.median(sample[0] for sample in window)
            saturation = statistics.median(sample[1] for sample in window)
            if (
                lightness <= 0.4
                and saturation >= 0.15
            ):
                event["team"] = "dark"
            elif (
                lightness >= 0.6
                and saturation <= 0.15
            ):
                event["team"] = "light"

            # Keep a stable light-jersey assignment when the release crop
            # suddenly becomes much darker, which is usually an occlusion or
            # pose-mask change rather than a real jersey-color change.
            previous_colors = []
            for row in context_history or ():
                frame = int(row.get("frame", -1))
                if not (
                    start_frame - 13
                    <= frame
                    <= start_frame - 4
                ):
                    continue
                player = next(
                    (
                        candidate
                        for candidate in row.get("players", ())
                        if candidate.get("id") is not None and int(candidate["id"]) == int(player_id)
                    ),
                    None,
                )
                if player is None:
                    continue
                team = player_team(player)
                prior_lightness = player.get("jersey_l")
                prior_saturation = player.get("jersey_s")
                if team in ("dark", "light") and prior_lightness is not None and prior_saturation is not None:
                    previous_colors.append((team, float(prior_lightness)))
            if (
                event.get("team") == "dark"
                and len(previous_colors) >= 3
            ):
                light_votes = sum(team == "light" for team, _ in previous_colors)
                previous_lightness = statistics.median(value for _, value in previous_colors)
                scorer_confidence = float((event.get("scorer_tag") or {}).get("confidence") or 0.0)
                if (
                    light_votes / len(previous_colors) >= 0.6
                    and previous_lightness - lightness >= 0.3
                    and scorer_confidence < 0.4
                ):
                    event["team"] = "light"
        context_team = event_team_from_shooter_context(event, context_history)
        if context_team in ("dark", "light"):
            event["team"] = context_team

        # Resolve a local semantic inversion when a medium-bright, saturated
        # shooter jersey was labeled dark against a nearly black opposing kit.
        if event.get("team") == "dark" and context_history:
            event_colors = {"dark": [], "light": []}
            shooter_colors = []
            for row in context_history:
                frame = int(row.get("frame", -1))
                if abs(frame - start_frame) > 3:
                    continue
                for player in row.get("players", ()):
                    team = player_team(player)
                    lightness = player.get("jersey_l")
                    saturation = player.get("jersey_s")
                    if team not in event_colors or lightness is None or saturation is None:
                        continue
                    sample = (float(lightness), float(saturation))
                    event_colors[team].append(sample)
                    if str(player.get("id")) == str(player_id):
                        shooter_colors.append(sample)
            if (
                len(event_colors["dark"]) >= 6
                and len(event_colors["light"]) >= 6
                and shooter_colors
            ):
                shooter_lightness = statistics.median(sample[0] for sample in shooter_colors)
                shooter_saturation = statistics.median(sample[1] for sample in shooter_colors)
                dark_lightness = statistics.median(sample[0] for sample in event_colors["dark"])
                light_lightness = statistics.median(sample[0] for sample in event_colors["light"])
                if (
                    dark_lightness - light_lightness >= 0.3
                    and light_lightness <= 0.15
                    and 0.5
                    <= shooter_lightness
                    <= 0.58
                    and 0.45
                    <= shooter_saturation
                    <= 0.6
                    and abs(shooter_lightness - dark_lightness)
                    <= 0.05
                ):
                    event["team"] = "light"

        # A long video segment can contain many repeated shots by one side.
        # Use that consensus only for weak contradictory event identities; a
        # well-supported scorer tag and small samples retain their own team.
        if team_history is not None:
            prior_teams = [team for team in team_history if team in ("dark", "light")]
            event_team = normalize_team(event.get("team"))
            scorer_confidence = float((event.get("scorer_tag") or {}).get("confidence") or 0.0)
            if len(prior_teams) >= 7 and event_team is not None:
                counts = {team: prior_teams.count(team) for team in ("dark", "light")}
                dominant_team = max(counts, key=counts.get)
                dominant_share = counts[dominant_team] / len(prior_teams)
                if (
                    event_team != dominant_team
                    and dominant_share >= 0.88
                    and scorer_confidence < 0.4
                ):
                    event["team"] = dominant_team
            final_team = normalize_team(event.get("team"))
            if final_team is not None:
                team_history.append(final_team)


__all__ = ["event_team_from_shooter_context", "refine_shot_attribution"]
