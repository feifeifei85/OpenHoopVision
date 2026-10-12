from collections import deque
import math

from tracking.ball import normalize_fps
from events.context import player_team, pose_wrists, normalize_team


class PossessionEventDetector:
    """Track possession from holder detections and ball-player proximity."""

    def __init__(self, fps):
        self.fps = normalize_fps(fps)
        self.last_holder = None
        self.last_team = None
        self.candidate_holder = None
        self.candidate_team = None
        self.candidate_teams = []
        self.candidate_start = None
        self.candidate_frames = 0
        self.candidate_ball_frames = 0
        self.unheld_since = None
        self.pending_miss = None
        self.last_holder_frame = None
        self.last_holder_box = None
        self.last_ball_position = None
        self.candidate_ball_position = None
        self.last_team_observations = deque(maxlen=5)
        self.last_frame = 0

    def update(self, row, events=()):
        frame = int(row.get("frame", self.last_frame + 1))
        candidate_gap_frames = max(1, round(0.15 * self.fps))
        self.last_frame = frame
        for event in events:
            if event.get("event") == "shot":
                if event.get("result") == "miss":
                    self.pending_miss = event
                else:
                    self.pending_miss = None
                    self.last_holder = None
                    self.last_team = None
                    self.last_holder_frame = None
                    self.last_holder_box = None
                    self.last_ball_position = None
                    self.last_team_observations.clear()
        if self.pending_miss is not None:
            missed_at = int(self.pending_miss.get("end_frame", frame))
            if frame - missed_at > round(3.0 * self.fps):
                self.pending_miss = None

        holder_id = row.get("holder_id")
        player = next(
            (
                item
                for item in row.get("players", [])
                if holder_id is not None and str(item.get("id")) == str(holder_id)
            ),
            None,
        )
        # A clear ball-to-wrist match can correct a missing or stale holder label.
        inferred_player = self._ball_holder(row)
        inferred_id = None if inferred_player is None else self._player_id(inferred_player)
        if inferred_id is not None and (player is None or str(inferred_id) != str(holder_id)):
            holder_id, player = inferred_id, inferred_player
        if holder_id is None or player is None:
            if self.unheld_since is None:
                self.unheld_since = frame
            if self.candidate_start is None or frame - self.candidate_start > candidate_gap_frames:
                self._reset_candidate()
            return []

        team = player_team(player)
        same_holder = self.last_holder is not None and str(holder_id) == str(self.last_holder)
        regained_after_miss = (
            same_holder
            and self.pending_miss is not None
            and self.unheld_since is not None
            and frame - self.unheld_since >= 2
        )
        if same_holder and not regained_after_miss:
            self.last_holder_box = self._player_box(player)
            if team is not None:
                self.last_team_observations.append(team)
                self.last_team = self._team_vote(self.last_team_observations, self.last_team)
            if self._ball_near_player(row, player):
                self.last_ball_position = tuple(map(float, row["ball"]["center"][:2]))
                self.unheld_since = None
            else:
                ball = row.get("ball") or {}
                if (
                    ball.get("observed", not ball.get("interpolated", False))
                    and ball.get("center") is not None
                    and self.unheld_since is None
                ):
                    self.unheld_since = frame
            self.last_holder_frame = frame
            self._reset_candidate()
            return []

        same_candidate = (
            str(holder_id) == str(self.candidate_holder)
            and self.candidate_start is not None
            and frame - self.candidate_start <= candidate_gap_frames
        )
        if not same_candidate:
            self.candidate_holder = holder_id
            self.candidate_teams = [] if team is None else [team]
            self.candidate_team = team
            self.candidate_start = frame
            self.candidate_frames = 1
            self.candidate_ball_frames = int(self._ball_near_player(row, player))
            self.candidate_ball_position = (
                tuple(map(float, row["ball"]["center"][:2])) if self.candidate_ball_frames else None
            )
        else:
            if team is not None:
                self.candidate_teams.append(team)
            self.candidate_team = self._team_vote(self.candidate_teams, self.candidate_team)
            self.candidate_frames += 1
            if self._ball_near_player(row, player):
                self.candidate_ball_frames += 1
                self.candidate_ball_position = tuple(map(float, row["ball"]["center"][:2]))
        if self.candidate_frames < max(3, round(0.1 * self.fps)):
            return []

        previous_holder = self.last_holder
        previous_team = normalize_team(self.last_team)
        previous_frame = self.last_holder_frame
        previous_box = self.last_holder_box
        previous_ball_position = self.last_ball_position
        self.last_holder = holder_id
        self.last_team = self.candidate_team
        self.last_holder_frame = frame
        self.last_holder_box = self._player_box(player)
        self.last_team_observations.clear()
        self.last_team_observations.extend(self.candidate_teams)
        events = []
        if self.candidate_ball_frames >= 2:
            event = self._rebound(frame, self.candidate_team)
            if event is None:
                event = self._pass(
                    frame,
                    previous_holder,
                    previous_team,
                    previous_frame,
                    self.candidate_team,
                    player,
                    row,
                )
            if event is None:
                events = self._turnover(
                    frame,
                    previous_holder,
                    previous_team,
                    previous_frame,
                    self.candidate_team,
                    previous_box,
                    previous_ball_position,
                    player,
                )
            else:
                events = [event]
        self.last_ball_position = self.candidate_ball_position
        self.unheld_since = None
        self._reset_candidate()
        return events

    def _rebound(self, frame, team):
        shot = self.pending_miss
        if shot is None or shot.get("result") != "miss":
            return None
        start = int(self.candidate_start)
        team = normalize_team(team)
        shot_team = normalize_team(shot.get("team"))
        rebound_type = (
            "unknown" if team is None or shot_team is None
            else "offensive" if team == shot_team
            else "defensive"
        )
        self.pending_miss = None
        return {
            "event": "rebound",
            "start_frame": start,
            "end_frame": int(frame),
            "completed_frame": int(frame),
            "start": round(max(0.0, (start - 1) / self.fps), 4),
            "end": round(max(0.0, (frame - 1) / self.fps), 4),
            "player_id": self.last_holder,
            "team": team,
            "rebound_type": rebound_type,
            "shot_id": shot.get("shot_id"),
        }

    def _turnover(
        self, frame, previous_holder, previous_team, previous_frame, team,
        previous_box, previous_ball_position, player,
    ):
        team = normalize_team(team)
        previous_team = normalize_team(previous_team)
        if (
            self.pending_miss is not None
            or previous_holder is None
            or previous_frame is None
            or frame - previous_frame > round(0.5 * self.fps)
            or team is None
            or previous_team is None
            or team == previous_team
        ):
            return []
        start = int(self.candidate_start)
        interval = {
            "start_frame": start,
            "end_frame": int(frame),
            "completed_frame": int(frame),
            "start": round(max(0.0, (start - 1) / self.fps), 4),
            "end": round(max(0.0, (frame - 1) / self.fps), 4),
        }
        is_steal = self._direct_steal_supported(
            previous_box, previous_ball_position, player, self.candidate_ball_position
        )
        events = [{
            "event": "turnover",
            **interval,
            "player_id": previous_holder,
            "team": previous_team,
            "gained_by": self.last_holder,
            "gained_team": team,
            "cause": "steal" if is_steal else "unknown",
            "evidence": "stable_opponent_control_with_visible_ball",
        }]
        if is_steal:
            events.append({
                "event": "steal",
                **interval,
                "player_id": self.last_holder,
                "team": team,
                "turnover_player_id": previous_holder,
                "evidence": "visible_ball_transfer_during_close_opponent_contest",
            })
        return events

    @classmethod
    def _direct_steal_supported(cls, previous_box, previous_ball, player, current_ball):
        current_box = cls._player_box(player)
        if (
            previous_box is None
            or current_box is None
            or previous_ball is None
            or current_ball is None
        ):
            return False
        _, oy1, _, oy2 = previous_box
        _, ny1, _, ny2 = current_box
        old_height, new_height = oy2 - oy1, ny2 - ny1
        if min(old_height, new_height) <= 0:
            return False
        if cls._box_overlap_ratio(previous_box, current_box) < 0.1:
            return False

        return (
            cls._distance_to_box(previous_ball, previous_box) <= 0.15 * old_height
            and cls._distance_to_box(current_ball, current_box) <= 0.45 * new_height
            and math.dist(previous_ball, current_ball) >= max(20.0, 0.2 * min(old_height, new_height))
        )

    @staticmethod
    def _box_overlap_ratio(first, second):
        x1, y1, x2, y2 = first
        nx1, ny1, nx2, ny2 = second
        intersection = max(0.0, min(x2, nx2) - max(x1, nx1)) * max(
            0.0, min(y2, ny2) - max(y1, ny1)
        )
        first_area = (x2 - x1) * (y2 - y1)
        second_area = (nx2 - nx1) * (ny2 - ny1)
        union = first_area + second_area - intersection
        return intersection / max(union, 1.0)

    @staticmethod
    def _distance_to_box(point, box):
        x, y = point
        x1, y1, x2, y2 = box
        return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))

    @staticmethod
    def _player_box(player):
        box = player.get("box") or ()
        return tuple(map(float, box[:4])) if len(box) >= 4 else None

    @staticmethod
    def _player_id(player):
        return player.get("id") if player.get("id") is not None else player.get("local_track_id")

    @classmethod
    def _ball_holder(cls, row):
        ball = row.get("ball") or {}
        center = ball.get("center")
        if (
            not ball.get("observed", not ball.get("interpolated", False))
            or center is None
            or len(center) < 2
        ):
            return None
        candidates = []
        for player in row.get("players", []):
            box = player.get("box") or ()
            wrists = pose_wrists(player.get("pose"))
            if cls._player_id(player) is None or len(box) < 4 or not wrists:
                continue
            height = max(float(box[3]) - float(box[1]), 1.0)
            distance = min(math.dist(center[:2], wrist) for wrist in wrists) / height
            if distance <= 0.35:
                candidates.append((distance, player))
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.08:
            return None
        return candidates[0][1]

    def _pass(self, frame, previous_holder, previous_team, previous_frame, team, player, row):
        team = normalize_team(team)
        previous_team = normalize_team(previous_team)
        box = player.get("box") or ()
        flight_frames = None if self.unheld_since is None else self.candidate_start - self.unheld_since
        if (
            self.pending_miss is not None
            or previous_holder is None
            or previous_frame is None
            or frame - previous_frame > round(1.5 * self.fps)
            or team is None
            or (previous_team is not None and previous_team != team)
            or len(box) < 4
        ):
            return None
        flight_pass = (
            flight_frames is not None
            and flight_frames >= 2
            and self.last_ball_position is not None
            and self.candidate_ball_position is not None
            and math.dist(self.last_ball_position, self.candidate_ball_position)
            >= max(50.0, 0.4 * (float(box[3]) - float(box[1])))
        )
        direct_pass = self._direct_pass_supported(row, previous_holder, player)
        if not flight_pass and not direct_pass:
            return None
        start = int(self.unheld_since if self.unheld_since is not None else self.candidate_start)
        return {
            "event": "pass",
            "start_frame": start,
            "end_frame": int(frame),
            "completed_frame": int(frame),
            "start": round(max(0.0, (start - 1) / self.fps), 4),
            "end": round(max(0.0, (frame - 1) / self.fps), 4),
            "player_id": previous_holder,
            "receiver_id": self.last_holder,
            "team": team,
        }

    def _direct_pass_supported(self, row, previous_holder, receiver):
        ball = row.get("ball") or {}
        center = ball.get("center")
        if not ball.get("observed", not ball.get("interpolated", False)) or center is None or len(center) < 2:
            return False
        players = {str(player.get("id")): player for player in row.get("players", [])}
        sender = players.get(str(previous_holder))
        old_box = None if sender is None else self._player_box(sender)
        new_box = self._player_box(receiver)
        if old_box is None or new_box is None:
            return False
        ox1, oy1, ox2, oy2 = old_box
        nx1, ny1, nx2, ny2 = new_box
        old_height, new_height = oy2 - oy1, ny2 - ny1
        if min(old_height, new_height) <= 0:
            return False
        center_gap = math.dist(((ox1 + ox2) / 2, (oy1 + oy2) / 2), ((nx1 + nx2) / 2, (ny1 + ny2) / 2))
        return (
            self._ball_near_player(row, receiver)
            and center_gap >= 0.45 * min(old_height, new_height)
            and self._box_overlap_ratio(old_box, new_box) < 0.15
            and self._distance_to_box(center[:2], old_box) >= 0.12 * old_height
        )

    @staticmethod
    def _team_vote(teams, fallback=None):
        known = [team for team in teams if normalize_team(team) is not None]
        if not known:
            return fallback
        winner = max(set(known), key=known.count)
        if known.count(winner) < 2 or any(
            team != winner and known.count(team) >= known.count(winner)
            for team in known
        ):
            return fallback
        return winner

    @staticmethod
    def _ball_near_player(row, player):
        ball = row.get("ball") or {}
        center = ball.get("center")
        box = player.get("box")
        if (
            not ball.get("observed", not ball.get("interpolated", False))
            or center is None
            or len(center) < 2
            or box is None
            or len(box) < 4
        ):
            return False
        x1, y1, x2, y2 = map(float, box[:4])
        x, y = map(float, center[:2])
        return (
            x1 - 0.65 * (x2 - x1) <= x <= x2 + 0.65 * (x2 - x1)
            and y1 - 0.15 * (y2 - y1) <= y <= y2 + 0.15 * (y2 - y1)
        )

    def _reset_candidate(self):
        self.candidate_holder = None
        self.candidate_team = None
        self.candidate_teams = []
        self.candidate_start = None
        self.candidate_frames = 0
        self.candidate_ball_frames = 0
        self.candidate_ball_position = None
