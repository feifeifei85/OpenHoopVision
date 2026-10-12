import math
from collections import deque

from tracking.ball import DetectionBox, normalize_fps
from tracking.rim import RimTracker
from events.context import nearest_wrist_distance, player_team, pose_wrists, scorer_feature
from events.possession import PossessionEventDetector
from events.shot import ShotStateMachine


def _ball_inside_player(ball, player):
    box = None if player is None else player.get("box")
    if ball is None or ball.interpolated or box is None or len(box) < 4:
        return False
    x1, y1, x2, y2 = map(float, box[:4])
    padding = 0.05 * max(y2 - y1, 1.0)
    return x1 - padding <= ball.x <= x2 + padding and y1 - padding <= ball.y <= y2 + padding


class EventTimeline:
    """Resolve overlapping actions in shot, pass, dribble priority order."""

    _EXCLUSIVE_ACTIONS = frozenset({"shot", "dribble", "pass"})

    def __init__(self):
        self.pending_passes = []
        self.pending_dribbles = []
        self.pending_uncertain_shots = []
        self.events = deque(maxlen=64)

    @staticmethod
    def _interval(event):
        try:
            return int(event["start_frame"]), int(event["end_frame"])
        except (KeyError, TypeError, ValueError):
            return None

    def publish(self, events, active_shot_start=None, active_dribble_start=None):
        events = list(events or ())
        confirmed = [
            event for event in self.pending_passes + events
            if event.get("event") != "dribble"
            and not (event.get("event") == "shot" and event.get("result") == "unknown")
        ]
        action_priority = {"shot": 0, "pass": 1}
        confirmed.sort(key=lambda event: action_priority.get(event.get("event"), 2))
        self.pending_passes = []
        uncertain_shots = self.pending_uncertain_shots + [
            event for event in events if event.get("event") == "shot" and event.get("result") == "unknown"
        ]
        dribbles = self.pending_dribbles + [
            event for event in events if event.get("event") == "dribble"
        ]
        self.pending_dribbles = []
        self.pending_uncertain_shots = []
        published = []

        for event in confirmed:
            interval = self._interval(event)
            if (
                event.get("event") == "pass"
                and interval is not None
                and active_shot_start is not None
                and interval[1] >= active_shot_start
            ):
                self.pending_passes.append(event)
                continue
            if event.get("event") == "shot" and interval is not None:
                self._forget_overlapping_passes(interval)
            if interval is not None and self._overlaps(event, interval):
                continue
            published.append(event)
            if interval is not None:
                self.events.append(event)

        for event in dribbles:
            interval = self._interval(event)
            if interval is None:
                continue
            if active_shot_start is not None and interval[1] >= active_shot_start:
                self.pending_dribbles.append(event)
            elif not self._overlaps(event, interval):
                published.append(event)
                self.events.append(event)

        for event in uncertain_shots:
            interval = self._interval(event)
            if interval is None:
                continue
            if active_dribble_start is not None and interval[1] >= active_dribble_start:
                self.pending_uncertain_shots.append(event)
            elif self._overlaps(event, interval):
                continue
            else:
                published.append(event)
                self.events.append(event)
        return published

    def _forget_overlapping_passes(self, interval):
        start, end = interval
        for event in list(self.events):
            if event.get("event") != "pass":
                continue
            previous = self._interval(event)
            if previous is not None and start <= previous[1] and end >= previous[0]:
                self.events.remove(event)

    def _overlaps(self, candidate, interval):
        if candidate.get("event") not in self._EXCLUSIVE_ACTIONS:
            return False
        start, end = interval
        for previous_event in self.events:
            if previous_event.get("event") not in self._EXCLUSIVE_ACTIONS:
                continue
            previous = self._interval(previous_event)
            if previous is not None and start <= previous[1] and end >= previous[0]:
                return True
        return False

    def _accepted(self, status):
        if status is None:
            return False
        interval = self._interval(status)
        return interval is not None and any(
            status.get("event") == event.get("event")
            and interval == self._interval(event)
            for event in self.events
        )

    def status(self, shot_status, dribble_status):
        if shot_status is None:
            return (
                dribble_status
                if dribble_status is not None
                and (dribble_status.get("end_frame") is None or self._accepted(dribble_status))
                else None
            )
        phase = shot_status.get("result")
        shot_start = shot_status.get("start_frame")
        if phase in (ShotStateMachine.RISING, ShotStateMachine.DESCENDING):
            if dribble_status is not None and (
                dribble_status.get("end_frame") is None or self._accepted(dribble_status)
            ):
                dribble_end = dribble_status.get("end_frame")
                if dribble_end is None or shot_start is None or int(dribble_end) >= int(shot_start):
                    return dribble_status
            return {**shot_status, "event": "ball_motion"}
        if not self._accepted(shot_status):
            shot_status = None
        if (
            dribble_status is not None
            and dribble_status.get("end_frame") is not None
            and not self._accepted(dribble_status)
        ):
            dribble_status = None
        if shot_status is None:
            return dribble_status
        if dribble_status is None:
            return shot_status

        dribble_end = dribble_status.get("end_frame")
        shot_end = shot_status.get("end_frame")
        dribble_start = dribble_status.get("start_frame")
        if shot_end is not None and dribble_start is not None and int(dribble_start) > int(shot_end):
            return dribble_status
        if phase == "unknown":
            return dribble_status
        return shot_status


class BallEventPipeline:
    def __init__(self, fps, event_parameters=None, tracker_parameters=None):
        self.fps = normalize_fps(fps)
        self.event_parameters = event_parameters or {}
        max_gap_frames = (tracker_parameters or {}).get("max_gap_frames", 5)
        self.rims = RimTracker(self.event_parameters)
        self.machine = ShotStateMachine(self.fps, self.event_parameters, max_gap_frames)
        self.possession_events = PossessionEventDetector(self.fps)
        self.timeline = EventTimeline()
        self.last_rim_box = None
        self.last_scale = None
        self.last_scale_hint = None
        self.last_frame = 0
        self.last_time = 0.0

    def update(self, ball, row):
        frame = int(row.get("frame", self.last_frame + 1))
        row_time = row.get("time")
        t = max(0, frame - 1) / self.fps if row_time is None else float(row_time)
        self.last_frame, self.last_time = frame, t

        rim_dets = []
        detections = row.get("detections") or {}
        for item in detections.get("rim", []):
            box = item.get("box")
            if box and len(box) >= 4:
                rim_dets.append(DetectionBox(float(item.get("score") or 0.0), tuple(map(float, box[:4]))))
        rim_feature = row.get("rim") or {}
        if not rim_dets and rim_feature.get("box"):
            rim_dets.append(
                DetectionBox(
                    float(rim_feature.get("score") or 0.0),
                    tuple(map(float, rim_feature["box"][:4])),
                )
            )

        rim_box, scale, scale_hint = self.rims.update(rim_dets, ball, self.machine.locked_rim_center, frame)
        shot = self.machine.shot
        raw_rim_box = self.rims.current_detection_box
        if (
            rim_box is not None
            and shot is not None
            and shot["crossed"]
            and self.rims.drift > 0
            and raw_rim_box is not None
        ):
            rim_width = max(
                rim_box[2] - rim_box[0],
                raw_rim_box[2] - raw_rim_box[0],
                1.0,
            )
            pan_shift = math.dist(self.rims._center(rim_box), self.rims._center(raw_rim_box))
            if pan_shift > self.event_parameters.get("rim_crossing_pan_snap_widths", 1.0) * rim_width:
                rim_box = raw_rim_box
        self.last_rim_box = rim_box
        self.last_scale, self.last_scale_hint = scale, scale_hint

        tag = row.get("scorer_tag")
        if tag is None:
            tag = scorer_feature(row)
        row["scorer_tag"] = tag
        wrists, shooter = self._shooter(row, ball, tag)
        shooter["holder_id"] = row.get("holder_id")
        if ball is not None and not ball.interpolated and wrists:
            shooter["wrist_distance_px"] = nearest_wrist_distance(
                (ball.x, ball.y), wrists
            )
        events = self.machine.update(
            ball, rim_box, frame, t, scale, scale_hint, wrists, shooter, rim_fresh=bool(rim_dets)
        )
        active_shot_start = None if self.machine.shot is None else self.machine.shot["start_frame"]
        active_dribble = self.machine.active_dribble
        active_dribble_start = None if active_dribble is None else active_dribble["start_frame"]
        events = self.timeline.publish(events, active_shot_start, active_dribble_start)
        rebound_events = self.possession_events.update(row, events)
        events.extend(
            self.timeline.publish(rebound_events, active_shot_start, active_dribble_start)
        )
        row.update(self.status_fields_at(frame))
        return events

    def _shooter(self, row, ball, tag):
        players = row.get("players", [])
        scorer_candidates = row.get("scorer_candidates", players)
        holder_id = row.get("holder_id")
        holder_player = next(
            (p for p in players if holder_id is not None and str(p.get("id")) == str(holder_id)),
            None,
        )
        tagged_player = (
            next(
                (p for p in scorer_candidates if str(p.get("id")) == str(tag.get("id"))),
                None,
            )
            if tag is not None
            and tag.get("id") is not None
            and float(tag.get("confidence") or 0.0) >= 0.45
            else None
        )
        if tagged_player is None and tag is not None and tag.get("local_track_id") is not None:
            tagged_player = next(
                (
                    p for p in scorer_candidates
                    if str(p.get("local_track_id")) == str(tag.get("local_track_id"))
                ),
                None,
            )
        active_id = self.machine.active_player_id
        selected = next((p for p in players if active_id is not None and str(p.get("id")) == str(active_id)), None)
        if selected is None:
            selected = holder_player
            if (
                tagged_player is not None
                and float((tag or {}).get("confidence") or 0.0) >= 0.45
                and _ball_inside_player(ball, tagged_player)
                and not _ball_inside_player(ball, holder_player)
            ):
                selected = tagged_player
            if selected is None:
                selected = tagged_player
        if selected is None and ball is not None and not ball.interpolated:
            nearest, nearest_distance = None, math.inf
            for player in players:
                points = pose_wrists(player.get("pose"))
                if points:
                    distance = nearest_wrist_distance((ball.x, ball.y), points)
                    if distance < nearest_distance:
                        nearest, nearest_distance = player, distance
            selected = nearest
        holder_tag = tag if holder_id is not None and tag is not None and str(tag.get("id")) == str(holder_id) else None
        holder_wrists = [] if holder_player is None else pose_wrists(holder_player.get("pose"))
        holder_context = None
        if holder_player is not None:
            holder_context = {
                "player_id": holder_id,
                "team": player_team(holder_player),
                "scorer_tag": holder_tag,
                "player_box": holder_player.get("box"),
                "holder_id": holder_id,
            }
            if ball is not None and not ball.interpolated and holder_wrists:
                holder_context["wrist_distance_px"] = nearest_wrist_distance(
                    (ball.x, ball.y), holder_wrists
                )
        if selected is None:
            return None, {
                "player_id": None if tag is None else tag.get("id"),
                "local_track_id": None if tag is None else tag.get("local_track_id"),
                "team": None if tag is None else tag.get("team_semantic"),
                "scorer_tag": tag,
                "player_box": None,
                "court_xy": None,
                "holder_id": holder_id,
                "holder_scorer_tag": holder_tag,
                "holder_context": holder_context,
                "holder_wrists": holder_wrists,
            }
        attached_tag = (
            tag
            if tag is not None
            and tag.get("id") is not None
            and selected.get("id") is not None
            and str(tag["id"]) == str(selected["id"])
            else None
        )
        return pose_wrists(selected.get("pose")), {
            "player_id": selected.get("id"),
            "local_track_id": selected.get("local_track_id", selected.get("id")),
            "team": player_team(selected),
            "scorer_tag": attached_tag,
            "player_box": selected.get("box"),
            "court_xy": selected.get("court_xy"),
            "holder_id": holder_id,
            "holder_scorer_tag": holder_tag,
            "holder_context": holder_context,
            "holder_wrists": holder_wrists,
        }

    def status_at(self, frame):
        return self.timeline.status(
            self.machine.status_at(frame),
            self.machine.dribble_status_at(frame),
        )

    def status_fields_at(self, frame):
        status = self.status_at(frame)
        kind = None if status is None else status.get("event")
        return {
            "event_status": status if kind in ("shot", "dribble", "pass") else None,
            "ball_status": status if kind == "ball_motion" else None,
            "shot_status": status if kind == "shot" else None,
            "dribble_status": status if kind == "dribble" else None,
        }

    def flush(self):
        events = self.machine.flush(self.last_frame, self.last_time, None, self.last_scale, self.last_scale_hint)
        for event in events:
            if event.get("result") == "miss" and event.get("verdict_confidence") == "occluded":
                event.update(
                    result="unknown",
                    verdict_confidence="trajectory_only",
                    result_source="eof_truncated_unresolved",
                    points=0,
                    score_applied=False,
                )
            end_frame = min(int(event.get("end_frame", self.last_frame)), self.last_frame)
            event["end_frame"] = end_frame
            event["end"] = round(max(0.0, (end_frame - 1) / self.fps), 4)
            if event.get("event") == "shot":
                event["completed_frame"] = end_frame
            if event is self.machine.last_event:
                self.machine.last_event_frame = end_frame
        return self.timeline.publish(events)
