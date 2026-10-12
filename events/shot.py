from collections import deque
import math

from events.scoring import shot_point_value
from tracking.ball import BallTrack, normalize_fps
from events.context import event_team, nearest_wrist_distance, normalize_team


class ShotOutcomeResolver:
    """Apply shot verdicts and build the stable event record."""

    def __init__(self, machine):
        self.machine = machine

    def ended(self, rim_box, frame, confidence):
        machine = self.machine
        shot = machine.shot
        if shot is None or not shot["entered"]:
            machine._discard()
            return None
        if not shot["crossed"] and rim_box is not None and shot["min_y"] > rim_box[3]:
            machine._discard()
            return None
        return self.resolve("miss", confidence, frame)

    def resolve(self, result, confidence, frame):
        machine = self.machine
        shot = machine.shot
        scorer_tag = shot.get("scorer_tag") or {}
        scorer_confidence = float(scorer_tag.get("confidence") or 0.0)
        if result == "miss" and not shot["crossed"]:
            duration = (int(frame) - shot["start_frame"]) / machine.fps
            if duration < machine.parameters.get('noncross_min_flight_seconds', 0.65):
                observed = [sample for sample in shot["samples"] if not sample[4]]
                rim_width = max(float(shot.get("rim_width") or 0.0), 1.0)
                rise = shot["release_y"] - shot["min_y"]
                descent = observed[-1][3] - min(sample[3] for sample in observed) if observed else 0.0
                holder_fraction = shot["holder_frames"] / max(shot["flight_context_frames"], 1)
                minimum_arc_widths = machine.parameters.get('short_noncross_min_arc_rim_widths', 0.75)
                short_scorer_path = (
                    duration >= machine.parameters.get('no_rim_min_flight_seconds', 0.4)
                    and shot["entered"]
                    and scorer_confidence >= machine.parameters.get('no_rim_entry_scorer_min_conf', 0.5)
                    and shot.get("min_rim_center_distance_widths", math.inf)
                    <= machine.parameters.get('no_rim_entry_scorer_max_dist_w', 2.0)
                    and len(observed) >= machine.parameters.get('no_rim_min_observed_samples', 8)
                    and rise >= minimum_arc_widths * rim_width
                    and descent >= minimum_arc_widths * rim_width
                    and holder_fraction < machine.parameters.get('persistent_holder_min_fraction', 0.85)
                )
                if not short_scorer_path:
                    machine._discard()
                    machine.await_exit = True
                    return None
        if not shot["crossed"]:
            if scorer_confidence < machine.parameters.get('late_entry_min_scorer_confidence', 0.3):
                distance_parameter = "late_entry_max_rim_dist_w" if shot["late_entry"] else "noncross_max_rim_dist_w"
                max_distance = machine.parameters.get(
                    distance_parameter,
                    1.25 if shot["late_entry"] else 1.85,
                )
                wrist_distance = shot.get("release_wrist_distance_widths")
                rim_distance = shot.get("min_rim_center_distance_widths", math.inf)
                wrist_too_far = (
                    not shot["late_entry"]
                    and wrist_distance is not None
                    and wrist_distance > machine.parameters.get('noncross_max_wrist_dist_w', 6.0)
                )
                weak_release_far = (
                    not shot["late_entry"]
                    and rim_distance > machine.parameters.get('late_entry_max_rim_dist_w', 1.25)
                    and not shot.get("strong_release", False)
                )
                if rim_distance > max_distance or wrist_too_far or weak_release_far:
                    machine._discard()
                    machine.await_exit = True
                    return None
            if machine.crossing.persistent_holder():
                machine._discard()
                machine.await_exit = True
                return None
        event_scorer_tag = shot.get("team_scorer_tag") or shot.get("peak_scorer_tag") or scorer_tag
        court_points = shot_point_value(shot.get("release_court_xy")) if result == "made" else None
        event = self.event_record(
            shot,
            frame,
            result,
            confidence,
            "observed_rim_crossing" if shot["crossed"] else "observed_rim_trajectory",
            event_scorer_tag,
            points=(court_points or 2) if result == "made" else 0,
        )
        if result == "made":
            event["points_source"] = "court_location" if court_points is not None else "two_point_fallback"
        machine.last_event = event
        machine.last_event_frame = int(frame)
        machine.shot = None
        machine.phase = machine.RESOLVED
        machine.await_exit = True
        return event

    def event_record(self, shot, frame, result, confidence, result_source, scorer_tag, points=None):
        machine = self.machine
        event = {
            "event": "shot",
            "shot_id": int(shot["shot_id"]),
            "start_frame": int(shot["start_frame"]),
            "end_frame": int(frame),
            "completed_frame": int(frame),
            "start": round(
                max(0.0, (shot["start_frame"] - 1) / machine.fps), 4
            ),
            "end": round(max(0.0, (frame - 1) / machine.fps), 4),
            "result": result,
            "verdict_confidence": confidence,
            "result_source": result_source,
            "player_id": shot["player_id"],
            "local_track_id": shot.get("local_track_id"),
            "team": normalize_team(shot.get("team")) or event_team({"scorer_tag": scorer_tag}),
            "scorer_tag": scorer_tag,
        }
        if points is not None:
            event["points"] = points
        event["score_applied"] = False
        return event


class ShotRimCrossing:
    """Track rim crossings and resolve the resulting shot outcome."""

    def __init__(self, machine):
        self.machine = machine

    def persistent_holder(self, min_context_frames=None):
        machine = self.machine
        min_context_frames = (
            machine.parameters.get('persist_holder_min_context', 0)
            if min_context_frames is None
            else min_context_frames
        )
        shot = machine.shot
        if shot is None or shot["player_id"] is None:
            return False
        context = shot["flight_context_frames"]
        holder_fraction = shot["holder_frames"] / max(context, 1)
        inside_fraction = shot["ball_inside_player_box_frames"] / max(shot["player_box_frames"], 1)
        return (
            context >= min_context_frames
            and holder_fraction >= machine.parameters.get('persistent_holder_min_fraction', 0.85)
            and inside_fraction >= machine.parameters.get('persistent_holder_box_min_fraction', 0.5)
        )

    def crossing(self, previous, ball, rim_box, frame):
        machine = self.machine
        if previous is None:
            return
        pf, _, px, py, _ = previous
        gap = frame - pf
        if gap < 1 or gap > machine.parameters.get('crossing_bridge_max_frames', 14):
            return
        x1, top, x2, _ = rim_box
        if ball.y <= py or not py < top <= ball.y:
            return
        fraction = (top - py) / max(ball.y - py, 1e-06)
        cross_x = px + fraction * (ball.x - px)
        rim_width = max(x2 - x1, 1.0)
        # Allow only a small rim-width-scaled edge error across short occlusions.
        edge_tolerance = machine.parameters.get('occluded_cross_edge_tol_w', 0.15) * rim_width if gap > 1 else 0.0
        if not x1 - edge_tolerance <= cross_x <= x2 + edge_tolerance:
            return
        shot = machine.shot
        if shot["crossed"]:
            shot["rattled"] = True
            if shot["popped_out"]:
                shot["recrossed"] = True
        else:
            shot["crossed"] = True
            shot["crossing_frame"] = frame
            shot["crossing_x"] = cross_x
            shot["crossing_occluded"] = gap > 1
            rim_center_x = (x1 + x2) * 0.5
            shot["crossing_edge"] = (
                gap > 1
                and abs(cross_x - rim_center_x) / rim_width
                >= machine.parameters.get('occluded_edge_cross_margin_w', 0.35)
            )

    def after_crossing(self, ball, rim_box, frame):
        machine = self.machine
        shot = machine.shot
        x1, top, x2, bottom = rim_box
        width = max(x2 - x1, 1.0)
        span_margin = machine.parameters.get('crossing_span_margin_widths', 0.25) * width
        in_span = x1 - span_margin <= ball.x <= x2 + span_margin
        if (
            shot.get("near_rim_rearmed_after_unknown")
            and shot.get("crossing_frame") == frame
            and not shot["crossing_occluded"]
            and ball.vy > 0
            and ball.bbox is not None
            and ball.y >= top
            and ball.bbox[3] > bottom
            and in_span
        ):
            return machine.outcomes.resolve("made", "clean", frame)
        if ball.y > bottom:
            bridged_through_rim = shot["crossing_occluded"] and shot["crossing_x"] is not None
            if bridged_through_rim and shot["crossing_edge"]:
                center_x = (x1 + x2) * 0.5
                clear_widths = (ball.y - bottom) / width
                lateral_widths = abs(ball.x - center_x) / width
                if (
                    clear_widths < machine.parameters.get('occluded_edge_make_clearance_widths', 2.0)
                    or lateral_widths < machine.parameters.get('occluded_edge_make_lateral_widths', 0.5)
                ):
                    # A long-gap crossing near the rim edge can be a tracker
                    # ID switch or a ball popping back out. Wait for a clear
                    # below-rim exit before calling it a make.
                    if machine._near_rim(ball, rim_box):
                        return None
                    return machine.outcomes.resolve("miss", "occluded", frame)
            if (in_span or bridged_through_rim) and not (shot["popped_out"] and not shot["recrossed"]):
                confidence = "rattled" if shot["rattled"] else ("occluded" if shot["crossing_occluded"] else "clean")
                return machine.outcomes.resolve("made", confidence, frame)
            return machine.outcomes.resolve("miss", "occluded" if shot["crossing_occluded"] else "rattled", frame)
        if ball.y < top:
            shot["rattled"] = True
            shot["popped_out"] = True
            if top - ball.y > machine.parameters.get('rattle_pop_max_rim_widths', 1.1) * width:
                return machine.outcomes.resolve("miss", "rattled", frame)
        if ball.vy < 0:
            shot["rattled"] = True
        if not machine._near_rim(ball, rim_box):
            confidence = "occluded" if shot["crossing_occluded"] else ("rattled" if shot["rattled"] else "clean")
            return machine.outcomes.resolve("miss", confidence, frame)
        return None

    def without_crossing(self, ball, rim_box, frame):
        machine = self.machine
        shot = machine.shot
        if shot["entered"] and not machine._near_rim(ball, rim_box) and ball.y > rim_box[1]:
            if shot["min_y"] > rim_box[3]:
                if shot.get("holder_scorer_release"):
                    deadline = shot.setdefault(
                        "id_switch_grace_until",
                        int(frame) + round(machine.fps * machine.parameters.get('holder_id_switch_grace_s', 0.25)),
                    )
                    if int(frame) <= deadline:
                        return None
                machine._discard()
                return None
            return machine.outcomes.resolve("miss", "clean", frame)
        if not shot["entered"] and ball.y >= shot["release_y"]:
            machine._discard()
        return None


class ShotTrajectoryAnalyzer:
    """Recover shot events from ball motion around tracking gaps."""

    def __init__(self, machine):
        self.machine = machine

    def late_rim_entry(self, ball, rim_box, frame, t, scale, shooter, rim_fresh):
        machine = self.machine
        minimum_scorer_confidence = machine.parameters.get('late_entry_min_scorer_confidence', 0.3)
        if (
            not ball.velocity_valid
            or rim_box is None
            or not machine._near_rim(ball, rim_box)
            or ball.vy
            < max(
                machine.parameters.get('late_entry_min_downward_speed_px_s', 0.0),
                machine.parameters.get('release_downward_scale_ratio', 0.35) * (scale or 0.0),
            )
        ):
            return None

        history = list(machine.motion_history)
        upward_limit = max(
            machine.parameters.get('late_entry_min_upward_speed_px_s', 250.0),
            machine.parameters.get('release_min_upward_speed_ms', 1.2) * (scale or 0.0),
            machine.parameters.get('release_min_upward_speed_px_s', 300.0),
        )
        upward = [index for index, sample in enumerate(history[:-1]) if sample[6] and sample[5] <= -upward_limit]
        # Near-rim descent alone is not enough to identify a shot. If the ball
        # track joined after its apex, require an associated scorer detection.
        scorer_confidence = 0.0
        if not upward:
            tag = None if shooter is None else shooter.get("scorer_tag")
            scorer_confidence = 0.0 if tag is None else float(tag.get("confidence") or 0.0)
            if scorer_confidence < minimum_scorer_confidence:
                return None
            # A scorer-linked late entry can use a softer ascent threshold to
            # recover the ball's pre-apex history after a tracker ID switch.
            # Ordinary releases keep the stricter ShotTracker-style threshold.
            upward_limit = max(
                machine.parameters.get('late_entry_scorer_min_up_px_s', 100.0),
                machine.parameters.get('late_entry_scorer_up_speed_ratio', 0.35) * upward_limit,
            )
            upward = [index for index, sample in enumerate(history[:-1]) if sample[6] and sample[5] <= -upward_limit]
        start_index = upward[-1] if upward else len(history) - 1
        gap_limit = machine.max_gap_frames + 1
        index = start_index
        while (
            index > 0
            and history[index][0] - history[index - 1][0] <= gap_limit
            and history[index - 1][6]
            and history[index - 1][5] <= -upward_limit
        ):
            index -= 1
            start_index = index

        selected = history[start_index:]
        start = selected[0]
        # A late-entry recovery may inspect motion samples from a shot that
        # was already resolved. Do not emit a second event from that history.
        if start[0] <= machine.last_event_frame:
            return None
        current = history[-1]
        previous = history[-2] if len(history) > 1 else current
        if frame - previous[0] > gap_limit:
            previous = current
        samples = [(s[0], s[2], s[3], s[4], False) for s in selected]
        last_observed = (previous[0], previous[2], previous[3], previous[4], False)
        machine.flight_context.begin_shot(
            frame,
            start[0],
            start[2],
            start[4],
            samples,
            last_observed,
            rim_box,
            shooter,
            entered=True,
            phase=machine.DESCENDING,
        )
        # If a scorer-linked ball was reacquired after its apex, the retained
        # trajectory may already contain the rim crossing. Replay those
        # observed samples against the fresh rim box before processing this
        # frame, so a late entry can still resolve the actual result.
        if scorer_confidence >= minimum_scorer_confidence and rim_fresh:
            for previous_sample, current_sample in zip(selected, selected[1:]):
                if current_sample[0] >= frame:
                    break
                previous = (
                    previous_sample[0],
                    previous_sample[2],
                    previous_sample[3],
                    previous_sample[4],
                    False,
                )
                current_ball = BallTrack(
                    current_sample[3],
                    current_sample[4],
                    0.0,
                    current_sample[5],
                    False,
                    velocity_valid=current_sample[6],
                )
                machine.crossing.crossing(previous, current_ball, rim_box, current_sample[0])
        return machine._flight(ball, rim_box, frame, float(t), rim_fresh, shooter)

    def discarded_flight_event(self, shot):
        machine = self.machine
        if shot is None or shot["crossed"] or shot["player_id"] is None:
            return None
        samples = [sample for sample in shot["samples"] if not sample[4]]
        if len(samples) < machine.parameters.get('no_rim_min_observed_samples', 8):
            return None

        rise = samples[0][3] - min(sample[3] for sample in samples)
        apex_y = min(sample[3] for sample in samples)
        descent = samples[-1][3] - apex_y
        duration = samples[-1][1] - shot["start_time"]
        holder_fraction = shot["holder_frames"] / max(shot["flight_context_frames"], 1)
        inside_fraction = shot["ball_inside_player_box_frames"] / max(shot["player_box_frames"], 1)
        scorer_tag = shot.get("peak_scorer_tag") or shot.get("scorer_tag") or {}
        scorer_confidence = float(scorer_tag.get("confidence") or 0.0)
        min_rim_distance = shot.get("min_rim_center_distance_widths", math.inf)
        strong_scorer_path = (
            shot["entered"]
            and min_rim_distance <= machine.parameters.get('no_rim_entry_scorer_max_dist_w', 2.0)
            and rise >= machine.parameters.get('strong_unresolved_rise_px', 100.0)
            and scorer_confidence
            >= machine.parameters.get('no_rim_strong_scorer_confidence', 0.6)
        )
        holder_scorer_path = (
            not shot["entered"]
            and scorer_confidence >= machine.parameters.get('no_rim_holder_scorer_confidence', 0.4)
            and holder_fraction >= machine.parameters.get('no_rim_holder_min_fraction', 0.85)
            and inside_fraction >= machine.parameters.get('no_rim_holder_box_min_fraction', 0.6)
            and rise >= machine.parameters.get('no_rim_holder_min_rise_px', 60.0)
        )
        near_rim_path = (
            shot["entered"]
            and len(samples) >= machine.parameters.get('no_rim_near_min_observed_samples', 10)
            and scorer_confidence >= machine.parameters.get('no_rim_near_min_scorer_confidence', 0.35)
            and duration >= machine.parameters.get('no_rim_near_min_flight_seconds', 0.5)
            and rise >= machine.parameters.get('no_rim_near_min_rise_px', 50.0)
            and min_rim_distance <= machine.parameters.get('no_rim_near_max_distance_widths', 3.0)
            and holder_fraction >= machine.parameters.get('no_rim_near_holder_min_fraction', 0.85)
            and inside_fraction >= machine.parameters.get('no_rim_near_holder_box_min_fraction', 0.5)
        )
        near_rim_frames = int(shot.get("near_rim_frames", 0))
        persistent_near_rim = (
            shot["entered"]
            and duration >= machine.parameters.get('no_rim_persist_near_min_flight_s', 1.0)
            and near_rim_frames
            >= round(machine.fps * machine.parameters.get('no_rim_persist_near_min_duration_s', 0.5))
            and near_rim_frames / max(len(samples), 1)
            >= machine.parameters.get('no_rim_persistent_near_min_fraction', 0.5)
            and min_rim_distance <= machine.parameters.get('no_rim_persist_near_max_dist_w', 0.75)
            and holder_fraction < machine.parameters.get('persistent_holder_min_fraction', 0.85)
        )
        normal_arc_path = (
            duration >= machine.parameters.get('no_rim_min_flight_seconds', 0.4)
            and rise >= machine.parameters.get('no_rim_min_rise_px', 50.0)
            and descent >= machine.parameters.get('no_rim_min_descent_px', 40.0)
            and (strong_scorer_path or holder_scorer_path or near_rim_path)
        )
        if not (normal_arc_path or persistent_near_rim):
            return None

        previous_event = machine.last_event or {}
        if (
            abs(int(shot["start_frame"]) - int(previous_event.get("start_frame", -(10**9))))
            <= machine.parameters.get('recent_event_suppression_frames', 30)
        ):
            return None

        frame = int(machine.last_frame)
        result_source = (
            "persistent_near_rim_unresolved_trajectory"
            if persistent_near_rim
            else "scorer_confirmed_unresolved_trajectory"
        )
        return machine.outcomes.event_record(shot, frame, "unknown", "trajectory_only", result_source, scorer_tag)


class ShotFlightContext:
    def __init__(self, machine):
        self.machine = machine

    def remember_motion(self, ball, frame, t):
        machine = self.machine
        if ball is None or ball.interpolated:
            return
        history = machine.motion_history
        max_gap = machine.max_gap_frames
        if history and (frame - history[-1][0] > max_gap or ball.track_id != history[-1][1]):
            history.clear()
        history.append((frame, ball.track_id, t, ball.x, ball.y, ball.vy, ball.velocity_valid))

    def begin_shot(
        self,
        shot_id,
        start_frame,
        start_time,
        release_y,
        samples,
        last_observed,
        rim_box,
        shooter,
        *,
        entered,
        phase,
        strong_release=False,
    ):
        machine = self.machine
        player_id = None if shooter is None else shooter.get("player_id")
        holder_id = None if shooter is None else shooter.get("holder_id")
        machine.shot = {
            "shot_id": int(shot_id),
            "start_frame": int(start_frame),
            "start_time": float(start_time),
            "release_y": float(release_y),
            "release_court_xy": None if shooter is None else shooter.get("court_xy"),
            "samples": list(samples),
            "last_observed": last_observed,
            "min_y": min(sample[3] for sample in samples),
            "entered": bool(entered),
            "near_rim_frames": 0,
            "late_entry": bool(entered),
            "strong_release": bool(strong_release),
            "crossed": False,
            "crossing_frame": None,
            "crossing_x": None,
            "crossing_occluded": False,
            "crossing_edge": False,
            "popped_out": False,
            "recrossed": False,
            "rattled": False,
            "unseen": 0,
            "player_id": player_id,
            "local_track_id": None if shooter is None else shooter.get("local_track_id"),
            "team": None if shooter is None else shooter.get("team"),
            "released_with_holder": (
                player_id is not None and holder_id is not None and str(player_id) == str(holder_id)
            ),
            "team_candidate": None,
            "team_candidate_frames": 0,
            "team_scorer_tag": None,
            "scorer_tag": None if shooter is None else shooter.get("scorer_tag"),
            "peak_scorer_tag": None if shooter is None else shooter.get("scorer_tag"),
            "release_wrist_distance_widths": (
                None
                if shooter is None or shooter.get("wrist_distance_px") is None or rim_box is None
                else float(shooter["wrist_distance_px"])
                / max(rim_box[2] - rim_box[0], 1.0)
            ),
            "flight_context_frames": 0,
            "holder_frames": 0,
            "player_box_frames": 0,
            "ball_inside_player_box_frames": 0,
            "rim_center": None if rim_box is None else machine._center(rim_box),
            "rim_width": (
                None if rim_box is None else max(rim_box[2] - rim_box[0], 1.0)
            ),
        }
        machine.phase = phase

    def observe_shooter_flight(self, ball, shooter):
        machine = self.machine
        shot = machine.shot
        if shot is None or shooter is None:
            return
        if normalize_team(shot.get("team")) is None:
            shot["team"] = normalize_team(shooter.get("team"))
        shot["flight_context_frames"] += 1
        if (
            shot["player_id"] is not None
            and shooter.get("holder_id") is not None
            and str(shooter["holder_id"]) == str(shot["player_id"])
        ):
            shot["holder_frames"] += 1
        tag = shooter.get("scorer_tag")
        if tag is not None and str(tag.get("id")) == str(shot["player_id"]):
            previous_tag = shot.get("peak_scorer_tag") or {}
            if float(tag.get("confidence") or 0.0) > float(previous_tag.get("confidence") or 0.0):
                shot["peak_scorer_tag"] = dict(tag)
            if normalize_team(shot.get("team")) is None:
                shot["team"] = event_team({"scorer_tag": tag})
            tag_team = event_team({"scorer_tag": tag})
            tag_confidence = float(tag.get("confidence") or 0.0)
            if (
                shot.get("released_with_holder")
                and tag_team is not None
                and tag_team != normalize_team(shot.get("team"))
                and tag_confidence >= machine.parameters.get('team_reassign_scorer_confidence', 0.8)
            ):
                if shot.get("team_candidate") == tag_team:
                    shot["team_candidate_frames"] += 1
                else:
                    shot["team_candidate"] = tag_team
                    shot["team_candidate_frames"] = 1
                if shot["team_candidate_frames"] >= machine.parameters.get('team_reassign_consecutive_frames', 2):
                    shot["team"] = tag_team
                    shot["team_scorer_tag"] = dict(tag)
                    shot["team_candidate"] = None
                    shot["team_candidate_frames"] = 0
            else:
                shot["team_candidate"] = None
                shot["team_candidate_frames"] = 0
        else:
            shot["team_candidate"] = None
            shot["team_candidate_frames"] = 0
        box = shooter.get("player_box")
        if (
            ball is None
            or ball.interpolated
            or box is None
            or shot["player_id"] is None
            or shooter.get("player_id") is None
            or str(shooter.get("player_id")) != str(shot["player_id"])
        ):
            return
        x1, y1, x2, y2 = map(float, box[:4])
        shot["player_box_frames"] += 1
        if x1 <= ball.x <= x2 and y1 <= ball.y <= y2:
            shot["ball_inside_player_box_frames"] += 1

    def remember_rim_distance(self, ball, rim_box, rim_fresh):
        machine = self.machine
        if machine.shot is None or ball is None or ball.interpolated or not rim_fresh or rim_box is None:
            return
        cx, cy = machine._center(rim_box)
        width = max(rim_box[2] - rim_box[0], 1.0)
        distance = math.hypot(ball.x - cx, ball.y - cy) / width
        machine.shot["min_rim_center_distance_widths"] = min(
            machine.shot.get("min_rim_center_distance_widths", math.inf), distance
        )


class ShotReleaseDetector:
    def __init__(self, machine):
        self.machine = machine

    def update(self, ball, rim_box, frame, t, scale, wrists, shooter, rim_fresh):
        machine = self.machine
        observed = ball is not None and not ball.interpolated
        unknown_rearmed = False
        if machine.await_exit and observed:
            if rim_box is None or not machine._near_rim(ball, rim_box):
                machine.await_exit = False
        if not observed:
            machine.release_streak.clear()
            return []

        if scale:
            strong = machine.parameters.get('release_min_upward_speed_ms', 1.2) * scale
            soft = machine.parameters.get('pose_release_min_speed_mps', 0.5) * scale
            separation = machine.parameters.get('release_separation_m', 0.35) * scale
        else:
            strong = machine.parameters.get('release_min_upward_speed_px_s', 300.0)
            soft = None
            separation = machine.parameters.get('release_separation_px', 60.0)
        separated = None
        if wrists:
            separated = nearest_wrist_distance((ball.x, ball.y), wrists) > separation
        if not ball.velocity_valid:
            machine.release_streak.clear()
            return []

        up_speed = -ball.vy
        if up_speed >= strong:
            rising = separated is True
        else:
            rising = (
                soft is not None
                and wrists is not None
                and len(wrists) == 2
                and bool(separated)
                and ball.y < min(y for _, y in wrists)
                and up_speed >= soft
            )
        holder_release = False
        tag = None if shooter is None else shooter.get("scorer_tag")
        holder_id = None if shooter is None else shooter.get("holder_id")
        player_id = None if shooter is None else shooter.get("player_id")
        if (
            not rising
            and up_speed >= machine.parameters.get('holder_release_min_up_px_s', 15.0)
            and holder_id is not None
            and player_id is not None
            and str(holder_id) == str(player_id)
            and tag is not None
            and str(tag.get("id")) == str(holder_id)
            and float(tag.get("confidence") or 0.0) >= machine.parameters.get('late_entry_min_scorer_confidence', 0.3)
        ):
            rising = True
            holder_release = True
        if not rising:
            machine.release_streak.clear()
            if machine.await_exit:
                return []
            late_event = machine.trajectory.late_rim_entry(ball, rim_box, frame, t, scale, shooter, rim_fresh)
            if late_event is not None:
                return late_event
            return []

        if machine.await_exit:
            last_player = None if machine.last_event is None else machine.last_event.get("player_id")
            player_id = None if shooter is None else shooter.get("player_id")
            last_was_unknown = machine.last_event is not None and machine.last_event.get("result") == "unknown"
            if (
                not last_was_unknown
                or player_id is None
                or (last_player is not None and str(player_id) == str(last_player))
            ):
                machine.release_streak.clear()
                return []
            machine.await_exit = False
            unknown_rearmed = True

        machine.release_streak.append((frame, t, ball.x, ball.y))
        if len(machine.release_streak) < machine.parameters.get('release_consecutive_frames', 1):
            return []
        first = machine.release_streak[0]
        samples = [(f, sample_t, x, y, False) for f, sample_t, x, y in machine.release_streak]
        machine.flight_context.begin_shot(
            frame,
            first[0],
            first[1],
            first[3],
            samples,
            samples[-1],
            rim_box,
            shooter,
            entered=False,
            phase=machine.RISING,
            strong_release=up_speed >= strong,
        )
        machine.shot["holder_scorer_release"] = holder_release
        machine.shot["near_rim_rearmed_after_unknown"] = unknown_rearmed
        machine.flight_context.observe_shooter_flight(ball, shooter)
        machine.flight_context.remember_rim_distance(ball, rim_box, rim_fresh)
        machine.release_streak.clear()
        return []


class ShotStateMachine:
    IDLE, RISING, DESCENDING, RESOLVED = "idle", "rising", "descending", "resolved"

    def __init__(self, fps, event_parameters, max_gap_frames=5):
        self.fps = normalize_fps(fps)
        self.parameters = event_parameters or {}
        self.max_gap_frames = max_gap_frames
        self.phase = self.IDLE
        self.shot = None
        self.release_streak = []
        self.await_exit = False
        self.last_event = None
        self.last_event_frame = -1
        self.last_frame = 0
        self.pending_events = []
        self.motion_history = deque(
            maxlen=max(
                self.parameters.get('motion_history_min_samples', 2),
                round(self.fps * self.parameters.get('motion_history_seconds', 2.0)),
            )
        )
        self.outcomes = ShotOutcomeResolver(self)
        self.crossing = ShotRimCrossing(self)
        self.trajectory = ShotTrajectoryAnalyzer(self)
        self.flight_context = ShotFlightContext(self)
        self.release_detector = ShotReleaseDetector(self)
        self.dribble_holder = None
        self.dribble_last_valid_time = None
        self.dribble_last_y = None
        self.dribble_direction = 0
        self.dribble_extreme = None
        self.dribble_top = None
        self.dribble_bottom = None
        self.dribble_cycle = None
        self.dribble_bounce_count = 0
        self.dribble_last_bounce = None
        self.active_dribble = None
        self.completed_dribbles = deque(maxlen=32)
        self.last_dribble_event = None
        self.last_dribble_event_frame = -1

    @property
    def active_player_id(self):
        return None if self.shot is None else self.shot["player_id"]

    @property
    def locked_rim_center(self):
        return None if self.shot is None else self.shot["rim_center"]

    def update(self, ball, rim_box, frame, t, scale, scale_hint, wrists, shooter, rim_fresh=False):
        frame = int(frame)
        self.last_frame = frame
        self.pending_events = []
        handoff_events = []
        if rim_box is not None:
            rim_box = tuple(map(float, rim_box))
        if self.shot is not None and self.shot["rim_center"] is None and rim_box is not None:
            self.shot["rim_center"] = self._center(rim_box)
            self.shot["rim_width"] = max(rim_box[2] - rim_box[0], 1.0)
        self.flight_context.remember_motion(ball, frame, float(t))
        dribble_events = self._update_dribble(ball, rim_box, frame, float(t), shooter)

        holder_id = None if shooter is None else shooter.get("holder_id")
        holder_tag = None if shooter is None else shooter.get("holder_scorer_tag")
        holder_context = None if shooter is None else shooter.get("holder_context")
        holder_wrists = None if shooter is None else shooter.get("holder_wrists")
        release_confirmed = (
            ball is not None
            and not ball.interpolated
            and ball.velocity_valid
            and holder_id is not None
            and holder_tag is not None
            and str(holder_tag.get("id")) == str(holder_id)
            and float(holder_tag.get("confidence") or 0.0) >= self.parameters.get('holder_switch_scorer_conf_min', 0.35)
            and -ball.vy >= self.parameters.get('holder_switch_release_min_up_px_s', 100.0)
        )
        if self.shot is not None and not self.shot["crossed"]:
            shot = self.shot
            different_holder = holder_id is not None and str(holder_id) != str(shot["player_id"])
            if different_holder:
                elapsed = (frame - shot["start_frame"]) / self.fps
                scorer_tag = shot.get("peak_scorer_tag") or shot.get("scorer_tag") or {}
                scorer_confidence = float(scorer_tag.get("confidence") or 0.0)
                if release_confirmed:
                    event = self.outcomes.resolve("miss", "rattled", frame - 1)
                    if self.shot is None:
                        if event is not None:
                            handoff_events.append(event)
                        self.await_exit = False
                        if holder_context is not None:
                            shooter, wrists = holder_context, holder_wrists
                elif (
                    elapsed >= self.parameters.get('noncross_min_flight_seconds', 0.65)
                    and scorer_confidence < self.parameters.get('late_entry_min_scorer_confidence', 0.3)
                ):
                    near_rim = (
                        ball is not None
                        and not ball.interpolated
                        and rim_box is not None
                        and self._near_rim(ball, rim_box)
                    )
                    if shot["near_rim_frames"] >= self.parameters.get('holder_switch_near_rim_min_samples', 3):
                        event = self.outcomes.resolve("miss", "rattled", frame)
                        if event is not None:
                            handoff_events.append(event)
                    elif not near_rim:
                        self._discard()
        elif self.shot is None and self.await_exit and release_confirmed:
            last_player = None if self.last_event is None else self.last_event.get("player_id")
            if last_player is None or str(holder_id) != str(last_player):
                self.await_exit = False
                if holder_context is not None:
                    shooter, wrists = holder_context, holder_wrists

        if self.phase == self.RESOLVED:
            self.phase = self.IDLE
        if self.phase == self.IDLE:
            shot_events = self.release_detector.update(
                ball, rim_box, frame, float(t), scale_hint or scale, wrists, shooter, rim_fresh
            )
        else:
            shot_events = self._flight(ball, rim_box, frame, float(t), rim_fresh, shooter)
        completed_shots = [
            event for event in handoff_events + (shot_events or [])
            if event.get("event") == "shot"
        ]
        if completed_shots and self.active_dribble is not None:
            event = self._finish_dribble(int(completed_shots[-1].get("start_frame", frame)) - 1)
            if event is not None:
                dribble_events.append(event)
        return handoff_events + (shot_events or []) + dribble_events + self.pending_events

    def _update_dribble(self, ball, rim_box, frame, t, shooter):
        context = None if shooter is None else shooter.get("holder_context")
        if context is not None and context.get("holder_id") is not None and context.get("player_box"):
            holder = {
                "id": context["holder_id"],
                "team": context.get("team"),
                "box": context["player_box"],
            }
            if self.dribble_holder is not None and str(holder["id"]) != str(self.dribble_holder["id"]):
                event = self._finish_dribble()
                self._reset_dribble_motion()
                self.dribble_bounce_count = 0
                events = [] if event is None else [event]
            else:
                events = []
            self.dribble_holder = holder
        else:
            events = []
            holder = self.dribble_holder
            if (
                holder is not None
                and self.dribble_last_valid_time is not None
                and t - self.dribble_last_valid_time <= 0.25
            ):
                context = {
                    "holder_id": holder["id"],
                    "team": holder["team"],
                    "player_box": holder["box"],
                }
            else:
                context = None

        sample = self._dribble_sample(ball, rim_box, context)
        if sample is None:
            if self._dribble_timed_out(t):
                event = self._finish_dribble()
                if event is not None:
                    events.append(event)
            if self.dribble_last_valid_time is None or t - self.dribble_last_valid_time > 0.25:
                self._reset_dribble_motion()
                if self.active_dribble is None:
                    self.dribble_holder = None
                    self.dribble_bounce_count = 0
            return events

        self.dribble_last_valid_time = t
        relative_y, relative_x, rim_distance = sample
        current_weight = 0.70
        smoothed_y = (
            relative_y
            if self.dribble_last_y is None
            else current_weight * relative_y + (1.0 - current_weight) * self.dribble_last_y
        )
        if self.dribble_cycle is not None:
            self.dribble_cycle["min_y"] = min(self.dribble_cycle["min_y"], smoothed_y)
            self.dribble_cycle["horizontal"] = max(
                self.dribble_cycle["horizontal"],
                abs(relative_x - self.dribble_cycle["top_x"]),
            )
            if rim_distance is not None:
                self.dribble_cycle["rim_distance"] = min(self.dribble_cycle["rim_distance"], rim_distance)

        if self.dribble_direction == 0:
            if self.dribble_last_y is not None and self.dribble_extreme is not None:
                delta = smoothed_y - self.dribble_extreme[1]
                if abs(delta) >= 0.08:
                    if delta > 0 and self.dribble_top is None:
                        self._set_dribble_top(self.dribble_extreme, rim_distance)
                    self.dribble_direction = 1 if delta > 0 else -1
                    self.dribble_extreme = (frame, smoothed_y, relative_x)
            else:
                self.dribble_extreme = (frame, smoothed_y, relative_x)
        elif self.dribble_direction > 0:
            if smoothed_y >= self.dribble_extreme[1]:
                self.dribble_extreme = (frame, smoothed_y, relative_x)
            elif self.dribble_extreme[1] - smoothed_y >= 0.08:
                self._record_dribble_bottom(self.dribble_extreme)
                self.dribble_direction = -1
                self.dribble_extreme = (frame, smoothed_y, relative_x)
        else:
            if smoothed_y <= self.dribble_extreme[1]:
                self.dribble_extreme = (frame, smoothed_y, relative_x)
            elif smoothed_y - self.dribble_extreme[1] >= 0.08:
                self._record_dribble_top(self.dribble_extreme, rim_distance)
                self.dribble_direction = 1
                self.dribble_extreme = (frame, smoothed_y, relative_x)

        self.dribble_last_y = smoothed_y
        if self._dribble_timed_out(t):
            event = self._finish_dribble()
            if event is not None:
                events.append(event)
        return events

    def _dribble_timed_out(self, t):
        # Keep the 1.3 s cycle limit plus time to confirm the returning high point.
        return (
            self.active_dribble is not None
            and self.dribble_last_bounce is not None
            and t - self.dribble_last_bounce[1] >= 1.4
        )

    def _dribble_sample(self, ball, rim_box, context):
        if ball is None or ball.interpolated or context is None:
            return None
        box = context.get("player_box")
        if not box or len(box) < 4:
            return None
        x1, y1, x2, y2 = map(float, box[:4])
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height < 35.0:
            return None
        margin = width * 0.35
        if not x1 - margin <= ball.x <= x2 + margin:
            return None
        relative_y = (ball.y - y1) / height
        if not 0.25 <= relative_y <= 1.3:
            return None
        # Image y grows downward, so a physical high-low-high bounce is small-large-small.
        relative_x = (ball.x - (x1 + x2) * 0.5) / height
        rim_distance = None
        if rim_box is not None:
            rim_center = self._center(rim_box)
            rim_distance = math.dist((ball.x, ball.y), rim_center) / max(
                rim_box[2] - rim_box[0], 1.0
            )
        return relative_y, relative_x, rim_distance

    def _record_dribble_bottom(self, bottom):
        frame, y, _ = bottom
        if y < 0.68:
            self._reset_dribble_motion()
            if self.active_dribble is None:
                self.dribble_bounce_count = 0
                self.dribble_last_bounce = None
            return
        self.dribble_bottom = (int(frame), max(0.0, (int(frame) - 1) / self.fps), float(y))

    def _record_dribble_top(self, top, rim_distance):
        frame, y, _ = top
        if y > 0.68:
            self._reset_dribble_motion()
            return
        if self.dribble_top is None or self.dribble_bottom is None:
            self._set_dribble_top(top, rim_distance)
            return

        interval = max(0.0, (int(frame) - 1) / self.fps) - self.dribble_top[1]
        drop = self.dribble_bottom[2] - max(self.dribble_top[2], y)
        cycle = self.dribble_cycle or {}
        if (
            0.18 <= interval <= 1.3
            and self.dribble_bottom[2] >= 0.68
            and drop >= 0.18
            and cycle.get("min_y", 1.0) >= 0.25
            and cycle.get("horizontal", math.inf) <= 0.65
            and cycle.get("rim_distance", math.inf) >= 2.5
            and not (self.shot is not None and self.shot["crossed"])
        ):
            if self.shot is not None:
                self.shot = None
                self.phase = self.IDLE
                self.release_streak.clear()
            self.dribble_bounce_count += 1
            if self.active_dribble is None:
                self.active_dribble = {
                    "event": "dribble",
                    "start_frame": int(self.dribble_top[0]),
                    "start": round(self.dribble_top[1], 4),
                    "player_id": self.dribble_holder["id"],
                    "team": self.dribble_holder["team"],
                    "bounce_count": self.dribble_bounce_count,
                }
            else:
                self.active_dribble["bounce_count"] = self.dribble_bounce_count
            self.dribble_last_bounce = (int(frame), max(0.0, (int(frame) - 1) / self.fps))
        elif self.active_dribble is None:
            self.dribble_bounce_count = 0
            self.dribble_last_bounce = None

        self._set_dribble_top(top, rim_distance)

    def _set_dribble_top(self, top, rim_distance):
        frame, y, relative_x = top
        t = max(0.0, (int(frame) - 1) / self.fps)
        self.dribble_top = (int(frame), t, float(y), relative_x)
        self.dribble_bottom = None
        self.dribble_cycle = {
            "top_x": relative_x,
            "horizontal": 0.0,
            "min_y": y,
            "rim_distance": math.inf if rim_distance is None else rim_distance,
        }

    def _reset_dribble_motion(self):
        self.dribble_last_y = None
        self.dribble_direction = 0
        self.dribble_extreme = None
        self.dribble_top = None
        self.dribble_bottom = None
        self.dribble_cycle = None

    def _finish_dribble(self, end_frame=None):
        event = self.active_dribble
        if event is None:
            return None
        last_bounce = self.dribble_last_bounce
        default_end = last_bounce[0] if last_bounce else event["start_frame"]
        frame = int(default_end if end_frame is None else end_frame)
        event["end_frame"] = max(int(event["start_frame"]), frame)
        event["end"] = round(max(0.0, (event["end_frame"] - 1) / self.fps), 4)
        event["completed_frame"] = int(self.last_frame)
        self.active_dribble = None
        self.dribble_bounce_count = 0
        self.dribble_last_bounce = None
        self._reset_dribble_motion()
        self.completed_dribbles.append(event)
        self.last_dribble_event = event
        self.last_dribble_event_frame = int(self.last_frame)
        return event

    def dribble_status_at(self, frame):
        if self.shot is not None:
            return None
        event = self.active_dribble
        if (
            event is None
            and self.last_dribble_event is not None
            and int(frame) - self.last_dribble_event_frame <= round(0.4 * self.fps)
        ):
            event = self.last_dribble_event
        if event is None:
            return None
        return {
            "event": "dribble",
            "player_id": event.get("player_id"),
            "bounce_count": int(event.get("bounce_count", self.dribble_bounce_count)),
            "start_frame": event.get("start_frame"),
            "end_frame": event.get("end_frame"),
        }

    def attach_dribble_context(self, events):
        dribbles = list(self.completed_dribbles)
        if self.active_dribble is not None:
            dribbles.append(self.active_dribble)
        for event in events or ():
            if event.get("event") != "shot":
                continue
            player_id, start_frame = event.get("player_id"), event.get("start_frame")
            if player_id is None or start_frame is None:
                continue
            for dribble in reversed(dribbles):
                if str(dribble.get("player_id")) != str(player_id):
                    continue
                start = int(dribble.get("start_frame", 0))
                end = int(dribble.get("end_frame", self.last_frame))
                if start <= int(start_frame) <= end + round(0.5 * self.fps):
                    event["dribble_context"] = {
                        "start_frame": start,
                        "end_frame": end,
                        "bounce_count": int(dribble.get("bounce_count", 0)),
                    }
                    break

    def _flight(self, ball, rim_box, frame, t, rim_fresh=False, shooter=None):
        shot = self.shot
        self.flight_context.observe_shooter_flight(ball, shooter)
        self.flight_context.remember_rim_distance(ball, rim_box, rim_fresh)
        if (
            not shot["crossed"]
            and not shot["late_entry"]
            and self.crossing.persistent_holder(
                round(self.fps * self.parameters.get('persistent_holder_discard_s', 0.5))
            )
        ):
            self._discard()
            return []
        timed_out = t - shot["start_time"] > self.parameters.get('max_shot_seconds', 2.0)
        if ball is None:
            shot["unseen"] += 1
            if shot["unseen"] <= self.parameters.get('occlusion_hold_frames', 30) and not timed_out:
                return []
            event = self.outcomes.ended(rim_box, frame, "occluded")
            return [] if event is None else [event]

        shot["unseen"] = 0
        previous = shot["last_observed"]
        if not ball.interpolated:
            if rim_box is not None:
                self.crossing.crossing(previous, ball, rim_box, frame)
            sample = (frame, t, ball.x, ball.y, False)
            shot["last_observed"] = sample
            shot["samples"].append(sample)
            shot["min_y"] = min(shot["min_y"], ball.y)
        else:
            shot["samples"].append((frame, t, ball.x, ball.y, True))

        if rim_box is not None and self._near_rim(ball, rim_box):
            shot["entered"] = True
            if not ball.interpolated:
                shot["near_rim_frames"] += 1
        if self.phase == self.RISING:
            if shot["crossed"]:
                self.phase = self.DESCENDING
            elif ball.vy > 0:
                if ball.y < shot["release_y"] or shot["entered"]:
                    self.phase = self.DESCENDING
                else:
                    self._discard()
                    return []
        if self.phase != self.DESCENDING:
            event = None if not timed_out else self.outcomes.ended(rim_box, frame, "occluded")
            return [] if event is None else [event]
        if rim_box is None:
            if not ball.interpolated and not shot["entered"] and ball.y >= shot["release_y"]:
                self._discard()
                return []
            event = None if not timed_out else self.outcomes.ended(rim_box, frame, "occluded")
            return [] if event is None else [event]
        if ball.interpolated:
            event = None if not timed_out else self.outcomes.ended(rim_box, frame, "occluded")
            return [] if event is None else [event]

        event = (
            self.crossing.after_crossing(ball, rim_box, frame)
            if shot["crossed"]
            else self.crossing.without_crossing(ball, rim_box, frame)
        )
        if event is not None or self.shot is None:
            return [] if event is None else [event]
        event = None if not timed_out else self.outcomes.ended(rim_box, frame, "occluded")
        return [] if event is None else [event]

    def _discard(self):
        shot = self.shot
        event = self.trajectory.discarded_flight_event(shot)
        if event is not None:
            self.pending_events.append(event)
            self.last_event = event
            self.last_event_frame = int(event["completed_frame"])
            self.phase = self.RESOLVED
            self.await_exit = True
        else:
            self.phase = self.IDLE
        self.shot = None
        self.release_streak.clear()

    def _near_rim(self, ball, box):
        x1, y1, x2, y2 = box
        margin = self.parameters.get('rim_neighborhood_scale', 2.0) * max(x2 - x1, 1.0)
        return x1 - margin <= ball.x <= x2 + margin and y1 - margin <= ball.y <= y2 + margin

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    def status_at(self, frame):
        if self.phase in (self.RISING, self.DESCENDING):
            return {
                "event": "shot",
                "result": self.phase,
                "verdict_confidence": None,
                "start_frame": None if self.shot is None else self.shot["start_frame"],
                "player_id": self.active_player_id,
                "local_track_id": None if self.shot is None else self.shot.get("local_track_id"),
            }
        if self.last_event is not None and int(frame) - self.last_event_frame <= round(
            self.fps * self.parameters.get('status_linger_seconds', 2.5)
        ):
            return self.last_event
        return None

    def flush(self, frame, t, rim_box, scale, scale_hint):
        events = []
        steps = self.parameters.get('occlusion_hold_frames', 30) + self.parameters.get(
            'occlusion_flush_extra_frames', 2
        )
        for step in range(steps):
            if self.phase == self.IDLE:
                break
            next_frame = int(frame) + step + 1
            next_time = float(t) + (step + 1) / self.fps
            events.extend(self.update(None, rim_box, next_frame, next_time, scale, scale_hint, None, None))
            if self.phase == self.IDLE:
                break
        event = self._finish_dribble()
        if event is not None:
            events.append(event)
        return events
