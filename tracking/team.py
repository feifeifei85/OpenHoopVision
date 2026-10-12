from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from tracking.pose import validated_pose

_COLOR_PROFILE_SIZE = 12


def normalize_team(team: Any) -> str | None:
    """Normalize numeric and textual team IDs to the stable dark/light labels."""
    if team in ("dark", "light"):
        return team
    if isinstance(team, str):
        team = team.strip().lower()
        if team in ("0", "1"):
            team = int(team)
    if team in (0, 1):
        return "dark" if int(team) == 0 else "light"
    return None


class JerseyTeamClassifier:
    """Learn stable team IDs and shirt colors from local player observations."""

    def __init__(self) -> None:
        self.samples = {}
        self.last_seen = {}
        self.color_history = {}
        self.palette_samples = {}
        self.latest_color_features = {}
        self.team_votes = {}
        self.team_locked = {}
        self.team_last_vote_frame = {}
        self.centers = None
        self.identity_centers = None
        self.team_colors = None
        self.team_colors_locked = False
        self._color_stable_fits = 0
        self.fit_frame = -(10**9)

    def reset_segment_locks(self) -> None:
        self.team_votes.clear()
        self.team_locked.clear()
        self.team_last_vote_frame.clear()

    def observe(
        self,
        track_id: int,
        descriptor: np.ndarray | None,
        jersey_bgr: np.ndarray | None,
        frame_index: int | None = None,
        palette_sample: bool = True,
    ) -> int | None:
        """Add one jersey-color observation and return the current team vote."""
        track_id = int(track_id)
        if frame_index is not None:
            self.last_seen[track_id] = int(frame_index)
        if descriptor is None or jersey_bgr is None:
            return self.predict_track(track_id)
        descriptor = np.asarray(descriptor, dtype=np.float32)
        jersey_bgr = np.asarray(jersey_bgr, dtype=np.float32)
        color_pixel = np.clip(np.rint(jersey_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        lab_l = (
            float(cv2.cvtColor(color_pixel, cv2.COLOR_BGR2LAB)[0, 0, 0])
            / 255.0
        )
        hsv_s = (
            float(cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0, 0, 1])
            / 255.0
        )
        color_history = self.color_history.setdefault(
            track_id, deque(maxlen=120)
        )
        color_frame = int(frame_index) if frame_index is not None else -1
        color_history.append((color_frame, lab_l, hsv_s))
        self.latest_color_features[track_id] = (color_frame, lab_l, hsv_s)
        if palette_sample:
            palette_history = self.palette_samples.setdefault(
                track_id, deque(maxlen=12)
            )
            palette_history.append(jersey_bgr.copy())
        history = self.samples.setdefault(track_id, deque(maxlen=12))
        if history:
            previous_descriptor, previous_color = history[-1]
            descriptor = (1 - 0.25) * previous_descriptor + 0.25 * descriptor
            jersey_bgr = (1 - 0.25) * previous_color + 0.25 * jersey_bgr
        history.append((descriptor.copy(), jersey_bgr.copy()))
        return self.predict_track(track_id)

    def prune(
        self,
        active_ids: set[int],
        frame_index: int,
        fps: float,
        ttl_seconds: float | None = None,
    ) -> None:
        active = {int(track_id) for track_id in active_ids}
        current = int(frame_index)
        for track_id in active:
            self.last_seen[track_id] = current
        ttl_seconds = 10.0 if ttl_seconds is None else ttl_seconds
        ttl = max(1, round(float(fps) * float(ttl_seconds)))
        stale = [
            track_id
            for track_id in set(self.last_seen) | set(self.samples)
            if track_id not in active and current - self.last_seen.get(track_id, current) > ttl
        ]
        for track_id in stale:
            self.forget_track(track_id)
        inactive = sorted(
            (track_id for track_id in self.samples if track_id not in active),
            key=lambda track_id: self.last_seen.get(track_id, -(10**9)),
        )
        for track_id in inactive[: max(0, len(self.samples) - 30)]:
            self.forget_track(track_id)

    def forget_track(self, track_id: int) -> None:
        """Remove cached color/team state for a local track."""
        for store in (
            self.last_seen,
            self.samples,
            self.color_history,
            self.palette_samples,
            self.latest_color_features,
            self.team_votes,
            self.team_locked,
            self.team_last_vote_frame,
        ):
            store.pop(track_id, None)

    def maybe_fit(self, frame_index: int, force: bool = False) -> bool:
        ready = [
            track_id
            for track_id, history in self.samples.items()
            if len(history) >= 4
        ]
        if len(ready) < 4:
            return False
        if (
            not force
            and self.centers is not None
            and frame_index - self.fit_frame < 60
        ):
            return False
        summaries = np.asarray(
            [np.median(np.asarray(self.samples[track_id]), axis=0) for track_id in ready], np.float32
        )
        descriptors = summaries[:, 0]
        color_profiles = np.asarray(
            [
                np.median(
                    np.asarray(
                        list(self.color_history[track_id])[-_COLOR_PROFILE_SIZE:],
                        dtype=np.float32,
                    )[:, 1:3],
                    axis=0,
                )
                for track_id in ready
            ],
            np.float32,
        )
        color_centers = self._fit_identity_color_centers(color_profiles)
        if color_centers is not None:
            self.identity_centers = color_centers
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.0001,
        )
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(
            descriptors,
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1).astype(int)
        centers = centers.astype(np.float32)
        if np.linalg.norm(centers[0] - centers[1]) < 0.1:
            return False
        if self.centers is None:
            swap = centers[0, 0] > centers[1, 0]
        else:
            same = np.linalg.norm(centers - self.centers, axis=1).sum()
            swapped = np.linalg.norm(centers[::-1] - self.centers, axis=1).sum()
            swap = swapped < same
        if swap:
            centers = centers[::-1]
            labels = 1 - labels
        palette_tracks = [
            (track_id, labels[index])
            for index, track_id in enumerate(ready)
            if len(self.palette_samples.get(track_id, ())) >= 2
        ]
        candidate_colors = []
        for team_id in range(2):
            cluster = [
                np.median(np.asarray(self.palette_samples[track_id]), axis=0)
                for track_id, label in palette_tracks
                if label == team_id
            ]
            if not cluster:
                candidate_colors = None
                break
            representative = np.median(np.asarray(cluster), axis=0)
            if team_id == 1:
                lab = cv2.cvtColor(
                    np.asarray(representative, dtype=np.uint8).reshape(1, 1, 3),
                    cv2.COLOR_BGR2LAB,
                )[0, 0]
                if lab[0] >= 90 and np.linalg.norm(lab[1:].astype(np.float32) - 128) <= 18:
                    representative = np.full(3, 245, dtype=np.float32)
            candidate_colors.append(tuple(int(value) for value in np.clip(representative, 0, 255)))
        if candidate_colors is not None:
            if self.team_colors is None:
                self.team_colors = candidate_colors
                self._color_stable_fits = 1
            else:
                previous_colors = np.asarray(self.team_colors, dtype=np.float32)
                current_colors = np.asarray(candidate_colors, dtype=np.float32)
                color_delta = float(
                    np.mean(
                        np.linalg.norm(current_colors - previous_colors, axis=1)
                        / 255.0
                    )
                )
                self._color_stable_fits = (
                    self._color_stable_fits + 1
                    if color_delta <= 0.12
                    else 1
                )
                if not self.team_colors_locked:
                    self.team_colors = candidate_colors
                if self._color_stable_fits >= 3:
                    self.team_colors_locked = True
        self.centers = centers
        self.fit_frame = int(frame_index)
        return True

    @staticmethod
    def _fit_identity_color_centers(profiles):
        profiles = np.asarray(profiles, dtype=np.float32).reshape(-1, 2)
        if len(profiles) < 2 or not np.isfinite(profiles).all():
            return None
        distances = np.sum((profiles[:, None, :] - profiles[None, :, :]) ** 2, axis=2)
        if not np.isfinite(distances).all():
            return None
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        if first == second or distances[first, second] < 0.01:
            return None
        centers = np.stack((profiles[first], profiles[second])).astype(np.float32)
        for _ in range(50):
            labels = np.argmin(
                np.sum((profiles[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
            )
            counts = np.bincount(labels, minlength=2)
            if np.any(counts == 0):
                return None
            updated = np.stack(
                [profiles[labels == team_id].mean(axis=0) for team_id in range(2)]
            ).astype(np.float32)
            if not np.isfinite(updated).all():
                return None
            if np.linalg.norm(updated - centers) <= 0.0001:
                centers = updated
                break
            centers = updated
        if np.linalg.norm(centers[0] - centers[1]) < 0.1:
            return None
        if centers[0, 0] > centers[1, 0]:
            centers = centers[::-1]
        return centers

    def predict_raw_track(self, track_id: int) -> int | None:
        history = self.samples.get(int(track_id))
        if not history or self.centers is None:
            return None
        descriptor = history[-1][0]
        return int(np.argmin(np.linalg.norm(self.centers - descriptor[None, :], axis=1)))

    def _raw_predict_identity_track(self, track_id):
        history = self.color_history.get(int(track_id))
        if not history or self.identity_centers is None:
            return None
        profile = np.median(
            np.asarray(list(history)[-_COLOR_PROFILE_SIZE:], dtype=np.float32)[:, 1:3], axis=0
        )
        return int(np.argmin(np.linalg.norm(self.identity_centers - profile[None, :], axis=1)))

    def _stable_team(self, track_id):
        track_id = int(track_id)
        locked_team = self.team_locked.get(track_id)
        sample = self.latest_color_features.get(track_id)
        if sample is None or sample[0] < 0:
            return locked_team
        frame_index = int(sample[0])
        if self.team_last_vote_frame.get(track_id) == frame_index:
            return locked_team
        descriptor_team = self.predict_raw_track(track_id)
        color_team = self._raw_predict_identity_track(track_id)
        if descriptor_team is None and color_team is None:
            return locked_team
        if descriptor_team is None:
            vote = color_team
        elif color_team is None or descriptor_team == color_team:
            vote = descriptor_team
        else:
            vote = None
        votes = self.team_votes.setdefault(track_id, deque())
        self.team_last_vote_frame[track_id] = frame_index
        minimum_votes = 12
        minimum_share = 0.9
        minimum_span = 30

        # Track IDs can be reused after an occlusion or an ID switch. Keep a
        # stable label through noisy frames, but let sustained jersey evidence
        # correct a stale lock instead of freezing it for the rest of the clip.
        if locked_team is not None:
            if vote == locked_team:
                votes.clear()
                return int(locked_team)
            if vote is None:
                return int(locked_team)
            votes.append((frame_index, vote))
            while votes and frame_index - votes[0][0] > 60:
                votes.popleft()
            candidate_votes = [team for _, team in votes]
            counts = Counter(candidate_votes)
            candidate, count = counts.most_common(1)[0]
            span = votes[-1][0] - votes[0][0]
            if (
                count >= minimum_votes
                and count / len(candidate_votes) >= minimum_share
                and span >= minimum_span
            ):
                self.team_locked[track_id] = int(candidate)
                votes.clear()
                return int(candidate)
            return int(locked_team)

        votes.append((frame_index, vote))
        window = 120
        while votes and frame_index - votes[0][0] > window:
            votes.popleft()
        valid_votes = [(frame, team) for frame, team in votes if team is not None]
        if len(valid_votes) < minimum_votes:
            return None
        counts = Counter(team for _, team in votes if team is not None)
        team, count = counts.most_common(1)[0]
        share = count / len(votes)
        span = valid_votes[-1][0] - valid_votes[0][0]
        if (
            share >= minimum_share
            and span >= minimum_span
        ):
            self.team_locked[track_id] = int(team)
            votes.clear()
            return int(team)
        return None

    def predict_track(self, track_id: int) -> int | None:
        return self._stable_team(track_id)

    def predict_identity_track(self, track_id: int) -> int | None:
        return self.predict_track(track_id)

    def color_features_at(self, track_id: int, frame_index: int) -> tuple[float, float] | None:
        sample = self.latest_color_features.get(int(track_id))
        if sample is None or sample[0] != int(frame_index):
            return None
        return sample[1:]


def build_jersey_mask(
    frame_shape: tuple[int, ...],
    keypoints: np.ndarray,
    scores: np.ndarray,
    box: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a local image mask around the player's visible jersey torso."""
    height, width = frame_shape[:2]
    box = np.asarray(box, dtype=np.float32)
    box_size = np.maximum(box[2:] - box[:2], 1.0)
    padding = np.maximum(
        2.0,
        0.12 * box_size,
    )
    roi_start = np.maximum((0, 0), np.floor(box[:2] - padding)).astype(int)
    roi_end = np.minimum((width, height), np.ceil(box[2:] + padding)).astype(int)
    roi_end = np.maximum(roi_end, roi_start + 1)
    mask = np.zeros((roi_end[1] - roi_start[1], roi_end[0] - roi_start[0]), dtype=np.uint8)
    torso_ids = np.asarray([5, 6, 11, 12])
    pose = validated_pose(keypoints, scores, box, jersey_pose=True)
    torso_valid = (
        pose is not None
        and len(pose[0]) >= 13
        and np.all(pose[2][torso_ids])
    )
    if torso_valid:
        torso = pose[0][torso_ids].copy()
        axis = torso[2:].mean(axis=0) - torso[:2].mean(axis=0)
        torso[:2] += 0.08 * axis
        torso[2:] -= 0.16 * axis
        torso_shrink = 0.84
        torso[:2] = torso[:2].mean(axis=0) + torso_shrink * (torso[:2] - torso[:2].mean(axis=0))
        torso[2:] = torso[2:].mean(axis=0) + torso_shrink * (torso[2:] - torso[2:].mean(axis=0))
        polygon = torso[[0, 1, 3, 2]]
        polygon = (np.clip(polygon, (0, 0), (width - 1, height - 1)) - roi_start).astype(np.float32)
        if abs(float(cv2.contourArea(polygon))) >= 24.0:
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
            return mask, (*roi_start, *roi_end)
    body_start = np.maximum(
        roi_start,
        np.rint(box[:2] + np.asarray([0.24, 0.18]) * box_size).astype(int),
    )
    body_end = np.minimum(
        roi_end,
        np.rint(box[:2] + np.asarray([0.76, 0.58]) * box_size).astype(int),
    )
    if np.all(body_end > body_start):
        local_start = body_start - roi_start
        local_end = body_end - roi_start
        mask[local_start[1] : local_end[1], local_start[0] : local_end[0]] = 255
    return mask, (*roi_start, *roi_end)


def extract_jersey_color(
    frame: np.ndarray,
    jersey_region: tuple[np.ndarray, tuple[int, int, int, int]],
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Summarize jersey color pixels for classification and palette rendering."""
    jersey_mask, roi = jersey_region
    min_pixels = 80
    if jersey_mask is None or not np.any(jersey_mask):
        return None, None, False
    roi = roi or (0, 0, frame.shape[1], frame.shape[0])
    x0, y0, x1, y1 = map(int, roi)
    roi_frame = frame[y0:y1, x0:x1]
    if roi_frame.shape[:2] != jersey_mask.shape[:2]:
        raise ValueError("jersey ROI and local mask must have the same shape")
    erosion_kernel = 3
    eroded = cv2.erode(
        jersey_mask,
        np.ones((erosion_kernel, erosion_kernel), np.uint8),
        iterations=1,
    )
    pixels = roi_frame[eroded > 0]
    if len(pixels) < min_pixels:
        pixels = roi_frame[jersey_mask > 0]
    if len(pixels) < min_pixels:
        return None, None, False
    max_pixels = 6000
    if len(pixels) > max_pixels:
        pixels = pixels[:: max(1, len(pixels) // max_pixels)]
    pixel_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    median_lab = np.median(pixel_lab, axis=0)
    descriptor = np.asarray(
        (
            median_lab[0] / 255.0,
            (median_lab[1] - 128.0) / 127.0,
            (median_lab[2] - 128.0) / 127.0,
        ),
        np.float32,
    )
    fit_step = max(1, len(pixel_lab) // 2500)
    fit_pixels = np.ascontiguousarray(pixel_lab[::fit_step].astype(np.float32))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.1,
    )
    _, _, centers = cv2.kmeans(
        fit_pixels,
        2,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    distances = np.linalg.norm(pixel_lab[:, None, :].astype(np.float32) - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    dominant_label = int(np.argmax(np.bincount(labels, minlength=2)))
    dominant_pixels = pixels[labels == dominant_label]
    palette_sample = len(dominant_pixels) / len(pixels) >= 0.65
    if len(dominant_pixels) == 0:
        dominant_pixels = pixels
    jersey_bgr = np.clip(np.median(dominant_pixels, axis=0).astype(np.float32), 0, 255)
    return descriptor, jersey_bgr, palette_sample


__all__ = [
    "JerseyTeamClassifier",
    "build_jersey_mask",
    "extract_jersey_color",
    "normalize_team",
]
