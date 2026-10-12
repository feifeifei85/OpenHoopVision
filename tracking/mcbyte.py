"""McByte++ tracking primitives adapted from McBytePlusPlus (Apache-2.0).

The tracker keeps ByteTrack's high/low confidence association, adds McByte's
clear-match locking, ORB camera motion compensation, and pose-guided masks.
Global identity and ReID are handled by :mod:`reid.resolver`.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from tracking.pose import validated_pose

try:
    import lap
except ImportError:  # pragma: no cover - SciPy fallback for minimal installs
    lap = None


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class KalmanFilter:
    """ByteTrack's constant-velocity Kalman filter for ``xywh`` boxes."""

    def __init__(self):
        self.motion = np.eye(8, dtype=np.float32)
        self.motion[:4, 4:] = np.eye(4, dtype=np.float32)
        self.measurement = np.eye(4, 8, dtype=np.float32)
        self.position_weight = 1.0 / 20
        self.velocity_weight = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        w, h = measurement[2:]
        std = np.asarray((w, h, w, h, w, h, w, h), dtype=np.float32)
        std[:4] *= 2 * self.position_weight
        std[4:] *= 10 * self.velocity_weight
        return mean, np.diag(std * std)

    def predict(self, mean, covariance):
        w, h = mean[2:4]
        std = np.asarray((w, h, w, h, w, h, w, h), dtype=np.float32)
        std[:4] *= self.position_weight
        std[4:] *= self.velocity_weight
        mean = self.motion @ mean
        covariance = self.motion @ covariance @ self.motion.T + np.diag(std * std)
        return mean, covariance

    def multi_predict(self, means, covariances):
        if len(means) == 0:
            return means, covariances
        sizes = means[:, 2:4]
        std = np.concatenate((sizes, sizes, sizes, sizes), axis=1).astype(np.float32)
        std[:, :4] *= self.position_weight
        std[:, 4:] *= self.velocity_weight
        covariances = (
            self.motion @ covariances @ self.motion.T
            + np.eye(8, dtype=np.float32)[None] * (std * std)[:, :, None]
        )
        return means @ self.motion.T, covariances

    def update(self, mean, covariance, measurement):
        projected_mean = self.measurement @ mean
        projected_covariance = self.measurement @ covariance @ self.measurement.T
        w, h = mean[2:4]
        std = np.asarray((w, h, w, h), dtype=np.float32) * self.position_weight
        projected_covariance += np.diag(std * std)
        cross_covariance = covariance @ self.measurement.T
        gain = np.linalg.solve(projected_covariance, cross_covariance.T).T
        innovation = measurement - projected_mean
        mean = mean + gain @ innovation
        covariance = covariance - gain @ projected_covariance @ gain.T
        return mean, covariance


def _iou(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    area_a = np.prod(np.maximum(0.0, boxes_a[:, 2:] - boxes_a[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0.0, boxes_b[:, 2:] - boxes_b[:, :2]), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-8)


class IoU:
    """Small NumPy IoU adapter used by pose matching and holder selection."""

    def compute(self, boxes_a, boxes_b):
        return _iou(boxes_a, boxes_b)


def _linear_assignment(cost, threshold):
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return np.empty((0, 2), dtype=int), tuple(range(rows)), tuple(range(columns))
    cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6).astype(np.float64)
    if lap is not None:
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=float(threshold))
        matches = np.asarray([(i, j) for i, j in enumerate(x) if j >= 0], dtype=int).reshape(-1, 2)
        return matches, np.flatnonzero(x < 0), np.flatnonzero(y < 0)

    from scipy.optimize import linear_sum_assignment

    penalty = float(threshold) / 2 + 1e-7
    size = rows + columns
    padded = np.full((size, size), 1e6, dtype=np.float64)
    valid = cost <= threshold
    padded[:rows, :columns] = np.where(valid, cost, 1e6)
    padded[:rows, columns:] = penalty
    padded[rows:, :columns] = penalty
    padded[rows:, columns:] = 0.0
    assigned_rows, assigned_columns = linear_sum_assignment(padded)
    matches = [
        (i, j)
        for i, j in zip(assigned_rows, assigned_columns)
        if i < rows and j < columns and valid[i, j]
    ]
    matched_rows = {i for i, _ in matches}
    matched_columns = {j for _, j in matches}
    return (
        np.asarray(matches, dtype=int).reshape(-1, 2),
        tuple(i for i in range(rows) if i not in matched_rows),
        tuple(j for j in range(columns) if j not in matched_columns),
    )


class ORBCameraMotion:
    """McByte++ ORB camera-motion estimate with conservative affine checks."""

    def __init__(self, downscale=4):
        self.downscale = max(1, int(downscale))
        self.detector = cv2.FastFeatureDetector_create(20)
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous_points = None
        self.previous_descriptors = None
        self.applied = 0
        self.rejected = 0
        self.skipped = 0

    def estimate(self, frame, detections):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (width // self.downscale, height // self.downscale))
        small_h, small_w = gray.shape
        mask = np.zeros_like(gray)
        mask[int(0.02 * small_h) : int(0.98 * small_h), int(0.02 * small_w) : int(0.98 * small_w)] = 255
        for box in detections:
            x1, y1, x2, y2 = (np.asarray(box[:4]) / self.downscale).astype(int)
            x1, x2 = np.clip((x1, x2), 0, small_w)
            y1, y2 = np.clip((y1, y2), 0, small_h)
            mask[y1:y2, x1:x2] = 0
        keypoints = self.detector.detect(gray, mask)
        keypoints, descriptors = self.extractor.compute(gray, keypoints)
        identity = np.eye(2, 3, dtype=np.float32)
        warp = identity
        if self.previous_descriptors is not None and descriptors is not None:
            matches = []
            displacements = []
            max_distance = 0.25 * np.asarray((small_w, small_h), dtype=np.float32)
            for pair in self.matcher.knnMatch(self.previous_descriptors, descriptors, 2):
                if len(pair) != 2 or pair[0].distance >= 0.9 * pair[1].distance:
                    continue
                match = pair[0]
                old = self.previous_points[match.queryIdx].pt
                new = keypoints[match.trainIdx].pt
                delta = np.asarray(old) - np.asarray(new)
                if np.all(np.abs(delta) < max_distance):
                    matches.append(match)
                    displacements.append(delta)
            if len(matches) >= 12:
                displacements = np.asarray(displacements, dtype=np.float32)
                center = np.mean(displacements, axis=0)
                spread = np.std(displacements, axis=0) + 1e-6
                inliers = np.all(np.abs(displacements - center) < 2.5 * spread, axis=1)
                old_points = np.asarray(
                    [self.previous_points[m.queryIdx].pt for m, keep in zip(matches, inliers) if keep],
                    dtype=np.float32,
                )
                new_points = np.asarray(
                    [keypoints[m.trainIdx].pt for m, keep in zip(matches, inliers) if keep],
                    dtype=np.float32,
                )
                if len(old_points) > 4:
                    estimated, ransac_inliers = cv2.estimateAffinePartial2D(
                        old_points, new_points, method=cv2.RANSAC
                    )
                    if estimated is not None and ransac_inliers is not None and int(ransac_inliers.sum()) >= 5:
                        estimated = np.asarray(estimated, dtype=np.float32)
                        estimated[:, 2] *= self.downscale
                        scale_x = float(np.linalg.norm(estimated[:2, 0]))
                        scale_y = float(np.linalg.norm(estimated[:2, 1]))
                        safe = (
                            np.isfinite(estimated).all()
                            and 0.3 <= scale_x <= 3.0
                            and 0.3 <= scale_y <= 3.0
                            and abs(float(estimated[0, 2])) <= 0.25 * width
                            and abs(float(estimated[1, 2])) <= 0.25 * height
                        )
                        if safe:
                            warp = estimated
                        else:
                            self.rejected += 1
                    else:
                        self.rejected += 1
                else:
                    self.rejected += 1
            else:
                self.skipped += 1
        elif self.previous_descriptors is not None:
            self.skipped += 1

        if np.any(warp != identity):
            self.applied += 1
        self.previous_points = keypoints
        self.previous_descriptors = descriptors
        return warp


@dataclass(slots=True)
class Tracklet:
    """One short-lived ByteTrack tracklet."""

    box: np.ndarray
    score: float
    track_id: int = 0
    state: int = TrackState.NEW
    activated: bool = False
    source_index: int = -1
    frame_id: int = 0
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    kalman_filter: KalmanFilter | None = None

    _next_id = 0

    @classmethod
    def next_id(cls):
        cls._next_id += 1
        return cls._next_id

    @property
    def end_frame(self):
        return self.frame_id

    @property
    def tlwh(self):
        if self.mean is None:
            return self.box.copy()
        result = self.mean[:4].copy()
        result[:2] -= result[2:] / 2
        return result

    @property
    def tracker_id(self):
        return self.track_id

    @property
    def xyxy(self):
        return self.tlbr

    @property
    def tlbr(self):
        box = self.tlwh
        box[2:] += box[:2]
        return box

    @property
    def xywh(self):
        return np.r_[self.tlwh[:2] + self.tlwh[2:] / 2, self.tlwh[2:]]

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.xywh)
        self.state = TrackState.TRACKED
        self.activated = frame_id == 1
        self.frame_id = int(frame_id)

    def update(self, detection, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, detection.xywh
        )
        self.frame_id = int(frame_id)
        self.state = TrackState.TRACKED
        self.activated = True
        self.score = detection.score

    def re_activate(self, detection, frame_id):
        self.update(detection, frame_id)

    def predict(self):
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[6:] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    @staticmethod
    def multi_predict(tracklets):
        if not tracklets:
            return
        means = np.asarray([track.mean.copy() for track in tracklets])
        covariances = np.asarray([track.covariance for track in tracklets])
        for index, track in enumerate(tracklets):
            if track.state != TrackState.TRACKED:
                means[index, 6:] = 0
        means, covariances = tracklets[0].kalman_filter.multi_predict(means, covariances)
        for track, mean, covariance in zip(tracklets, means, covariances):
            track.mean, track.covariance = mean, covariance

    @staticmethod
    def multi_gmc(tracklets, warp):
        if not tracklets:
            return
        linear = warp[:, :2]
        matrix = np.kron(np.eye(4, dtype=np.float32), linear)
        translation = warp[:, 2]
        for track in tracklets:
            track.mean = matrix @ track.mean
            track.mean[:2] += translation
            track.covariance = matrix @ track.covariance @ matrix.T

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED


def _joint(first, second):
    by_id = {track.track_id: track for track in first}
    by_id.update((track.track_id, track) for track in second)
    return list(by_id.values())


def _subtract(first, second):
    removed = {track.track_id for track in second}
    return [track for track in first if track.track_id not in removed]


class McByteTracker:
    """Local implementation of McByte++ short-term tracking.

    ReID remains in ``IdentityResolver`` so local track lifecycle and persistent
    global identity stay independent.
    """

    def __init__(
        self,
        lost_track_buffer=30,
        frame_rate=30.0,
        track_activation_threshold=0.4,
        high_conf_det_threshold=0.3,
        cmc_method="orb",
        cmc_downscale=4,
        **_,
    ):
        self.max_time_lost = max(1, round(float(frame_rate) / 30 * int(lost_track_buffer)))
        self.new_track_threshold = float(track_activation_threshold) + 0.1
        self.high_conf_threshold = float(high_conf_det_threshold)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.cmc = GuardedCMC(ORBCameraMotion(cmc_downscale)) if cmc_method != "none" else None
        self.pose_mask_detections = 0
        self.pose_mask_pairs_scored = 0
        Tracklet._next_id = 0

    @property
    def tracked_objects(self):
        return self._as_detections(self.tracked_stracks + self.lost_stracks)

    @staticmethod
    def _as_detections(tracklets):
        if not tracklets:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.asarray([track.tlbr for track in tracklets], dtype=np.float32),
            confidence=np.asarray([track.score for track in tracklets], dtype=np.float32),
            tracker_id=np.asarray([track.track_id for track in tracklets], dtype=np.int64),
        )

    @staticmethod
    def _detection_tracks(detections):
        if detections is None or len(detections) == 0:
            return []
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = (
            np.ones(len(boxes), dtype=np.float32)
            if detections.confidence is None
            else np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
        )
        return [
            Tracklet(
                np.r_[box[:2], box[2:] - box[:2]].astype(np.float32),
                float(score),
                source_index=index,
            )
            for index, (box, score) in enumerate(zip(boxes, scores))
            if np.isfinite(box).all() and np.isfinite(score)
        ]

    @staticmethod
    def _cost(tracks, detections, fuse_score=False):
        cost = 1.0 - _iou([track.tlbr for track in tracks], [det.tlbr for det in detections])
        if fuse_score and cost.size:
            score = np.asarray([det.score for det in detections], dtype=np.float32)[None, :]
            cost = 1.0 - (1.0 - cost) * score
        return cost

    @staticmethod
    def _pose_mask(frame_shape, detection, pose_data):
        if pose_data is None:
            return None
        points, scores = pose_data
        index = detection.source_index
        if index < 0 or index >= len(points) or index >= len(scores):
            return None
        box = detection.tlbr
        pose = validated_pose(points[index], scores[index], box, frame_shape=frame_shape)
        if pose is None:
            return None
        points, _, valid = pose
        if len(points) < 17 or len(valid) < 17:
            return None
        torso = np.asarray((5, 6, 11, 12), dtype=np.int64)
        visible_torso = torso[valid[torso]]
        if len(visible_torso) < 3:
            return None
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width)
        y1, y2 = np.clip((y1, y2), 0, height)
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        local = points[:, :2] - np.asarray((x1, y1), dtype=np.float32)
        polygon = local[visible_torso]
        center = polygon.mean(axis=0)
        polygon = center + 1.12 * (polygon - center)
        polygon = np.clip(polygon, (0, 0), (mask.shape[1] - 1, mask.shape[0] - 1))
        cv2.fillConvexPoly(mask, np.rint(cv2.convexHull(polygon.astype(np.float32))).astype(np.int32), 255)
        box_width = max(1, x2 - x1)
        for first, second, radius in (
            (5, 7, 0.055), (7, 9, 0.045), (6, 8, 0.055), (8, 10, 0.045),
            (11, 13, 0.075), (13, 15, 0.06), (12, 14, 0.075), (14, 16, 0.06),
        ):
            if second < len(valid) and valid[first] and valid[second]:
                cv2.line(
                    mask,
                    tuple(np.rint(local[first]).astype(int)),
                    tuple(np.rint(local[second]).astype(int)),
                    255,
                    max(2, round(box_width * radius * 2)),
                )
        kernel = max(3, round(box_width * 0.035))
        kernel += 1 - kernel % 2
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
        if cv2.countNonZero(mask) < max(30, int(0.03 * mask.size)):
            return None
        return mask, (x1, y1, x2, y2)

    @staticmethod
    def _mask_coverage(mask_data, box):
        if mask_data is None:
            return 0.0
        mask, (rx1, ry1, rx2, ry2) = mask_data
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip((x1, x2), rx1, rx2)
        y1, y2 = np.clip((y1, y2), ry1, ry2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        total = int(cv2.countNonZero(mask))
        if total == 0:
            return 0.0
        inside = int(cv2.countNonZero(mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1]))
        return inside / total

    def _conditioned_assignment(
        self,
        cost,
        threshold,
        tracks,
        detections,
        *,
        frame=None,
        pose_data=None,
    ):
        if cost.size == 0:
            return _linear_assignment(cost, threshold)
        adjusted = cost.copy()
        possible = cost <= threshold
        row_counts = possible.sum(axis=1)
        column_counts = possible.sum(axis=0)
        near_ties = np.zeros_like(possible)
        pose_masks = {}
        if frame is not None and pose_data is not None:
            tie_margin = 0.02
            for row in range(len(tracks)):
                candidates = np.flatnonzero(possible[row])
                if len(candidates) > 1:
                    order = candidates[np.argsort(cost[row, candidates])]
                    if cost[row, order[1]] - cost[row, order[0]] <= tie_margin:
                        near_ties[row, order[:2]] = True
            for column in range(len(detections)):
                candidates = np.flatnonzero(possible[:, column])
                if len(candidates) > 1:
                    order = candidates[np.argsort(cost[candidates, column])]
                    if cost[order[1], column] - cost[order[0], column] <= tie_margin:
                        near_ties[order[:2], column] = True
            for column in np.flatnonzero(near_ties.any(axis=0)):
                pose_masks[column] = self._pose_mask(
                    frame.shape, detections[column], pose_data
                )
                if pose_masks[column] is not None:
                    self.pose_mask_detections += 1
        for row, column in zip(*np.where(possible)):
            if row_counts[row] == 1 and column_counts[column] == 1:
                adjusted[row, :] = threshold + 10
                adjusted[:, column] = threshold + 10
                adjusted[row, column] = cost[row, column]
                continue
            if not near_ties[row, column]:
                continue
            coverage = self._mask_coverage(pose_masks.get(column), tracks[row].tlbr)
            if coverage >= 0.55:
                adjusted[row, column] -= 0.2 * coverage
                self.pose_mask_pairs_scored += 1
        return _linear_assignment(adjusted, threshold)

    def update(self, detections, frame=None, pose_data=None):
        self.frame_id += 1
        high, low = [], []
        for detection in self._detection_tracks(detections):
            if detection.score > self.high_conf_threshold:
                high.append(detection)
            elif detection.score > 0.1:
                low.append(detection)

        unconfirmed = [track for track in self.tracked_stracks if not track.activated]
        confirmed = [track for track in self.tracked_stracks if track.activated]
        pool = _joint(confirmed, self.lost_stracks)
        Tracklet.multi_predict(pool)

        if self.cmc is not None and frame is not None:
            warp = self.cmc.estimate(frame, [track.tlbr for track in high])
            Tracklet.multi_gmc(pool, warp)
            Tracklet.multi_gmc(unconfirmed, warp)

        cost = self._cost(pool, high, fuse_score=True)
        matches, unmatched_tracks, unmatched_high = self._conditioned_assignment(
            cost, 0.9, pool, high, frame=frame, pose_data=pose_data,
        )
        activated, refound, newly_lost, removed = [], [], [], []
        for track_index, detection_index in matches:
            track, detection = pool[track_index], high[detection_index]
            if track.state == TrackState.TRACKED:
                track.update(detection, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(detection, self.frame_id)
                refound.append(track)

        remaining_tracked = [
            pool[index]
            for index in unmatched_tracks
            if pool[index].state == TrackState.TRACKED
        ]
        cost = self._cost(remaining_tracked, low)
        matches, unmatched_tracks, _ = self._conditioned_assignment(
            cost, 0.5, remaining_tracked, low, frame=frame, pose_data=pose_data,
        )
        for track_index, detection_index in matches:
            track = remaining_tracked[track_index]
            track.update(low[detection_index], self.frame_id)
            activated.append(track)
        for track_index in unmatched_tracks:
            track = remaining_tracked[track_index]
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)

        remaining_detections = [high[index] for index in unmatched_high]
        cost = self._cost(unconfirmed, remaining_detections, fuse_score=True)
        matches, unmatched_unconfirmed, unmatched_detections = self._conditioned_assignment(
            cost, 0.7, unconfirmed, remaining_detections, frame=frame, pose_data=pose_data,
        )
        for track_index, detection_index in matches:
            track = unconfirmed[track_index]
            track.update(remaining_detections[detection_index], self.frame_id)
            activated.append(track)
        for track_index in unmatched_unconfirmed:
            track = unconfirmed[track_index]
            track.mark_removed()
            removed.append(track)

        for detection_index in unmatched_detections:
            detection = remaining_detections[detection_index]
            if detection.score >= self.new_track_threshold:
                detection.activate(self.kalman_filter, self.frame_id)
                activated.append(detection)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        self.tracked_stracks = _joint(self.tracked_stracks, activated)
        self.tracked_stracks = _joint(self.tracked_stracks, refound)
        self.lost_stracks = _subtract(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = _joint(self.lost_stracks, newly_lost)
        self.lost_stracks = _subtract(self.lost_stracks, removed)
        self.removed_stracks = _joint(self.removed_stracks, removed)
        visible = [track for track in self.tracked_stracks if track.activated]
        return self._as_detections(visible)


class GuardedCMC:
    """Track CMC health counters and reject unsafe affine warps."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.interval = 2
        self.frame_count = 0
        self.applied = 0
        self.rejected = 0
        self.skipped = 0

    def estimate(self, frame, detections=None):
        self.frame_count += 1
        if self.frame_count % self.interval == 0:
            self.skipped += 1
            return np.eye(2, 3, dtype=np.float32)
        try:
            warp = self.estimator.estimate(frame, () if detections is None else detections)
        except Exception:
            self.rejected += 1
            return np.eye(2, 3, dtype=np.float32)
        if warp is None or np.asarray(warp).shape != (2, 3) or not np.isfinite(warp).all():
            self.rejected += 1
            return np.eye(2, 3, dtype=np.float32)
        warp = np.asarray(warp, dtype=np.float32)
        scale = np.linalg.norm(warp[:, :2], axis=0)
        if np.any(scale < 0.3) or np.any(scale > 3.0):
            self.rejected += 1
            return np.eye(2, 3, dtype=np.float32)
        if np.any(warp[:, 2] != 0):
            self.applied += 1
        else:
            self.skipped += 1
        return warp
