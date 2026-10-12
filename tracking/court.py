"""Court geometry and player-position mapping for basketball tracking."""

from typing import Any

import cv2
import numpy as np

from tracking.pose import get_player_ground_point, validated_pose


COURT_KEYPOINT_COUNT = 33
COURT_LENGTH_M = 28.65
COURT_WIDTH_M = 15.24
COURT_GEOMETRY = {
    "lane_half_width_m": 2.44,
    "baseline_offset_m": 0.91,
    "baseline_key_distance_m": 4.24,
    "free_throw_line_m": 5.79,
    "free_throw_radius_m": 1.83,
    "basket_offset_m": 1.6,
    "restricted_area_radius_m": 1.22,
    "three_point_radius_m": 7.24,
    "paint_width_m": 4.88,
    "three_point_corner_x_m": 8.35,
    "three_point_center_x_m": 8.84,
    "center_circle_samples": 72,
    "free_throw_circle_samples": 36,
    "three_point_arc_samples": 80,
}


def _arc(center, radius, start, end, samples):
    angles = np.deg2rad(np.linspace(start, end, samples))
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))).astype(
        np.float32
    )


def court_paths() -> tuple[np.ndarray, ...]:
    """Return shared NBA court line paths in metric coordinates."""
    geometry = COURT_GEOMETRY
    length = COURT_LENGTH_M
    width = COURT_WIDTH_M
    center = width * 0.5
    top = center - geometry["lane_half_width_m"]
    bottom = center + geometry["lane_half_width_m"]
    angle = np.degrees(
        np.arctan2(
            center - geometry["baseline_offset_m"], geometry["baseline_key_distance_m"] - geometry["basket_offset_m"]
        )
    )
    paths = [
        np.asarray(((0, 0), (length, 0), (length, width), (0, width), (0, 0)), np.float32),
        np.asarray(((length * 0.5, 0), (length * 0.5, width)), np.float32),
        _arc(
            (length * 0.5, center),
            geometry["free_throw_radius_m"],
            0,
            360,
            geometry["center_circle_samples"],
        ),
    ]
    for right in (False, True):
        x = (lambda value: length - value) if right else (lambda value: value)
        paths.extend(
            (
                np.asarray(
                    (
                        (x(0), top),
                        (x(geometry["free_throw_line_m"]), top),
                        (x(geometry["free_throw_line_m"]), bottom),
                        (x(0), bottom),
                        (x(0), top),
                    ),
                    np.float32,
                ),
                _arc(
                    (x(geometry["free_throw_line_m"]), center),
                    geometry["free_throw_radius_m"],
                    0,
                    360,
                    geometry["free_throw_circle_samples"],
                ),
                np.asarray(
                    (
                        (x(0), geometry["baseline_offset_m"]),
                        (x(geometry["baseline_key_distance_m"]), geometry["baseline_offset_m"]),
                    ),
                    np.float32,
                ),
                np.asarray(
                    (
                        (x(0), width - geometry["baseline_offset_m"]),
                        (x(geometry["baseline_key_distance_m"]), width - geometry["baseline_offset_m"]),
                    ),
                    np.float32,
                ),
                _arc(
                    (x(geometry["basket_offset_m"]), center),
                    geometry["restricted_area_radius_m"],
                    90 if right else -90,
                    270 if right else 90,
                    geometry["free_throw_circle_samples"],
                ),
                _arc(
                    (x(geometry["basket_offset_m"]), center),
                    geometry["three_point_radius_m"],
                    180 - angle if right else -angle,
                    180 + angle if right else angle,
                    geometry["three_point_arc_samples"],
                ),
            )
        )
    return tuple(paths)
COURT_PATHS = court_paths()


def default_nba_court_vertices() -> np.ndarray:
    """Return image-correspondence points in the court model's keypoint order."""
    geometry = COURT_GEOMETRY
    length = COURT_LENGTH_M
    width = COURT_WIDTH_M
    center = width * 0.5
    paint_left = (width - geometry["paint_width_m"]) * 0.5
    paint_right = width - paint_left
    left = np.asarray(
        [
            (0, 0),
            (0, geometry["baseline_offset_m"]),
            (0, paint_left),
            (0, paint_right),
            (0, width - geometry["baseline_offset_m"]),
            (0, width),
            (geometry["baseline_key_distance_m"], geometry["baseline_offset_m"]),
            (geometry["basket_offset_m"], center),
            (geometry["baseline_key_distance_m"], width - geometry["baseline_offset_m"]),
            (geometry["free_throw_line_m"], paint_left),
            (geometry["free_throw_line_m"], center),
            (geometry["free_throw_line_m"], paint_right),
            (geometry["three_point_center_x_m"], center),
            (geometry["three_point_corner_x_m"], 0),
            (geometry["three_point_corner_x_m"], width),
        ],
        dtype=np.float32,
    )
    middle = np.asarray(((length * 0.5, 0), (length * 0.5, center), (length * 0.5, width)), dtype=np.float32)
    right = np.asarray((length, width), dtype=np.float32) - left
    return np.concatenate((left, middle, right), axis=0)


class CourtMapper:
    """Estimate and refine a stable image-to-court homography across frames."""

    def __init__(self, vertices: np.ndarray, max_pending_gap: int | None = None) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        if self.vertices.shape != (COURT_KEYPOINT_COUNT, 2):
            raise ValueError(
                f"expected {COURT_KEYPOINT_COUNT} court vertices, got {self.vertices.shape}"
            )
        if max_pending_gap is None:
            max_pending_gap = 1
        self.max_pending_gap = max(1, int(max_pending_gap))
        self.homography = None
        self.last_valid_frame = -(10**9)
        self.last_stats = self._stats("uninitialized")
        self.court_min = self.vertices.min(axis=0)
        self.court_max = self.vertices.max(axis=0)
        anchors = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.5], [0.75, 0.5]],
            np.float32,
        )
        self.anchor_points = anchors * np.asarray((COURT_LENGTH_M, COURT_WIDTH_M), np.float32)
        self.pending = None
        self.line_warp = np.eye(3, dtype=np.float64)
        self.last_refine_frame = -(10**9)

    def reset(self) -> None:
        self.homography = None
        self.last_valid_frame = -(10**9)
        self.last_stats = self._stats("uninitialized")
        self.pending = None
        self.line_warp = np.eye(3, dtype=np.float64)
        self.last_refine_frame = -(10**9)

    @staticmethod
    def _stats(
        reason,
        valid=0,
        inliers=0,
        error=None,
        reacquired=False,
    ):
        return {
            "valid": int(valid),
            "inliers": int(inliers),
            "median_error": error,
            "reason": reason,
            "reacquired": bool(reacquired),
        }

    def _reject(self, reason, **updates):
        self.last_stats["reason"] = reason
        self.last_stats.update(updates)
        return False

    @staticmethod
    def _project_court_to_image(homography, points):
        ok, inverse = cv2.invert(np.asarray(homography, dtype=np.float64))
        if not ok or not np.isfinite(inverse).all():
            return None
        projected = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
        return projected if np.isfinite(projected).all() else None

    @staticmethod
    def _is_direction_flip(current_anchors, previous_anchors):
        direct_shift = float(np.median(np.linalg.norm(current_anchors - previous_anchors, axis=1)))
        orders = [[1, 0, 3, 2, 4, 6, 5], [2, 3, 0, 1, 4, 6, 5]]
        flipped_shift = min(
            float(np.median(np.linalg.norm(current_anchors - previous_anchors[list(order)], axis=1)))
            for order in orders
        )
        return flipped_shift + 25.0 < direct_shift

    def _confirm_reacquire(self, anchors, frame_index, frame_shape):
        confirmations = 1
        if self.pending is not None:
            previous, previous_frame, previous_count = self.pending
            shift = np.linalg.norm(np.asarray(anchors) - previous, axis=1)
            scale = max(frame_shape[:2])
            consistent = frame_index - previous_frame <= self.max_pending_gap and np.isfinite(shift).all()
            consistent &= np.median(shift) <= max(
                32.0,
                0.04 * scale,
            )
            consistent &= np.percentile(shift, 90) <= max(
                80.0,
                0.1 * scale,
            )
            confirmations = previous_count + 1 if consistent else 1
        self.pending = (np.asarray(anchors, np.float32).copy(), int(frame_index), confirmations)
        confirmed = confirmations >= 3
        if confirmed:
            self.pending = None
        return confirmed

    def update(
        self,
        keypoints: np.ndarray | None,
        scores: np.ndarray | None,
        frame_index: int,
        frame_shape: tuple[int, ...],
        continuity_data: tuple[Any, ...] | list[Any] = (),
    ) -> bool:
        """Update the homography from one court-keypoint observation."""
        frame_index = int(frame_index)
        self.last_stats = self._stats("no_input")
        if keypoints is None or scores is None:
            return False
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if (
            keypoints.shape != (COURT_KEYPOINT_COUNT, 2)
            or scores.shape != (COURT_KEYPOINT_COUNT,)
        ):
            return self._reject("invalid_input_shape")
        height, width = frame_shape[:2]
        valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores)
        valid &= scores >= 0.3
        valid &= (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width)
        valid &= (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height)
        image_points = keypoints[valid]
        court_points = self.vertices[valid]
        self.last_stats = self._stats("too_few_points", len(image_points))
        if len(image_points) < 6:
            return False
        frame_area = max(1.0, float(width * height))
        if (
            abs(float(cv2.contourArea(cv2.convexHull(image_points))))
            < 0.001 * frame_area
        ):
            return self._reject("degenerate_image_layout")
        cv2.setRNGSeed(42)
        homography, mask = cv2.findHomography(
            image_points, court_points, cv2.RANSAC, 0.5
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            return self._reject("homography_failed")
        inliers = mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if (
            inlier_count < 6
            or inlier_count / len(image_points) < 0.7
        ):
            return self._reject("insufficient_consensus", inliers=inlier_count)
        inlier_points = image_points[inliers]
        spread = np.linalg.svd(inlier_points - inlier_points.mean(axis=0), compute_uv=False)
        inlier_area = abs(float(cv2.contourArea(cv2.convexHull(inlier_points))))
        if (
            inlier_area < 0.0005 * frame_area
            or spread[0] <= 1e-06
            or spread[1] / spread[0] < 0.025
        ):
            return self._reject("degenerate_inlier_layout")
        projected = cv2.perspectiveTransform(inlier_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(projected - court_points[inliers], axis=1)))
        self.last_stats = self._stats("candidate", len(image_points), inlier_count, error)
        if not np.isfinite(error) or error > 0.5:
            return self._reject("large_reprojection_error")
        if abs(float(homography[2, 2])) > 1e-09:
            homography /= homography[2, 2]
        anchors = self._project_court_to_image(homography, self.anchor_points)
        if anchors is None or np.any(
            np.abs(anchors) > 8.0 * max(height, width)
        ):
            return self._reject("unstable_anchor_projection")
        frame_gap = frame_index - self.last_valid_frame
        reacquired = self.homography is None or frame_gap > 5
        previous = (
            None if self.homography is None else self._project_court_to_image(self.homography, self.anchor_points)
        )
        if self.homography is not None and previous is None:
            self.pending = None
            return self._reject("invalid_previous_anchor_projection")
        if previous is not None and not reacquired and self._is_direction_flip(anchors, previous):
            self.pending = None
            return self._reject("direction_flip")
        if previous is not None and 0 <= frame_gap <= 5:
            shift = np.linalg.norm(anchors - previous, axis=1)
            limit = 25.0 * max(1, frame_gap)
            if (
                not np.isfinite(shift).all()
                or np.median(shift) > limit
                or np.percentile(shift, 90)
                > max(
                    4.0 * limit,
                    100.0,
                )
            ):
                self.pending = None
                return self._reject("temporal_jump")
        if reacquired and len(continuity_data) >= 3:
            continuity = np.asarray(continuity_data, dtype=np.float32).reshape(-1, 5)
            mapped = cv2.perspectiveTransform(continuity[:, :2].reshape(-1, 1, 2), homography).reshape(-1, 2)
            ratio = np.linalg.norm(mapped - continuity[:, 2:4], axis=1) / np.maximum(
                continuity[:, 4], 0.001
            )
            if (
                not np.isfinite(ratio).all()
                or np.median(ratio) > 1.0
            ):
                self.pending = None
                return self._reject("player_continuity")
        if (
            reacquired
            and self.homography is not None
            and not self._confirm_reacquire(anchors, frame_index, frame_shape)
        ):
            return self._reject("awaiting_reacquire")
        if not reacquired or self.homography is None:
            self.pending = None
        if reacquired:
            self.line_warp = np.eye(3, dtype=np.float64)
        homography = np.linalg.inv(self.line_warp @ np.linalg.inv(homography))
        homography /= homography[2, 2]
        self.homography = homography.astype(np.float64)
        self.last_valid_frame = frame_index
        self.last_stats.update(reason="accepted", reacquired=bool(reacquired))
        return True

    def transform(
        self,
        point: np.ndarray | None,
        frame_index: int,
        allow_stale: bool = False,
        clip: bool = True,
    ) -> np.ndarray | None:
        """Map an image point to court meters, optionally accepting stale maps."""
        if point is None or self.homography is None:
            return None
        max_age = (
            30
            if allow_stale
            else 5
        )
        age = int(frame_index) - self.last_valid_frame
        if age < 0 or age > max_age:
            return None
        point = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        if not np.isfinite(point).all():
            return None
        mapped = cv2.perspectiveTransform(point, self.homography).reshape(2)
        lower = self.court_min - 2.0
        upper = self.court_max + 2.0
        if not np.isfinite(mapped).all() or np.any(mapped < lower) or np.any(mapped > upper):
            return None
        if clip:
            mapped = np.clip(mapped, self.court_min, self.court_max)
        return mapped.astype(np.float32)

    def refine(self, frame: np.ndarray, frame_index: int, occluders: tuple[Any, ...] | list[Any] = ()) -> bool:
        """Refine the current map by aligning projected lines with image edges."""
        return _refine_court_mapper(self, frame, frame_index, occluders)


def _refine_court_mapper(mapper, frame, frame_index, occluders=()):
    radius = 16
    frame_index = int(frame_index)
    if (
        mapper.homography is None
        or mapper.last_valid_frame != frame_index
        or (
            not mapper.last_stats.get("reacquired", False)
            and frame_index - mapper.last_refine_frame < 15
        )
    ):
        return False
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        320.0 / max(height, width),
    )
    min_dimension = 1
    size = (max(min_dimension, round(width * scale)), max(min_dimension, round(height * scale)))
    base_h = mapper.homography.copy()
    template = np.zeros((size[1], size[0]), dtype=np.uint8)
    projected_paths = [mapper._project_court_to_image(base_h, path) for path in COURT_PATHS]
    template_paths = [np.rint(points * scale).astype(np.int32) for points in projected_paths if points is not None]
    cv2.polylines(
        template,
        template_paths,
        False,
        255,
        2,
        cv2.LINE_AA,
    )
    band = cv2.dilate(
        template,
        np.ones(
            (max(3, round(radius * scale) * 2 + 1),) * 2,
            np.uint8,
        ),
    )
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    blur_kernel = 5
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (blur_kernel, blur_kernel), 0)
    median = float(np.median(gray))
    for box in occluders:
        x1, y1, x2, y2 = np.rint(np.asarray(box) * scale).astype(int)
        pad = 2
        cv2.rectangle(band, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
    edges = cv2.bitwise_and(
        cv2.Canny(
            gray,
            max(
                20,
                int(0.66 * median),
            ),
            min(
                255,
                int(1.33 * median),
            ),
        ),
        band,
    )
    if cv2.countNonZero(edges) < 60:
        return False
    template = cv2.GaussianBlur(template, (0, 0), 2.0)
    edges = cv2.GaussianBlur(edges, (0, 0), 2.0)
    warp = np.eye(3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        35,
        0.0001,
    )
    try:
        score, warp = cv2.findTransformECC(
            template,
            edges,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            band,
            5,
        )
    except cv2.error:
        return False
    if not np.isfinite(warp).all() or score < 0.18:
        return False
    scale_matrix = np.diag((scale, scale, 1.0))
    warp_full = np.linalg.inv(scale_matrix) @ warp.astype(np.float64) @ scale_matrix
    candidate_h = np.linalg.inv(warp_full @ np.linalg.inv(base_h))
    candidate_h /= candidate_h[2, 2]
    old_anchors = mapper._project_court_to_image(
        np.linalg.inv(np.linalg.inv(mapper.line_warp) @ np.linalg.inv(base_h)), mapper.anchor_points
    )
    new_anchors = mapper._project_court_to_image(candidate_h, mapper.anchor_points)
    if old_anchors is None or new_anchors is None:
        return False
    shift = np.linalg.norm(new_anchors - old_anchors, axis=1)
    max_shift = max(
        6.0,
        0.012 * max(height, width),
    )
    if (
        float(np.median(shift)) > max_shift
        or float(np.percentile(shift, 90))
        > 2.5 * max_shift
    ):
        return False
    mapper.homography = candidate_h.astype(np.float64)
    mapper.line_warp = warp_full @ mapper.line_warp
    mapper.last_refine_frame = frame_index
    mapper.last_stats.update(line_score=float(score), refined=True)
    return True


def update_court_state(
    track: Any,
    state: Any,
    court_mapper: CourtMapper,
    frame_index: int,
    fps: float,
    pose: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Smooth a local track's court position while rejecting implausible jumps."""
    observation = court_mapper.transform(get_player_ground_point(track, pose), frame_index, allow_stale=True)
    if observation is None:
        return
    if state.court_observation is None or state.court_position is None:
        position = observation.copy()
    else:
        elapsed = max((frame_index - state.court_frame) / fps, 1e-06)
        if float(np.linalg.norm(observation - state.court_observation) / elapsed) > 12.0:
            return
        position = (
            observation.copy()
            if elapsed > 1.0
            else 0.35 * observation
            + (1 - 0.35) * state.court_position
        )
    state.court_observation = observation.copy()
    state.court_position = position.astype(np.float32)
    state.court_frame = int(frame_index)


def is_player_on_court(
    track: Any,
    pose: tuple[np.ndarray, np.ndarray] | None,
    mapper: CourtMapper,
    frame_index: int,
    frame_shape: tuple[int, ...],
    min_fallback_height: float = 100,
) -> bool:
    """Return whether a local player track is eligible for on-court processing."""
    pose_data = None if pose is None else validated_pose(pose[0], pose[1], track.box)
    ankle_visible = pose_data is not None and bool(np.any(pose_data[2][15:17]))
    height = float(track.box[3] - track.box[1])
    if not ankle_visible and height < float(min_fallback_height):
        return False
    if mapper.homography is not None and int(frame_index) - mapper.last_valid_frame <= 30:
        court_point = mapper.transform(
            get_player_ground_point(track, pose),
            frame_index,
            allow_stale=True,
            clip=False,
        )
        margin = 0.75
        return court_point is not None and bool(
            np.all(court_point >= mapper.court_min - margin)
            and np.all(court_point <= mapper.court_max + margin)
        )
    x1, _, x2, y2 = map(float, track.box)
    height_px, width_px = frame_shape[:2]
    return (
        width_px * 0.11 <= (x1 + x2) * 0.5 <= width_px * 0.96
        and y2 >= height_px * 0.58
        and y2 <= height_px * 0.96
    )


__all__ = [
    "COURT_GEOMETRY",
    "COURT_KEYPOINT_COUNT",
    "COURT_LENGTH_M",
    "COURT_PATHS",
    "COURT_WIDTH_M",
    "CourtMapper",
    "court_paths",
    "default_nba_court_vertices",
    "is_player_on_court",
    "update_court_state",
]
