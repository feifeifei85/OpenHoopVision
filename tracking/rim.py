from collections import deque
import math
import statistics


class RimTracker:
    def __init__(self, event_parameters):
        self.parameters = event_parameters or {}
        self.samples = deque(maxlen=self.parameters.get('calibration_frames', 100))
        self.box = None
        self.last_seen = -(10**9)
        self.scale = None
        self.drift = 0
        self.current_detection_box = None
        self.drift_samples = deque(maxlen=max(2, int(self.parameters.get('rim_tracker_relock_min_history', 2))))

    @staticmethod
    def _median(boxes):
        return tuple(statistics.median(item[i] for item in boxes) for i in range(4))

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    def update(self, detections, ball, locked_center, frame):
        parameters = self.parameters
        minimum_width_ratio = parameters.get('rim_tracker_min_width_ratio', 0.5)
        maximum_width_ratio = parameters.get('rim_tracker_max_width_ratio', 2.0)
        calibration_min_samples = parameters.get('calibration_min_samples', 10)
        candidates = list(detections or [])
        selected = None
        if candidates:
            if locked_center is not None:
                selected = min(candidates, key=lambda d: math.dist(self._center(d.bbox), locked_center))
            elif ball is not None and not ball.interpolated:
                selected = min(candidates, key=lambda d: math.hypot(d.cx - ball.x, d.cy - ball.y))
            else:
                selected = max(candidates, key=lambda d: d.confidence)

        self.current_detection_box = None if selected is None else selected.bbox
        if selected is None:
            self.drift_samples.clear()
            if frame - self.last_seen <= parameters.get('occlusion_hold_frames', 30):
                return self.box, self.scale, self.scale
            return None, None, self.scale

        new_box = selected.bbox
        reference = (
            self._median(self.samples)
            if len(self.samples) >= calibration_min_samples
            else self.box
        )
        if reference is not None:
            width = max(reference[2] - reference[0], 1.0)
            center_shift = math.dist(self._center(reference), self._center(new_box))
            ratio = (new_box[2] - new_box[0]) / width
            if (
                center_shift
                > max(
                    parameters.get('rim_tracker_center_reset_min_px', 100.0),
                    parameters.get('rim_center_reset_width_ratio', 2.0) * width,
                )
                or not minimum_width_ratio <= ratio <= maximum_width_ratio
            ):
                self.samples.clear()
                self.scale = None
                self.drift = 0
                self.drift_samples.clear()
                reference = None

        if reference is not None:
            width = max(reference[2] - reference[0], 1.0)
            cx, cy = self._center(reference)
            nx, ny = self._center(new_box)
            disagree = (
                abs(nx - cx) > parameters.get('rim_center_disagree_x_width_ratio', 0.75) * width
                or abs(ny - cy) > parameters.get('rim_center_disagree_y_width_ratio', 1.5) * width
                or not minimum_width_ratio
                <= (new_box[2] - new_box[0]) / width
                <= maximum_width_ratio
            )
            if disagree:
                self.drift += 1
                self.drift_samples.append(new_box)
                if len(self.drift_samples) == self.drift_samples.maxlen:
                    drift_boxes = [item for item in self.drift_samples]
                    drift_width = max(
                        statistics.median(box[2] - box[0] for box in drift_boxes),
                        1.0,
                    )
                    centers = [self._center(box) for box in drift_boxes]
                    spread = max(math.dist(left, right) for i, left in enumerate(centers) for right in centers[i + 1 :])
                    if spread <= self.parameters.get('rim_relock_spread_widths', 1.5) * drift_width:
                        self.samples.clear()
                        self.samples.extend(drift_boxes)
                        self.box = self._median(drift_boxes)
                        self.scale = None
                        self.drift = 0
                        self.drift_samples.clear()
                        self.last_seen = int(frame)
                        return self.box, None, self.scale
                if self.drift < self.parameters.get('rim_tracker_drift_frames', 30):
                    self.last_seen = int(frame)
                    return reference, self.scale, self.scale
                self.samples.clear()
                self.scale = None
                self.drift = 0
                self.drift_samples.clear()
            else:
                self.drift = 0
                self.drift_samples.clear()

        self.samples.append(new_box)
        self.last_seen = int(frame)
        self.box = new_box
        scale = None
        if len(self.samples) >= calibration_min_samples:
            scale_box = self._median(self.samples)
            scale = (
                max(scale_box[2] - scale_box[0], 1.0)
                / 0.4572
            )
            self.scale = scale
        return self.box, scale, self.scale
